import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import warnings

# Suppress numpy deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=640,
    framerate=30,
    flip_method=0
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# COCO class names
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

class TensorRTDetector:
    def __init__(self, engine_path, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        
        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        
        # Load engine
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Get input and output shapes
        self.input_binding_idx = self.engine.get_binding_index('images')  # Default YOLOv8 input name
        if self.input_binding_idx == -1:
            # Try alternative input name if 'images' isn't found
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    self.input_binding_idx = i
                    break
        
        # Get the shape of the input binding
        self.input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        # Handle dynamic batch dimension
        if self.input_shape[0] == -1:
            self.input_shape = (1, self.input_shape[1], self.input_shape[2], self.input_shape[3])
            self.context.set_binding_shape(self.input_binding_idx, self.input_shape)
        
        self.batch_size, self.channels, self.height, self.width = self.input_shape
        
        # Create buffers for input and output
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        # Allocate memory for inputs and outputs
        for i in range(self.engine.num_bindings):
            binding_shape = self.engine.get_binding_shape(i)
            if binding_shape[0] == -1:  # Handle dynamic batch
                binding_shape = (1,) + binding_shape[1:]
                if self.engine.binding_is_input(i):
                    self.context.set_binding_shape(i, binding_shape)
            
            size = trt.volume(binding_shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Add to bindings
            self.bindings.append(int(device_mem))
            
            # Keep track of inputs and outputs
            if self.engine.binding_is_input(i):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape})
        
        print(f"Detector initialized with input shape: {self.input_shape}")
    
    def preprocess(self, image):
        """Preprocess image for inference - optimized for speed"""
        # Resize - use INTER_NEAREST for speed
        input_img = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # Preallocate arrays to avoid repeated memory allocations
        if not hasattr(self, 'input_buffer'):
            # Initialize preprocessing buffers
            self.input_buffer = np.zeros((self.batch_size, self.channels, self.height, self.width), dtype=np.float32)
        
        # Convert to RGB and normalize using vectorized operations
        # Avoid creating intermediate arrays
        input_img = input_img[:, :, ::-1]  # BGR to RGB (faster than cvtColor)
        np.divide(input_img, 255.0, out=input_img, dtype=np.float32)
        
        # Transpose directly into preallocated buffer (CHW format)
        for c in range(3):
            self.input_buffer[0, c] = input_img[:, :, c]
        
        return self.input_buffer
    
    def detect(self, image):
        """Run inference on an image and return detections"""
        # Preprocess the image
        input_img = self.preprocess(image)
        
        # Copy input data to input buffer
        np.copyto(self.inputs[0]['host'], input_img.ravel())
        
        # Transfer data from host to device (GPU)
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer predictions from device to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        # Synchronize the stream to wait for all operations to finish
        self.stream.synchronize()
        
        # Process detections based on YOLOv8 output format
        detections = []
        
        # Get output data
        output = self.outputs[0]['host']
        output_shape = self.outputs[0]['shape']
        
        # Use cached format detection after first frame
        if hasattr(self, 'detected_format'):
            format_type = self.detected_format
        else:
            # Detect format only on first run
            format_type = None
            # Format 1: Direct boxes - for engines exported with end2end NMS
            if len(output_shape) == 2 and output_shape[1] == 7:
                format_type = "nms"
            # Format 2: Raw output without NMS
            elif len(output_shape) == 3 and output_shape[2] > 5:
                format_type = "raw"
            # Format 3: Single linear array
            elif len(output_shape) == 1:
                # Try YOLOv8's format (common for 640x640 models)
                if len(output) % 84 == 0:
                    format_type = "linear84"
                elif len(output) % 85 == 0:
                    format_type = "linear85"
                else:
                    format_type = "unknown"
            
            # Cache the detected format
            self.detected_format = format_type
            print(f"Detected format: {format_type}, Output shape: {output_shape}")
        
        # Process based on detected format
        if format_type == "nms":
            # Format: [num_detections, 7] with NMS
            num_outputs = output_shape[0]
            reshaped_output = output.reshape(num_outputs, 7)
            
            # Process only the first few rows (many will be zero-padded)
            for i in range(min(100, num_outputs)):
                confidence = reshaped_output[i][5]
                if confidence <= self.conf_threshold:
                    continue
                
                # Get coordinates and class
                x1, y1, x2, y2 = reshaped_output[i][1:5]
                class_id = int(reshaped_output[i][6])
                
                # Normalize coordinates if needed
                if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
                    img_height, img_width = image.shape[:2]
                    x1, x2 = x1/img_width, x2/img_width
                    y1, y2 = y1/img_height, y2/img_height
                
                # Only add valid detections
                if x1 < x2 and y1 < y2 and 0 <= class_id < len(classNames):
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': class_id
                    })
            
        elif format_type == "raw":
            # Format: [batch, num_anchors, num_classes+5]
            batch_size = output_shape[0]
            num_anchors = output_shape[1]
            num_values = output_shape[2]
            
            # Process only first batch, limit anchors for speed
            max_anchors = min(1000, num_anchors)  # Limit processing
            
            # Vectorized approach for faster processing
            reshaped = output.reshape(batch_size, num_anchors, num_values)
            confidences = reshaped[0, :max_anchors, 4]
            
            # Filter by confidence
            confident_idx = np.where(confidences > self.conf_threshold)[0]
            
            for idx in confident_idx:
                class_scores = reshaped[0, idx, 5:]
                class_id = np.argmax(class_scores)
                
                if class_id < len(classNames):
                    x, y, w, h = reshaped[0, idx, 0:4]
                    
                    # Convert to corner format
                    x1 = max(0, x - w/2)
                    y1 = max(0, y - h/2)
                    x2 = min(1, x + w/2)
                    y2 = min(1, y + h/2)
                    
                    if x1 < x2 and y1 < y2:
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': float(confidences[idx]),
                            'class_id': int(class_id)
                        })
        
        elif format_type == "linear84" or format_type == "linear85":
            # Format: linear array that can be reshaped
            cols = 84 if format_type == "linear84" else 85
            num_anchors = len(output) // cols
            
            # Reshape and get only first 1000 for speed
            max_anchors = min(1000, num_anchors)
            reshaped = output.reshape(num_anchors, cols)[:max_anchors]
            
            # Vectorized confidence filtering
            confidences = reshaped[:, 4]
            mask = confidences > self.conf_threshold
            filtered = reshaped[mask]
            
            for i in range(len(filtered)):
                # Get class with highest score
                class_scores = filtered[i, 5:min(cols, 85)]
                class_id = np.argmax(class_scores)
                
                if class_id < len(classNames):
                    # Get coordinates
                    x, y, w, h = filtered[i, 0:4]
                    
                    # Convert to corner format
                    x1 = max(0, x - w/2)
                    y1 = max(0, y - h/2)
                    x2 = min(1, x + w/2)
                    y2 = min(1, y + h/2)
                    
                    if x1 < x2 and y1 < y2:
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': float(filtered[i, 4]),
                            'class_id': int(class_id)
                        })
        
        else:
            # Fallback for unknown formats - simplified to reduce overhead
            max_check = min(1000, len(output)//6*6)
            for i in range(0, max_check, 6):
                try:
                    if output[i+4] > self.conf_threshold:
                        x, y, w, h = output[i:i+4]
                        confidence = output[i+4]
                        class_id = int(output[i+5])
                        
                        if 0 <= class_id < len(classNames):
                            x1 = max(0, x - w/2)
                            y1 = max(0, y - h/2)
                            x2 = min(1, x + w/2)
                            y2 = min(1, y + h/2)
                            
                            if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1:
                                detections.append({
                                    'box': [x1, y1, x2, y2],
                                    'confidence': float(confidence),
                                    'class_id': class_id
                                })
                except:
                    continue
        
        # Fast NMS - only if we have more than 10 detections
        if len(detections) > 10:
            # Convert to format needed for NMS
            boxes = np.array([d['box'] for d in detections])
            scores = np.array([d['confidence'] for d in detections])
            
            # Simple filtering approach - faster than full NMS
            # Keep only top detections by confidence
            indices = np.argsort(scores)[::-1][:20]  # Keep top 20
            detections = [detections[i] for i in indices]
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw detections on image"""
        img_height, img_width = image.shape[:2]
        
        for det in detections:
            # Convert normalized coordinates to pixel coordinates
            box = det['box']
            x1, y1, x2, y2 = [
                int(box[0] * img_width), 
                int(box[1] * img_height), 
                int(box[2] * img_width), 
                int(box[3] * img_height)
            ]
            
            # Filter out invalid boxes (too small or too large)
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Skip tiny boxes or boxes covering most of the image
            if box_width < 5 or box_height < 5:
                continue
                
            # Skip boxes that are too large (more than 90% of image)
            if box_width > img_width * 0.9 and box_height > img_height * 0.9:
                continue
            
            # Ensure coordinates are within image boundaries
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            # Get class information and validate
            class_id = det['class_id']
            confidence = det['confidence']
            
            # Skip invalid class IDs
            if class_id < 0 or class_id >= len(classNames):
                continue
                
            class_name = classNames[class_id]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create label with class name and confidence
            label = f"{class_name} {confidence:.2f}"
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                image,
                (x1, y1 - text_size[1] - 5),
                (x1 + text_size[0] + 5, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                image,
                label,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        return image

# Define a simple NMS implementation since the import might not be available
def non_max_suppression_fast(boxes, scores, overlap_thresh):
    """
    Non-maximum suppression implementation for overlapping bounding boxes
    Args:
        boxes: array of [x1, y1, x2, y2] (normalized coordinates)
        scores: array of confidence scores
        overlap_thresh: overlap threshold for suppression
    Returns:
        indices of boxes to keep
    """
    # If no boxes, return empty list
    if len(boxes) == 0:
        return []
    
    # Convert boxes to numpy array if not already
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    
    # Initialize the list of picked indexes
    pick = []
    
    # Grab the coordinates of the boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute the area of the boxes
    area = (x2 - x1) * (y2 - y1)
    
    # Sort the scores from high to low
    idxs = np.argsort(scores)[::-1]
    
    # Keep looping while some indexes still remain in the idxs list
    while len(idxs) > 0:
        # Grab the last index in the idxs list and add to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[0]
        pick.append(i)
        
        # Find the largest coordinates for the start of the boxes and
        # the smallest coordinates for the end of the boxes
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        # Compute the width and height of the overlapping area
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[1:]]
        
        # Delete all indexes from the idxs list that have overlap greater than threshold
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
    
    return pick

# Main function
def main():
    # Create window
    window_name = "YOLOv8-TensorRT Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Load TensorRT engine
    print("Loading YOLOv8 TensorRT engine...")
    model = TensorRTDetector("yolov8n.engine", conf_threshold=0.25)  # Lower confidence threshold for testing
    print("Model loaded successfully!")
    
    # Camera settings - reduce resolution for speed
    capture_width = 640  # Reduced from 1280
    capture_height = 480  # Reduced from 720
    display_width = 640
    display_height = 480
    framerate = 30
    flip_method = 0
    
    # Create pipeline with lower resolution
    pipeline = gstreamer_pipeline(
        sensor_id=0,
        capture_width=capture_width,
        capture_height=capture_height,
        display_width=display_width,
        display_height=display_height,
        framerate=framerate,
        flip_method=flip_method
    )
    print("Starting camera with pipeline:", pipeline)
    
    # Create video capture object
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Failed to open camera!")
        exit()
    
    print("Camera opened successfully!")
    
    # Variables for FPS calculation
    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    processing_times = []
    
    # Skip frames variable for speed
    frame_skip_counter = 0
    frame_skip_rate = 1  # Process every Nth frame (adjust based on performance)
    
    # Detection loop
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame")
                break
                
            # Skip frames for better performance
            frame_skip_counter += 1
            if frame_skip_counter % frame_skip_rate != 0:
                # Still display the frame, just don't run inference
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                continue
            
            # Update FPS calculation
            frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                fps_start_time = time.time()
            
            # Run inference
            start_time = time.time()
            try:
                # Detect objects
                detections = model.detect(frame)
                
                # Draw detections directly on the frame
                model.draw_detections(frame, detections)
                
            except Exception as infer_error:
                print(f"Inference error: {infer_error}")
                # No need to create a copy, just continue with the original frame
            
            # Track processing time
            process_time = time.time() - start_time
            processing_times.append(process_time)
            if len(processing_times) > 30:
                processing_times.pop(0)
            avg_process_time = sum(processing_times) / len(processing_times)
            
            # Display FPS and processing time
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Inference: {avg_process_time*1000:.1f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1
            )
            
            # Display the frame
            cv2.imshow(window_name, frame)
            
            # Check for ESC key to exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            
            # Display the frame
            cv2.imshow(window_name, display_frame)
            
            # Check for ESC key to exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main()
