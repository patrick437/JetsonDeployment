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
        
        # Process detections based on detected output format
        detections = []
        output = self.outputs[0]['host']
        output_shape = self.outputs[0]['shape']
        
        # Process based on pre-determined format for better performance
        if self.output_format == "nms_boxes":
            # Format: [num_detections, 7] with [batch_id, x1, y1, x2, y2, confidence, class_id]
            num_outputs = output_shape[0]
            reshaped_output = output.reshape(num_outputs, 7)
            
            for i in range(num_outputs):
                confidence = reshaped_output[i][5]
                
                # Skip rows with zero confidence or batch_id != 0
                if confidence <= self.conf_threshold or reshaped_output[i][0] != 0:
                    continue
                
                # Get coordinates
                x1, y1, x2, y2 = reshaped_output[i][1:5]
                
                # Check if coordinates are normalized (0-1) or absolute
                if max(x1, x2, y1, y2) > 1.0:
                    # If coordinates are in pixels, convert to normalized
                    img_height, img_width = image.shape[:2]
                    x1, x2 = x1/img_width, x2/img_width
                    y1, y2 = y1/img_height, y2/img_height
                
                class_id = int(reshaped_output[i][6])
                
                # Add valid detections
                if x1 < x2 and y1 < y2 and 0 <= class_id < len(classNames):
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': int(class_id)
                    })
                
        elif self.output_format == "raw_output":
            # Format: [batch, num_anchors, num_classes+5]
            batch_size = output_shape[0]
            num_anchors = output_shape[1]
            num_values = output_shape[2]
            
            reshaped_output = output.reshape(batch_size, num_anchors, num_values)
            
            # Process only first batch for simplicity
            for i in range(num_anchors):
                # Get box confidence
                confidence = reshaped_output[0, i, 4]
                
                if confidence > self.conf_threshold:
                    # Get class scores
                    class_scores = reshaped_output[0, i, 5:]
                    class_id = np.argmax(class_scores)
                    
                    # YOLOv8 outputs center_x, center_y, width, height
                    x, y, w, h = reshaped_output[0, i, 0:4]
                    
                    # Convert to corner coordinates
                    x1 = max(0, x - w/2)
                    y1 = max(0, y - h/2)
                    x2 = min(1, x + w/2)
                    y2 = min(1, y + h/2)
                    
                    if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and class_id < len(classNames):
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'class_id': int(class_id)
                        })
                    
        elif self.output_format == "yolo_flat":
            # Format: Flat array that can be reshaped to [num_anchors, 84]
            num_anchors = len(output) // 84
            reshaped_output = output.reshape(num_anchors, 84)
            
            for i in range(num_anchors):
                confidence = reshaped_output[i, 4]
                if confidence > self.conf_threshold:
                    class_scores = reshaped_output[i, 5:84]
                    class_id = np.argmax(class_scores)
                    
                    # Convert from center format to corner format
                    x, y, w, h = reshaped_output[i, 0:4]
                    x1 = max(0, x - w/2)
                    y1 = max(0, y - h/2)
                    x2 = min(1, x + w/2)
                    y2 = min(1, y + h/2)
                    
                    if 0 <= x1import cv2
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
        self.input_binding_idx = 0  # Assume first binding is input
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
        
        # Determine output format once at initialization
        self.output_format = "unknown"
        output_shape = self.outputs[0]['shape']
        if len(output_shape) == 2 and output_shape[1] == 7:
            self.output_format = "nms_boxes"
        elif len(output_shape) == 3 and output_shape[2] > 5:
            self.output_format = "raw_output"
        elif len(output_shape) == 1:
            if len(self.outputs[0]['host']) % 84 == 0:
                self.output_format = "yolo_flat"
        print(f"Using output format: {self.output_format}")
    
    def preprocess(self, image):
        """Preprocess image for inference"""
        # Resize
        input_img = cv2.resize(image, (self.width, self.height))
        
        # Convert to RGB (YOLOv8 expects RGB)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1]
        input_img = input_img.astype(np.float32) / 255.0
        
        # HWC to CHW format (convert to channels first)
        input_img = input_img.transpose((2, 0, 1))
        
        # Add batch dimension
        input_img = np.expand_dims(input_img, axis=0)
        
        return input_img
    
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
        
        # Handle based on previously identified format (reduce debugging overhead)
        try:
            # Format 1: Direct boxes - for engines exported with end2end NMS
            # Shape: [num_detections, 7] where each row is [batch_id, x1, y1, x2, y2, confidence, class_id]
            if len(output_shape) == 2 and output_shape[1] == 7:
                num_outputs = output_shape[0]
                reshaped_output = output.reshape(num_outputs, 7)
                
                for i in range(num_outputs):
                    confidence = reshaped_output[i][5]
                    
                    # Skip rows with zero confidence or batch_id != 0
                    if confidence <= self.conf_threshold or reshaped_output[i][0] != 0:
                        continue
                    
                    # Get coordinates (these are often already in pixel coordinates for some engines)
                    x1, y1, x2, y2 = reshaped_output[i][1:5]
                    
                    # Check if coordinates are normalized (0-1) or absolute
                    if max(x1, x2, y1, y2) > 1.0:
                        # If coordinates are in pixels, convert to normalized
                        img_height, img_width = image.shape[:2]
                        x1, x2 = x1/img_width, x2/img_width
                        y1, y2 = y1/img_height, y2/img_height
                    
                    class_id = int(reshaped_output[i][6])
                    
                    # Add only valid detections
                    if x1 < x2 and y1 < y2 and 0 <= class_id < len(classNames):
                        detections.append({
                            'box': [x1, y1, x2, y2],  # Normalized coordinates
                            'confidence': float(confidence),
                            'class_id': int(class_id)
                        })
            
            # Format 2: Raw output without NMS
            # Shape: [batch, num_anchors, num_classes+5]
            elif len(output_shape) == 3 and output_shape[2] > 5:
                batch_size = output_shape[0]
                num_anchors = output_shape[1]
                num_values = output_shape[2]
                num_classes = num_values - 5
                
                reshaped_output = output.reshape(batch_size, num_anchors, num_values)
                
                # Process only first batch for simplicity
                for i in range(num_anchors):
                    # Get box confidence
                    confidence = reshaped_output[0, i, 4]
                    
                    if confidence > self.conf_threshold:
                        # Get class scores
                        class_scores = reshaped_output[0, i, 5:]
                        class_id = np.argmax(class_scores)
                        class_confidence = class_scores[class_id]
                        
                        if class_confidence > 0.1:  # Lower threshold for class confidence
                            # YOLOv8 outputs center_x, center_y, width, height in normalized coordinates
                            x, y, w, h = reshaped_output[0, i, 0:4]
                            
                            # Convert to corner coordinates (still normalized)
                            x1 = max(0, x - w/2)
                            y1 = max(0, y - h/2)
                            x2 = min(1, x + w/2)
                            y2 = min(1, y + h/2)
                            
                            # Check reasonable values to filter out bad detections
                            if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and class_id < len(classNames):
                                detections.append({
                                    'box': [x1, y1, x2, y2],
                                    'confidence': float(confidence * class_confidence),
                                    'class_id': int(class_id)
                                })
            
            # Format 3: Single linear array - needs special handling
            elif len(output_shape) == 1:
                # Check if output is multiple of 84 (4 coords + 1 obj + 79 classes)
                if len(output) % 84 == 0:
                    num_anchors = len(output) // 84
                    reshaped_output = output.reshape(num_anchors, 84)
                    
                    for i in range(num_anchors):
                        confidence = reshaped_output[i, 4]
                        if confidence > self.conf_threshold:
                            class_scores = reshaped_output[i, 5:84]
                            class_id = np.argmax(class_scores)
                            class_confidence = class_scores[class_id]
                            
                            if class_confidence > 0.1:
                                # Convert from center format to corner format
                                x, y, w, h = reshaped_output[i, 0:4]
                                x1 = max(0, x - w/2)
                                y1 = max(0, y - h/2)
                                x2 = min(1, x + w/2)
                                y2 = min(1, y + h/2)
                                
                                if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and class_id < len(classNames):
                                    detections.append({
                                        'box': [x1, y1, x2, y2],
                                        'confidence': float(confidence * class_confidence),
                                        'class_id': int(class_id)
                                    })
        except Exception as e:
            print(f"Error processing detections: {e}")
        
        # Apply simple NMS if we have many detections
        if len(detections) > 20:  # Only do NMS if we have many detections
            # Simple approach: keep only top 10 detections sorted by confidence
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:10]
        
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
    
    # Camera settings
    capture_width = 1280
    capture_height = 720
    display_width = 640
    display_height = 640
    framerate = 30
    flip_method = 0
    
    # Create pipeline
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
    
    # Detection loop
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame")
                break
                
            # Update FPS calculation
            frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                fps_start_time = time.time()
            
            # Run inference
            try:
                # Detect objects
                detections = model.detect(frame)
                
                # Draw detections on the frame
                display_frame = frame.copy()
                model.draw_detections(display_frame, detections)
                
            except Exception as infer_error:
                print(f"Inference error: {infer_error}")
                display_frame = frame.copy()
            
            # Display FPS
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
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
