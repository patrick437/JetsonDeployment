import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # Initializes CUDA context
import warnings
import traceback # For detailed error printing

# Suppress numpy deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- GStreamer Pipeline Function ---
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280, # Set desired capture width
    capture_height=720, # Set desired capture height
    display_width=1280, # Set desired display width (output size from GStreamer to OpenCV)
    display_height=720, # Set desired display height
    framerate=30,
    flip_method=0, # 0=None, 1=CounterClockwise, 2=Rotate180, 3=Clockwise, 4=HorizontalFlip, 5=UpperRightDiagonal, 6=VerticalFlip, 7=UpperLeftDiagonal
):
    # Note: nvvidconv handles resizing from capture_* to display_* efficiently
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True" # Add drop=True and max-buffers=1 for potential latency reduction
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width, # nvvidconv resizes to this size
            display_height, # nvvidconv resizes to this size
        )
    )

# --- COCO Class Names ---
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

# --- Non-Maximum Suppression Function ---
def non_max_suppression_fast(boxes, scores, overlap_thresh):
    """
    Non-maximum suppression implementation for overlapping bounding boxes
    Args:
        boxes: array of [x1, y1, x2, y2] (normalized coordinates)
        scores: array of confidence scores
        overlap_thresh: overlap threshold (IoU) for suppression
    Returns:
        indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Check for invalid boxes (area calculation)
    area = (x2 - x1) * (y2 - y1)
    valid_area_indices = np.where(area > 0)[0]
    if len(valid_area_indices) == 0:
        return []

    # Consider only boxes with valid area
    idxs = valid_area_indices[np.argsort(scores[valid_area_indices])] # Ascending order

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find intersection with remaining boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute width and height of the intersection
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # Compute the ratio of overlap (IoU: Intersection over Union)
        # Union = area[i] + area[other] - intersection
        intersection = w * h
        union = area[i] + area[idxs[:last]] - intersection
        # Avoid division by zero
        valid_union_indices = np.where(union > 1e-6)[0] # Indices relative to idxs[:last]
        overlap = np.zeros_like(union)
        if len(valid_union_indices) > 0:
            overlap[valid_union_indices] = intersection[valid_union_indices] / union[valid_union_indices]

        # Delete indices associated with overlaps greater than the threshold
        # Need to map indices back to the original `idxs` array before deleting
        indices_to_delete = np.where(overlap > overlap_thresh)[0]
        idxs = np.delete(idxs, np.concatenate(([last], indices_to_delete)))

    return pick


# --- TensorRT Detector Class ---
class TensorRTDetector:
    def __init__(self, engine_path, conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")

        # Load engine
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
             raise RuntimeError("Failed to deserialize CUDA engine.")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context.")

        # Get input binding details
        self.input_binding_idx = -1
        self.input_name = ""
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_binding_idx = i
                self.input_name = self.engine.get_binding_name(i)
                break
        if self.input_binding_idx == -1:
            raise ValueError("Could not find input binding in the engine")
        print(f"Found input binding '{self.input_name}' at index {self.input_binding_idx}")

        # Get input shape (handling dynamic dimensions, assume batch=1)
        self.input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        print(f"Original engine input shape: {self.input_shape}")
        if self.input_shape[0] == -1:
            print(f"Input shape has dynamic batch, assuming batch size = 1")
            self.input_shape = (1,) + tuple(self.input_shape[1:])
            # Set the binding shape for the context
            try:
                self.context.set_binding_shape(self.input_binding_idx, self.input_shape)
                print(f"Set context binding shape for input to: {self.input_shape}")
            except Exception as e:
                print(f"Warning: Failed to set dynamic input shape in context: {e}")
                print("Proceeding with assumed shape, but this might cause issues if context doesn't adapt.")

        # Ensure input has 4 dimensions (B, C, H, W)
        if len(self.input_shape) != 4:
             raise ValueError(f"Expected 4D input shape (B, C, H, W), but got {self.input_shape}")

        self.batch_size, self.channels, self.height, self.width = self.input_shape
        print(f"Using Input Shape (B, C, H, W): {self.batch_size}, {self.channels}, {self.height}, {self.width}")

        # Create buffers for input and output
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            # Use shape from context if possible (especially after setting dynamic input)
            try:
                binding_shape = tuple(self.context.get_binding_shape(i))
                # Resolve dynamic output dimensions based on input batch size if needed
                if binding_shape[0] == -1:
                   binding_shape = (self.batch_size,) + tuple(binding_shape[1:])
                   print(f"Resolved dynamic shape for binding {i} ({binding_name}): {binding_shape}")
            except Exception as e:
                 print(f"Warning: Using engine shape for binding {i} ({binding_name}), context shape error: {e}")
                 binding_shape = tuple(self.engine.get_binding_shape(i))
                 if binding_shape[0] == -1:
                     binding_shape = (self.batch_size,) + tuple(binding_shape[1:])
                     print(f"Resolved dynamic engine shape for binding {i} ({binding_name}): {binding_shape}")

            size = trt.volume(binding_shape)
            if size < 0:
                 raise RuntimeError(f"Binding {i} ({binding_name}) has negative size {size} with shape {binding_shape}. Check model export/engine build.")
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape, 'name': binding_name})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape, 'name': binding_name})
                print(f"Output binding '{binding_name}' at index {i} with shape {binding_shape} and dtype {dtype}")

        if not self.outputs:
             raise ValueError("Could not find any output bindings in the engine")

        print(f"Detector initialized. Input: {self.inputs[0]['name']} {self.inputs[0]['shape']}. Output: {self.outputs[0]['name']} {self.outputs[0]['shape']}")
        self.debug_output_printed = False
        self.debug_final_printed = False

    def preprocess(self, image):
        """Preprocess image for inference - assumes input 'image' is already resized"""
        # Convert BGR to RGB (YOLOv8 expects RGB)
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to float32 and normalize
        input_img = input_img.astype(np.float32) / 255.0

        # Transpose from HWC to CHW format (Height, Width, Channels) -> (Channels, Height, Width)
        input_img = input_img.transpose(2, 0, 1)

        # Add batch dimension
        input_img = np.expand_dims(input_img, axis=0) # Shape: (1, C, H, W)

        # Ensure the input is contiguous in memory
        input_img = np.ascontiguousarray(input_img)
        return input_img

    def detect(self, image_for_model):
        """Run inference on an image (already resized) and return detections"""
        # Preprocess the already resized image
        input_img = self.preprocess(image_for_model)

        # Verify input shape matches buffer
        if input_img.shape != self.inputs[0]['shape']:
             # This should ideally not happen if resizing is done correctly before calling detect
             raise ValueError(f"Mismatched input shape! Expected {self.inputs[0]['shape']} but got {input_img.shape} after preprocessing.")

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

        # --- Process Detections ---
        output_data = self.outputs[0]['host']
        output_shape = self.outputs[0]['shape']

        if not self.debug_output_printed:
            print(f"Raw Output shape: {output_shape}")
            print(f"Raw Output sample (first 20 elements): {output_data[:20]}")
            self.debug_output_printed = True # Print only once

        # --- Reshape and Transpose based on common YOLOv8 output ---
        num_classes = len(classNames)
        num_coords = 4
        expected_channels = num_classes + num_coords # e.g., 84 for COCO

        processed_output = None
        # Case 1: Expected shape [batch, channels, detections] e.g., (1, 84, 8400)
        if len(output_shape) == 3 and output_shape[0] == self.batch_size and output_shape[1] == expected_channels:
            num_detections = output_shape[2]
            processed_output = output_data.reshape(output_shape).transpose(0, 2, 1) # -> (batch, detections, channels)
            print(f"Processing output format: [B, C, D] -> Transposed to {processed_output.shape}")
        # Case 2: Expected shape [batch, detections, channels] e.g., (1, 8400, 84)
        elif len(output_shape) == 3 and output_shape[0] == self.batch_size and output_shape[2] == expected_channels:
            num_detections = output_shape[1]
            processed_output = output_data.reshape(output_shape) # Already in desired format
            print(f"Processing output format: [B, D, C] -> Shape {processed_output.shape}")
        # Case 3: Flattened output [batch, detections * channels] e.g., (1, 8400 * 84)
        elif len(output_shape) == 2 and output_shape[0] == self.batch_size:
            total_elements = output_shape[1]
            # Try to infer num_detections assuming channel size is correct
            if total_elements % expected_channels == 0:
                 num_detections = total_elements // expected_channels
                 try:
                      processed_output = output_data.reshape((self.batch_size, num_detections, expected_channels))
                      print(f"Processing output format: Flattened [B, D*C] -> Reshaped to {processed_output.shape}")
                 except ValueError as e:
                      print(f"Error reshaping flattened output: {e}")
                      return []
            else:
                print(f"Cannot process flattened output shape {output_shape}, not divisible by expected channels {expected_channels}")
                return []
        else:
             print(f"Error: Unexpected output shape {output_shape}. Cannot determine processing format.")
             return []

        # Process detections for the first batch
        detections_batch = processed_output[0] # Shape: (num_detections, channels)

        # Extract data
        boxes_xywh = detections_batch[:, :num_coords]      # cx, cy, w, h (normalized)
        all_scores = detections_batch[:, num_coords:]      # class scores

        # Get class IDs and max confidences per box
        class_ids = np.argmax(all_scores, axis=1)
        max_confidences = np.max(all_scores, axis=1)

        # Filter by confidence threshold
        keep = max_confidences >= self.conf_threshold
        if not np.any(keep): # Check if any detections passed
            return []

        filtered_boxes_xywh = boxes_xywh[keep]
        filtered_confidences = max_confidences[keep]
        filtered_class_ids = class_ids[keep]

        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] (still normalized)
        x_center, y_center, width, height = filtered_boxes_xywh.T
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        filtered_boxes_xyxy = np.stack((x1, y1, x2, y2), axis=1)

        # Apply Non-Maximum Suppression (NMS)
        indices_to_keep = non_max_suppression_fast(filtered_boxes_xyxy, filtered_confidences, self.nms_threshold)

        # Prepare final detections list
        final_detections = []
        for idx in indices_to_keep:
             box = filtered_boxes_xyxy[idx]
             confidence = filtered_confidences[idx]
             class_id = filtered_class_ids[idx]

             # Ensure box coordinates are valid [0, 1] and x1 < x2, y1 < y2
             box[0] = max(0.0, min(1.0, box[0])) # x1
             box[1] = max(0.0, min(1.0, box[1])) # y1
             box[2] = max(0.0, min(1.0, box[2])) # x2
             box[3] = max(0.0, min(1.0, box[3])) # y2

             if box[0] < box[2] and box[1] < box[3]: # Check validity
                  final_detections.append({
                       'box': box.tolist(), # [x1, y1, x2, y2] normalized
                       'confidence': float(confidence),
                       'class_id': int(class_id)
                  })

        # Debug print final detections count (only once)
        if len(final_detections) > 0 and not self.debug_final_printed:
             print(f"Found {len(final_detections)} final detections after NMS.")
             self.debug_final_printed = True
        elif np.any(keep) and len(final_detections) == 0 and not self.debug_final_printed:
             print("All detections were suppressed by NMS.")
             self.debug_final_printed = True

        return final_detections

    def draw_detections(self, image_to_draw_on, detections):
        """Draw detections on the provided image"""
        img_height, img_width = image_to_draw_on.shape[:2]

        for det in detections:
            # Get normalized box coordinates [x1, y1, x2, y2]
            box = det['box']
            x1_norm, y1_norm, x2_norm, y2_norm = box

            # Convert normalized coordinates to pixel coordinates based on the image dimensions
            x1 = int(x1_norm * img_width)
            y1 = int(y1_norm * img_height)
            x2 = int(x2_norm * img_width)
            y2 = int(y2_norm * img_height)

            # Ensure coordinates are within image boundaries after scaling
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            # Check if box is valid (sometimes NMS might leave tiny boxes or coords might swap)
            if x2 <= x1 or y2 <= y1:
                continue

            # Get class info
            class_id = det['class_id']
            confidence = det['confidence']

            # Skip invalid class IDs
            if class_id < 0 or class_id >= len(classNames):
                print(f"Warning: Skipping detection with invalid class_id {class_id}")
                continue

            class_name = classNames[class_id]

            # Draw bounding box
            cv2.rectangle(image_to_draw_on, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Create label
            label = f"{class_name} {confidence:.2f}"

            # Draw label background and text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            # Position background slightly above the top-left corner (y1)
            label_bg_y1 = max(y1 - text_size[1] - 5, 0) # Ensure background doesn't go off-screen top
            label_bg_y2 = y1 - 5
            # Adjust background position if it's too close to the top edge
            if label_bg_y1 < 0:
                label_bg_y1 = y1 + 5
                label_bg_y2 = y1 + text_size[1] + 5

            cv2.rectangle(image_to_draw_on, (x1, label_bg_y1), (x1 + text_size[0] + 5, label_bg_y2), (0, 255, 0), -1)
            # Calculate text position based on background position
            text_y = label_bg_y2 - 3 if label_bg_y1 < y1 else label_bg_y1 + text_size[1] + 3 # Adjust based on bg position
            cv2.putText(image_to_draw_on, label, (x1 + 3, text_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image_to_draw_on

# --- Main Function ---
def main():
    print("Starting YOLOv8 TensorRT Detection...")

    # --- Configuration ---
    # Performance Tip: Check Jetson power mode: sudo nvpmodel -q (should be MAXN or mode 0)
    # Use: sudo nvpmodel -m 0 to set max performance. Also check thermals: sudo tegrastats
    engine_file_path = "yolov8n.engine"
    confidence_threshold = 0.20 # Lowered threshold for better detection of distant/less confident objects
    nms_threshold = 0.45       # Non-Maximum Suppression threshold

    # Camera Capture Settings (High-Res Capture)
    capture_width = 1280  # Use a higher resolution for capture
    capture_height = 720
    # Display Settings (Size passed from GStreamer to OpenCV)
    # Let's keep display resolution the same as capture for simplicity here
    # OpenCV will receive frames of this size.
    display_width = capture_width
    display_height = capture_height
    framerate = 30 # Adjust if needed based on camera capabilities / performance
    flip_method = 0 # Adjust if your camera is mounted upside down

    window_name = "YOLOv8-TensorRT Detection (High-Res Capture)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Make window resizable

    # --- Initialize Model ---
    try:
        model = TensorRTDetector(
            engine_file_path,
            conf_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )
        print(f"Model {engine_file_path} loaded successfully!")
        # Get the expected input size from the initialized model
        model_input_width = model.width
        model_input_height = model.height
        print(f"Model expects input size: {model_input_width}x{model_input_height}")
    except Exception as e:
        print(f"Error initializing TensorRT model: {e}")
        traceback.print_exc()
        return # Exit if model fails to load

    # --- Initialize Camera ---
    pipeline = gstreamer_pipeline(
        sensor_id=0,
        capture_width=capture_width,
        capture_height=capture_height,
        display_width=display_width, # GStreamer outputs this size to OpenCV
        display_height=display_height,
        framerate=framerate,
        flip_method=flip_method
    )
    print("Starting camera with GStreamer pipeline:")
    print(pipeline)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("ERROR: Failed to open camera via GStreamer!")
        print("Attempting fallback to default camera (cv2.VideoCapture(0))...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
             print("ERROR: Failed to open default camera either.")
             return # Exit if camera fails completely
        else:
            # Update display dimensions if fallback camera has different resolution
            display_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            display_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Using default camera. Frame size: {display_width}x{display_height}")

    print("Camera opened successfully!")

    # --- Timing and FPS Variables ---
    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    processing_times = [] # For averaging processing time

    # --- Main Loop ---
    try:
        while True:
            t_start_loop = time.time() # Start time for full loop timing

            # --- Read Frame ---
            t_start_read = time.time()
            ret, frame_display_res = cap.read() # Frame received is display_width x display_height
            t_end_read = time.time()

            if not ret or frame_display_res is None:
                print("Warning: Failed to get frame or empty frame received. Skipping.")
                time.sleep(0.1) # Avoid busy-waiting on error
                continue

            # --- Prepare Frame for Model ---
            t_start_resize = time.time()
            # Resize the display frame down to the model's expected input size
            frame_for_model = cv2.resize(
                frame_display_res,
                (model_input_width, model_input_height),
                interpolation=cv2.INTER_LINEAR # Use linear for better downscaling
            )
            t_end_resize = time.time()

            # --- Detect Objects ---
            detections = []
            display_frame = frame_display_res # Start with the original frame for display

            t_start_detect = time.time()
            try:
                # Pass the specifically resized frame to the detection method
                detections = model.detect(frame_for_model)
            except Exception as detect_error:
                print(f"Error during model detection: {detect_error}")
                # Don't crash, just skip drawing detections for this frame
            t_end_detect = time.time()

            # --- Draw Detections ---
            t_start_draw = time.time()
            try:
                # Draw onto a *copy* of the original display resolution frame
                # The draw_detections function will scale normalized coords to this frame's size
                display_frame = model.draw_detections(frame_display_res.copy(), detections)
            except Exception as draw_error:
                print(f"Error during drawing detections: {draw_error}")
                # display_frame remains the original frame_display_res without boxes
            t_end_draw = time.time()

            # --- Calculate Timings and FPS ---
            t_end_loop = time.time()

            # Calculate individual step times (in milliseconds)
            read_time_ms = (t_end_read - t_start_read) * 1000
            resize_time_ms = (t_end_resize - t_start_resize) * 1000
            detect_time_ms = (t_end_detect - t_start_detect) * 1000 # Includes preprocess, inference, NMS
            draw_time_ms = (t_end_draw - t_start_draw) * 1000
            total_loop_time_ms = (t_end_loop - t_start_loop) * 1000

            processing_times.append(detect_time_ms) # Track detection time specifically
            if len(processing_times) > 30: processing_times.pop(0)
            avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0

            # Calculate overall FPS
            frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                fps_start_time = time.time()

            # --- Display Info on Frame ---
            fps_text = f"FPS: {fps:.1f}"
            inf_text = f"Detect+NMS: {avg_process_time:.1f}ms"
            # Add detailed timings optionally
            # timing_text = f"R:{read_time_ms:.0f} Re:{resize_time_ms:.0f} D:{detect_time_ms:.0f} Dr:{draw_time_ms:.0f} T:{total_loop_time_ms:.0f}"

            cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, inf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(display_frame, timing_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # --- Show Frame ---
            cv2.imshow(window_name, display_frame)

            # --- Exit Condition ---
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("ESC key pressed, exiting...")
                break
            elif key == ord('q') or key == ord('Q'): # Allow Q to quit too
                 print("Q key pressed, exiting...")
                 break

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        traceback.print_exc() # Print detailed traceback
    finally:
        # --- Release Resources ---
        print("Releasing resources...")
        if cap.isOpened():
            cap.release()
            print("Camera released.")
        cv2.destroyAllWindows()
        print("Windows destroyed.")
        # CUDA context is usually handled by pycuda.autoinit, but explicit cleanup can be added if needed
        # Consider deleting model object to potentially free GPU memory explicitly
        # del model
        # cuda.Context.pop() # Example if not using autoinit
        print("Done!")

if __name__ == "__main__":
    # Before running:
    # 1. Make sure 'yolov8n.engine' is in the same directory or provide the correct path.
    # 2. Check Jetson Power Mode: Run 'sudo nvpmodel -q' in terminal. If not MAXN/Mode 0, run 'sudo nvpmodel -m 0'.
    # 3. Monitor temperature during execution: Run 'sudo tegrastats' in another terminal.
    main()
