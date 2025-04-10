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
    capture_width=640,  # Lower capture resolution for performance
    capture_height=480,
    output_width=640,   # Target output width (e.g., model input width)
    output_height=480,  # Target output height (e.g., model input height)
    framerate=30,
    flip_method=0, # 0=None, 1=CCW, 2=180, 3=CW, 4=HFlip, 5=UR-Diag, 6=VFlip, 7=UL-Diag
):
    # Pipeline optimized for speed: Capture lower res, nvvidconv resizes to model input size
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! " # Resize happens here
        "videoconvert ! " # Convert to BGR
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True" # Deliver BGR frames
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            output_width,  # nvvidconv outputs this size
            output_height, # nvvidconv outputs this size
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

# Note: non_max_suppression_fast function is removed as we now use cv2.dnn.NMSBoxes

# --- TensorRT Detector Class ---
class TensorRTDetector:
    # Using cv2.dnn.NMSBoxes now
    def __init__(self, engine_path, conf_threshold=0.25, nms_threshold=0.45): # Adjusted default conf
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold # IoU Threshold for NMS

        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None: raise RuntimeError("Failed to deserialize engine.")
        self.context = self.engine.create_execution_context()
        if self.context is None: raise RuntimeError("Failed to create execution context.")

        self.input_binding_idx = -1
        self.input_name = ""
        # ... (Input binding finding logic - remains the same) ...
        for i in range(self.engine.num_bindings):
             if self.engine.binding_is_input(i):
                  self.input_binding_idx = i
                  self.input_name = self.engine.get_binding_name(i)
                  break
        if self.input_binding_idx == -1: raise ValueError("Input binding not found.")
        print(f"Found input binding '{self.input_name}' at index {self.input_binding_idx}")

        self.input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        print(f"Original engine input shape: {self.input_shape}")
        if self.input_shape[0] == -1:
            self.input_shape = (1,) + tuple(self.input_shape[1:])
            print(f"Assuming batch size 1 for dynamic input: {self.input_shape}")
            try:
                self.context.set_binding_shape(self.input_binding_idx, self.input_shape)
            except Exception as e:
                print(f"Warning: Failed to set dynamic input shape in context: {e}")
        if len(self.input_shape) != 4: raise ValueError("Expected 4D input.")
        self.batch_size, self.channels, self.height, self.width = self.input_shape
        print(f"Using Input Shape (B, C, H, W): {self.batch_size}, {self.channels}, {self.height}, {self.width}")

        # Allocate buffers (Host and Device)
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        # ... (Buffer allocation logic - remains the same) ...
        for i in range(self.engine.num_bindings):
             binding_name = self.engine.get_binding_name(i)
             dtype = trt.nptype(self.engine.get_binding_dtype(i))
             try:
                  binding_shape = tuple(self.context.get_binding_shape(i))
                  if binding_shape[0] == -1: binding_shape = (self.batch_size,) + tuple(binding_shape[1:])
             except Exception as e:
                  binding_shape = tuple(self.engine.get_binding_shape(i))
                  if binding_shape[0] == -1: binding_shape = (self.batch_size,) + tuple(binding_shape[1:])
                  print(f"Warning: Using engine shape for binding {i} ({binding_name}), context shape error: {e}")
             size = trt.volume(binding_shape)
             if size < 0: raise RuntimeError(f"Binding {i} ({binding_name}) has invalid size {size} for shape {binding_shape}")
             host_mem = cuda.pagelocked_empty(size, dtype)
             device_mem = cuda.mem_alloc(host_mem.nbytes)
             self.bindings.append(int(device_mem))
             if self.engine.binding_is_input(i):
                  self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape, 'name': binding_name})
             else:
                  self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape, 'name': binding_name})
                  print(f"Output binding '{binding_name}' at index {i} with shape {binding_shape} and dtype {dtype}")
        if not self.outputs: raise ValueError("Output bindings not found.")

        print(f"Detector initialized. Input: {self.inputs[0]['name']} {self.inputs[0]['shape']}. Output: {self.outputs[0]['name']} {self.outputs[0]['shape']}")
        self.debug_output_printed = False
        self.debug_final_printed = False
        self.nms_debug_printed = False

    def preprocess(self, image):
        """Preprocess image (already resized to model input size)."""
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1) # HWC -> CHW
        input_img = np.expand_dims(input_img, axis=0) # Add batch dim
        input_img = np.ascontiguousarray(input_img)
        return input_img

    # --- DETECT METHOD (WITH NMS BYPASSED and RAW SCORE PRINTING) ---
    def detect(self, frame_for_model):
        """
        Run inference on the frame (already resized) and return detections.
        Includes NMS processing to remove overlapping detections.
        """
        # Preprocess the frame provided (which should already be model input size)
        input_img = self.preprocess(frame_for_model)
    
        # Verify input shape matches buffer
        if input_img.shape != self.inputs[0]['shape']:
            raise ValueError(f"Mismatched input shape! Expected {self.inputs[0]['shape']} but got {input_img.shape}.")
    
        # --- Perform Inference ---
        np.copyto(self.inputs[0]['host'], input_img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        # --- End Inference ---
    
        # --- Get and Reshape Output ---
        output_data = self.outputs[0]['host']
        output_shape = self.outputs[0]['shape']
    
        if not self.debug_output_printed:
            print(f"Raw Output shape: {output_shape}")
            self.debug_output_printed = True
    
        num_classes = len(classNames) # Assumes classNames is defined globally or accessible
        num_coords = 4
        expected_channels = num_classes + num_coords
        processed_output = None
    
        # Handle different potential output shapes
        # Case 1: [B, C, D] e.g., (1, 84, 8400)
        if len(output_shape) == 3 and output_shape[0] == self.batch_size and output_shape[1] == expected_channels:
            processed_output = output_data.reshape(output_shape).transpose(0, 2, 1) # -> (B, D, C)
        # Case 2: [B, D, C] e.g., (1, 8400, 84)
        elif len(output_shape) == 3 and output_shape[0] == self.batch_size and output_shape[2] == expected_channels:
            processed_output = output_data.reshape(output_shape)
        # Case 3: Flattened [B, D*C] e.g., (1, 8400 * 84)
        elif len(output_shape) == 2 and output_shape[0] == self.batch_size:
            total_elements = output_shape[1]
            if total_elements % expected_channels == 0:
                num_detections = total_elements // expected_channels
                try: processed_output = output_data.reshape((self.batch_size, num_detections, expected_channels))
                except ValueError: return [] # Reshape failed
            else:
                print(f"Cannot process flattened output shape {output_shape}, not divisible by expected channels {expected_channels}")
                return [] # Cannot process
        else:
            print(f"Error: Unexpected output shape {output_shape}. Cannot determine processing format.")
            return []
        # --- End Reshape/Transpose ---
    
        # --- Initial Filtering by Confidence ---
        detections_batch = processed_output[0] # Shape: (num_detections, channels)
        boxes_xywh = detections_batch[:, :num_coords] # cx, cy, w, h (normalized)
        all_scores = detections_batch[:, num_coords:] # class scores (num_detections, num_classes)
        class_ids = np.argmax(all_scores, axis=1)
        max_confidences = np.max(all_scores, axis=1)
    
        # Indices of detections passing the confidence threshold *within detections_batch*
        keep_indices_in_batch = np.where(max_confidences >= self.conf_threshold)[0]
    
        if len(keep_indices_in_batch) == 0:
            # Reset flags if no detections found this frame, so next frame prints again if needed
            self.nms_debug_printed = False
            self.debug_final_printed = False
            return [] # Exit early
    
        # Get data ONLY for boxes that passed confidence threshold
        filtered_boxes_xywh = boxes_xywh[keep_indices_in_batch]
        filtered_confidences = max_confidences[keep_indices_in_batch]
        filtered_class_ids = class_ids[keep_indices_in_batch]
        filtered_raw_scores = all_scores[keep_indices_in_batch] # Get raw scores for filtered boxes
    
        # Generate xyxy boxes needed for NMS
        x_center, y_center, width, height = filtered_boxes_xywh.T
        x1 = x_center - width / 2; y1 = y_center - height / 2
        x2 = x_center + width / 2; y2 = y_center + height / 2
        filtered_boxes_xyxy = np.stack((x1, y1, x2, y2), axis=1)
    
        if not self.nms_debug_printed: # Debug: Print count pre-NMS once
            print(f"DEBUG: Found {len(filtered_boxes_xywh)} detections pre-NMS (Conf > {self.conf_threshold:.2f})")
            self.nms_debug_printed = True
    
        # --- Apply NMS using OpenCV's implementation ---
        # Convert data types to what cv2.dnn.NMSBoxes expects
        boxes_for_nms = filtered_boxes_xyxy.astype(np.float32)
        scores_for_nms = filtered_confidences.astype(np.float32)
        
        # Apply NMS (returns indices of kept boxes)
        try:
            # For OpenCV 4.5.1+, the resulting indices are directly returned
            nms_indices = cv2.dnn.NMSBoxes(
                boxes_for_nms.tolist(), 
                scores_for_nms.tolist(), 
                self.conf_threshold, 
                self.nms_threshold
            )
            
            # Handle different return formats based on OpenCV version
            if isinstance(nms_indices, tuple):
                # Some versions return tuple
                nms_indices = nms_indices[0]
            elif isinstance(nms_indices, np.ndarray) and len(nms_indices.shape) == 2:
                # Some versions return 2D array
                nms_indices = nms_indices.flatten()
            
            print(f"DEBUG: After NMS: kept {len(nms_indices)} out of {len(filtered_boxes_xywh)} detections")
        except Exception as e:
            print(f"Error during NMS: {e}. Using all filtered detections without NMS.")
            nms_indices = list(range(len(filtered_boxes_xywh)))  # Fallback: use all detections
    
        # --- Prepare final detections list after NMS ---
        final_detections = []
        
        # Print header only once when detections are found
        if not self.debug_final_printed and len(nms_indices) > 0:
            print(f"\n--- DETAILED SCORES FOR FRAME (After NMS, Conf > {self.conf_threshold:.2f}) ---")
    
        # Iterate through the indices kept after NMS
        for i, idx in enumerate(nms_indices):
            # Get data for this specific detection using the index 'idx'
            # which refers to the position within the *filtered* arrays
            box_xyxy = filtered_boxes_xyxy[idx]
            confidence = filtered_confidences[idx] # This is max_score
            class_id = filtered_class_ids[idx]     # This is argmax result
            raw_scores_for_this_box = filtered_raw_scores[idx] # Raw scores for this box
    
            # --- Print Raw Score Details (Only for the first frame with detections) ---
            if not self.debug_final_printed:
                print(f" Detection {i}: ArgMax Class={classNames[class_id]}({class_id}), MaxConf={confidence:.3f}")
                # Print Top 5 scores
                top_n = 5
                # Get indices of top N scores from raw_scores_for_this_box
                top_indices = np.argsort(raw_scores_for_this_box)[::-1][:top_n]
                print(f"   Top {top_n} Raw Scores:")
                for score_idx in top_indices:
                    score_val = raw_scores_for_this_box[score_idx]
                    # Only print if score is somewhat significant
                    if score_val > 0.01: # Adjust threshold to control verbosity
                        if 0 <= score_idx < len(classNames):
                            print(f"     - {classNames[score_idx]}({score_idx}): {score_val:.4f}")
                        else: # Should not happen if num_classes is correct
                            print(f"     - Invalid Class ID {score_idx}: {score_val:.4f}")
                print("-" * 10) # Separator
            # --- End Print Raw Score Details ---
    
            # Clamp normalized coords to [0, 1] and ensure validity (x1<x2, y1<y2)
            box_xyxy[0] = max(0.0, min(1.0, box_xyxy[0])) # x1
            box_xyxy[1] = max(0.0, min(1.0, box_xyxy[1])) # y1
            box_xyxy[2] = max(0.0, min(1.0, box_xyxy[2])) # x2
            box_xyxy[3] = max(0.0, min(1.0, box_xyxy[3])) # y2
    
            if box_xyxy[0] < box_xyxy[2] and box_xyxy[1] < box_xyxy[3]: # Check validity after clamping
                final_detections.append({
                    'box': box_xyxy.tolist(), # Store normalized [x1, y1, x2, y2]
                    'confidence': float(confidence),
                    'class_id': int(class_id)
                })
    
        # Set flag after printing scores for the first frame that has detections
        # This prevents printing scores for every single subsequent frame
        if len(nms_indices) > 0:
            self.debug_final_printed = True
    
        return final_detections


    def draw_detections(self, image_to_draw_on, detections):
        """Draw detections (using normalized coords) onto the provided image."""
        img_height, img_width = image_to_draw_on.shape[:2]
        # --- Drawing logic (remains the same) ---
        for det in detections:
            box = det['box'] # normalized [x1, y1, x2, y2]
            x1 = int(box[0] * img_width)
            y1 = int(box[1] * img_height)
            x2 = int(box[2] * img_width)
            y2 = int(box[3] * img_height)

            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            if x2 <= x1 or y2 <= y1: continue

            class_id = det['class_id']
            confidence = det['confidence']
            if class_id < 0 or class_id >= len(classNames): continue
            class_name = classNames[class_id]

            cv2.rectangle(image_to_draw_on, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_bg_y1 = max(y1 - text_size[1] - 5, 0)
            label_bg_y2 = y1 - 5
            if label_bg_y1 < 0:
                 label_bg_y1 = y1 + 5
                 label_bg_y2 = y1 + text_size[1] + 5

            cv2.rectangle(image_to_draw_on, (x1, label_bg_y1), (x1 + text_size[0] + 5, label_bg_y2), (0, 255, 0), -1)
            text_y = label_bg_y2 - 3 if label_bg_y1 < y1 else label_bg_y1 + text_size[1] # Simplified y-pos calc
            cv2.putText(image_to_draw_on, label, (x1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image_to_draw_on

# --- Main Function ---
def main():
    print("Starting YOLOv8 TensorRT Detection (Optimized for FPS)...")
    print(" Reminder: Check power mode (sudo nvpmodel -m 0) and thermals (sudo tegrastats).")

    # --- Configuration ---
    engine_file_path = "yolov8n.engine"
    confidence_threshold = 0.25 # Start here, tune based on dodgy detections vs missed objects
    nms_threshold = 0.45       # IoU threshold for NMS

    # --- Initialize Model FIRST to get its expected input size ---
    try:
        model = TensorRTDetector(
            engine_file_path,
            conf_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )
        model_input_width = model.width
        model_input_height = model.height
        print(f"Model loaded! Expects input size: {model_input_width}x{model_input_height}")
    except Exception as e:
        print(f"FATAL: Error initializing TensorRT model: {e}")
        traceback.print_exc()
        return

    # --- Camera Settings Optimized for Speed ---
    # Capture lower resolution, output size matches model input
    capture_width = 640
    capture_height = 480 # Common efficient capture size
    output_width = model_input_width   # GStreamer outputs this size (matches model)
    output_height = model_input_height # GStreamer outputs this size (matches model)
    framerate = 30 # Target framerate
    flip_method = 0

    window_name = f"YOLOv8 TensorRT ({output_width}x{output_height}) - Press ESC or Q to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # Use AUTOSIZE for fixed size window

    # --- Initialize Camera ---
    pipeline = gstreamer_pipeline(
        sensor_id=0,
        capture_width=capture_width,
        capture_height=capture_height,
        output_width=output_width, # Tell GStreamer to resize to model input size
        output_height=output_height,
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
             return
        else:
             # If using fallback, OpenCV gets frames at camera's default res, resize needed
             # We'll handle this by checking frame size in the loop (less efficient)
             print(f"Using default camera. Manual resize to {model_input_width}x{model_input_height} will be needed in loop.")
             needs_manual_resize = True
    else:
        needs_manual_resize = False # GStreamer pipeline handles resizing

    print("Camera opened successfully!")

    # --- Timing and FPS Variables ---
    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    processing_times = [] # For averaging detection time

    # --- Main Loop ---
    try:
        while True:
            t_start_loop = time.time()

            # --- Read Frame ---
            t_start_read = time.time()
            ret, frame = cap.read() # Frame should be output_width x output_height if GStreamer worked
            t_end_read = time.time()

            if not ret or frame is None:
                print("Warning: Failed to get frame. Skipping.")
                time.sleep(0.1)
                continue

            # --- Prepare Frame for Model (Only if using fallback camera) ---
            frame_for_model = frame # Assume frame is already correct size
            if needs_manual_resize:
                 # Check if frame size matches model input, resize if necessary
                 if frame.shape[1] != model_input_width or frame.shape[0] != model_input_height:
                      frame_for_model = cv2.resize(frame, (model_input_width, model_input_height), interpolation=cv2.INTER_LINEAR)
                 # Note: Display will still be the original capture size from fallback camera

            # --- Detect Objects ---
            detections = []
            t_start_detect = time.time()
            try:
                # Pass the correctly sized frame to the detection method
                detections = model.detect(frame_for_model)
            except Exception as detect_error:
                print(f"Error during model detection: {detect_error}")
            t_end_detect = time.time()

            # --- Draw Detections ---
            # Draw on the frame received from cap.read() (or the resized one if fallback)
            # If using GStreamer optimization, frame = frame_for_model
            display_frame = frame # Start with the frame we'll display
            t_start_draw = time.time()
            try:
                 # Pass the frame that will be displayed
                 display_frame = model.draw_detections(frame.copy(), detections)
            except Exception as draw_error:
                print(f"Error during drawing detections: {draw_error}")
            t_end_draw = time.time()

            # --- Calculate Timings and FPS ---
            t_end_loop = time.time()
            detect_time_ms = (t_end_detect - t_start_detect) * 1000
            processing_times.append(detect_time_ms)
            if len(processing_times) > 60: processing_times.pop(0) # Average over more frames
            avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0

            frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                fps_start_time = time.time()

            # --- Display Info on Frame ---
            fps_text = f"FPS: {fps:.1f}"
            inf_text = f"Detect+NMS: {avg_process_time:.1f}ms"
            cv2.putText(display_frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, inf_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- Show Frame ---
            cv2.imshow(window_name, display_frame)

            # --- Exit Condition ---
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                print(f"{chr(key)} key pressed, exiting...")
                break

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        traceback.print_exc()
    finally:
        print("Releasing resources...")
        if cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main()
