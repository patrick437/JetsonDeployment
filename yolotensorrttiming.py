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
# Optimized for speed: Capture lower res, nvvidconv resizes to model input size
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,  # Lower capture resolution for performance
    capture_height=480,
    output_width=640,   # Target output width (e.g., model input width)
    output_height=480,  # Target output height (e.g., model input height)
    framerate=30,
    flip_method=0, # 0=None, 1=CCW, 2=180, 3=CW, 4=HFlip, 5=UR-Diag, 6=VFlip, 7=UL-Diag
):
    return (
        "nvarguscamerasrc sensor-id=%d bufapi-version=1 ! " # Added bufapi-version for potentially better perf
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        # Output BGRx format, let nvvidconv handle conversion + scaling
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! " # videoconvert to convert BGRx to BGR
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

# --- TensorRT Detector Class ---
class TensorRTDetector:
    def __init__(self, engine_path, conf_threshold=0.25, nms_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold # IoU Threshold for NMS

        # Initialize TensorRT
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

        # Get input binding details
        self.input_binding_idx = -1
        self.input_name = ""
        for i in range(self.engine.num_bindings):
             if self.engine.binding_is_input(i):
                  self.input_binding_idx = i
                  self.input_name = self.engine.get_binding_name(i)
                  break
        if self.input_binding_idx == -1: raise ValueError("Input binding not found.")
        print(f"Found input binding '{self.input_name}' at index {self.input_binding_idx}")

        # Get input shape (handling dynamic dimensions, assume batch=1)
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

        # Initialize timing variables
        self.last_preprocess_ms = 0.0
        self.last_inference_ms = 0.0 # Includes HtoD, kernel execution, DtoH, sync
        self.last_postprocess_ms = 0.0
        # Debug flags
        self.debug_output_printed = False
        self.debug_final_printed = False # Controls detailed score printing
        self.nms_debug_printed = False   # Controls pre-NMS count printing

        print("Detector initialized. Timing variables ready.")


    def preprocess(self, image):
        """Preprocess image (already resized to model input size)."""
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1) # HWC -> CHW
        input_img = np.expand_dims(input_img, axis=0) # Add batch dim
        input_img = np.ascontiguousarray(input_img)
        return input_img


    # --- DETECT METHOD (WITH NMS BYPASSED, RAW SCORE PRINTING, AND DETAILED TIMING) ---
    def detect(self, frame_for_model):
        """
        Run inference, return detections. Calculates detailed step timings.
        *** NOTE: NMS IS BYPASSED in this version for debugging. ***
        *** Includes detailed raw score printing for the first frame with detections. ***
        """
        t0_start_preprocess = time.time()

        # --- Preprocessing ---
        input_img = self.preprocess(frame_for_model)
        if input_img.shape != self.inputs[0]['shape']:
             raise ValueError(f"Mismatched input shape! Expected {self.inputs[0]['shape']} but got {input_img.shape}.")
        # --- End Preprocessing ---

        t1_start_inference = time.time() # End preprocess, Start HtoD

        # --- Perform Inference (HtoD, Execute, DtoH, Sync) ---
        np.copyto(self.inputs[0]['host'], input_img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize() # Wait for all GPU operations to complete
        # --- End Inference ---

        t2_start_postprocess = time.time() # End sync, Start post-processing

        # --- Post-processing ---
        # Get Output
        output_data = self.outputs[0]['host']
        output_shape = self.outputs[0]['shape']

        # Print Raw shape once
        if not self.debug_output_printed:
            print(f"Raw Output shape: {output_shape}")
            self.debug_output_printed = True

        # Reshape/Transpose Output
        num_classes = len(classNames)
        num_coords = 4
        expected_channels = num_classes + num_coords
        processed_output = None
        # ... (Reshape/Transpose logic) ...
        # Case 1: [B, C, D]
        if len(output_shape) == 3 and output_shape[0] == self.batch_size and output_shape[1] == expected_channels:
             processed_output = output_data.reshape(output_shape).transpose(0, 2, 1) # -> (B, D, C)
        # Case 2: [B, D, C]
        elif len(output_shape) == 3 and output_shape[0] == self.batch_size and output_shape[2] == expected_channels:
             processed_output = output_data.reshape(output_shape)
        # Case 3: Flattened [B, D*C]
        elif len(output_shape) == 2 and output_shape[0] == self.batch_size:
             total_elements = output_shape[1]
             if total_elements % expected_channels == 0:
                  num_detections = total_elements // expected_channels
                  try: processed_output = output_data.reshape((self.batch_size, num_detections, expected_channels))
                  except ValueError: processed_output = None
             else: processed_output = None
        if processed_output is None:
            print("Error processing output shape, skipping post-processing.")
            t3_end_postprocess = time.time()
            self.last_preprocess_ms = (t1_start_inference - t0_start_preprocess) * 1000
            self.last_inference_ms = (t2_start_postprocess - t1_start_inference) * 1000
            self.last_postprocess_ms = (t3_end_postprocess - t2_start_postprocess) * 1000
            return []

        # Initial Filtering by Confidence
        detections_batch = processed_output[0]
        boxes_xywh = detections_batch[:, :num_coords]
        all_scores = detections_batch[:, num_coords:] # (num_detections, num_classes)
        class_ids = np.argmax(all_scores, axis=1)
        max_confidences = np.max(all_scores, axis=1)
        keep_indices_in_batch = np.where(max_confidences >= self.conf_threshold)[0]

        if len(keep_indices_in_batch) == 0:
            # Reset flags if no detections found, allow printing next time
            self.nms_debug_printed = False
            self.debug_final_printed = False
            t3_end_postprocess = time.time()
            self.last_preprocess_ms = (t1_start_inference - t0_start_preprocess) * 1000
            self.last_inference_ms = (t2_start_postprocess - t1_start_inference) * 1000
            self.last_postprocess_ms = (t3_end_postprocess - t2_start_postprocess) * 1000
            return []

        # Get filtered data
        filtered_boxes_xywh = boxes_xywh[keep_indices_in_batch]
        filtered_confidences = max_confidences[keep_indices_in_batch]
        filtered_class_ids = class_ids[keep_indices_in_batch]
        filtered_raw_scores = all_scores[keep_indices_in_batch]
        x_center, y_center, width, height = filtered_boxes_xywh.T
        x1 = x_center - width / 2; y1 = y_center - height / 2
        x2 = x_center + width / 2; y2 = y_center + height / 2
        filtered_boxes_xyxy = np.stack((x1, y1, x2, y2), axis=1)

        # Print pre-NMS count once
        if not self.nms_debug_printed:
             print(f"DEBUG: Found {len(filtered_boxes_xywh)} detections pre-NMS (Conf > {self.conf_threshold:.2f})")
             self.nms_debug_printed = True

        # --- NMS STEP BYPASSED FOR DEBUGGING ---
        indices_to_use = list(range(len(filtered_boxes_xywh)))
        # print(f"DEBUG: NMS bypassed. Using all {len(indices_to_use)} pre-NMS detections for output.")
        # --- END OF NMS BYPASS ---

        # Final List Creation + Raw Score Printing (if flag allows)
        final_detections = []
        print_scores_this_frame = (not self.debug_final_printed and len(indices_to_use) > 0)

        if print_scores_this_frame:
            print(f"\n--- DETAILED SCORES FOR FRAME (Pre-NMS, Conf > {self.conf_threshold:.2f}) ---")

        for i, idx in enumerate(indices_to_use):
             box_xyxy = filtered_boxes_xyxy[idx]
             confidence = filtered_confidences[idx]
             class_id = filtered_class_ids[idx]
             raw_scores_for_this_box = filtered_raw_scores[idx]

             # Print Raw Score Details
             if print_scores_this_frame:
                 # Limit how many boxes' details are printed if there are too many
                 if i < 15: # Print details for first 15 boxes only
                     print(f" Detection {i}: ArgMax Class={classNames[class_id]}({class_id}), MaxConf={confidence:.3f}")
                     top_n = 5
                     top_indices = np.argsort(raw_scores_for_this_box)[::-1][:top_n]
                     print(f"   Top {top_n} Raw Scores:")
                     for score_idx in top_indices:
                         score_val = raw_scores_for_this_box[score_idx]
                         if score_val > 0.01:
                             if 0 <= score_idx < len(classNames):
                                 print(f"     - {classNames[score_idx]}({score_idx}): {score_val:.4f}")
                             else: print(f"     - Invalid Class ID {score_idx}: {score_val:.4f}")
                     print("-" * 10)
                 elif i == 15:
                     print(f" (Skipping detailed scores for remaining {len(indices_to_use) - 15} detections...)")


             # Clamp and validate box
             box_xyxy[0] = max(0.0, min(1.0, box_xyxy[0])) # x1
             box_xyxy[1] = max(0.0, min(1.0, box_xyxy[1])) # y1
             box_xyxy[2] = max(0.0, min(1.0, box_xyxy[2])) # x2
             box_xyxy[3] = max(0.0, min(1.0, box_xyxy[3])) # y2

             if box_xyxy[0] < box_xyxy[2] and box_xyxy[1] < box_xyxy[3]:
                  final_detections.append({
                       'box': box_xyxy.tolist(),
                       'confidence': float(confidence),
                       'class_id': int(class_id)
                  })

        # Set flag after processing detections for this frame
        if print_scores_this_frame:
             self.debug_final_printed = True # Prevents printing scores every frame
        # --- End Post-processing ---

        t3_end_postprocess = time.time()

        # --- Store Timings ---
        self.last_preprocess_ms = (t1_start_inference - t0_start_preprocess) * 1000
        self.last_inference_ms = (t2_start_postprocess - t1_start_inference) * 1000
        self.last_postprocess_ms = (t3_end_postprocess - t2_start_postprocess) * 1000

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
            text_y = label_bg_y2 - 3 if label_bg_y1 < y1 else label_bg_y1 + text_size[1]
            cv2.putText(image_to_draw_on, label, (x1 + 3, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image_to_draw_on

# --- Main Function ---
def main():
    # Timestamp and Location Data (as per context request)
    # Current time is Thursday, April 10, 2025 at 9:47:49 AM IST.
    # Location is Galway, County Galway, Ireland.
    print(f"Script started: {time.strftime('%Y-%m-%d %H:%M:%S')}") # Use current time
    print("Location context: Galway, County Galway, Ireland")
    print("Starting YOLOv8 TensorRT Detection (Debugging Version)...")
    print(" Reminder: Check power mode (sudo nvpmodel -m 0) and thermals (sudo tegrastats).")
    print(" NOTE: NMS is currently BYPASSED in detect() for debugging.")

    # --- Configuration ---
    engine_file_path = "yolov8n.engine"
    # Start with a moderate confidence; adjust based on raw score printouts
    confidence_threshold = 0.25
    nms_threshold = 0.45 # IoU threshold (only relevant if NMS is re-enabled)

    # --- Initialize Model FIRST ---
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
    capture_width = 640
    capture_height = 480
    # GStreamer output size matches model input size for efficiency
    output_width = model_input_width
    output_height = model_input_height
    framerate = 30
    flip_method = 0 # Adjust if necessary

    window_name = f"YOLOv8 TRT Debug ({output_width}x{output_height}) | NMS BYPASSED"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # --- Initialize Camera ---
    pipeline = gstreamer_pipeline(
        sensor_id=0,
        capture_width=capture_width,
        capture_height=capture_height,
        output_width=output_width, # GStreamer resizes to model input size
        output_height=output_height,
        framerate=framerate,
        flip_method=flip_method
    )
    print("Starting camera with GStreamer pipeline:")
    print(pipeline)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    needs_manual_resize = False # Assume GStreamer works

    if not cap.isOpened():
        print("ERROR: Failed to open camera via GStreamer! Fallback...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
             print("ERROR: Failed to open default camera.")
             return
        else:
             print(f"Using default camera. Manual resize will be needed.")
             needs_manual_resize = True

    print("Camera opened successfully!")

    # --- Timing and FPS Variables ---
    frame_count = 0
    fps = 0
    fps_start_time = time.time()

    # --- Main Loop ---
    try:
        while True:
            t_start_loop = time.time()

            # --- Read Frame ---
            ret, frame = cap.read() # Frame should be output_width x output_height
            if not ret or frame is None:
                print("Warning: Failed to get frame. Skipping.")
                time.sleep(0.1)
                continue

            # --- Prepare Frame for Model (Only if using fallback camera) ---
            frame_for_model = frame
            if needs_manual_resize:
                 if frame.shape[1] != model_input_width or frame.shape[0] != model_input_height:
                      t_manual_resize_start = time.time()
                      frame_for_model = cv2.resize(frame, (model_input_width, model_input_height), interpolation=cv2.INTER_LINEAR)
                      # print(f"Manual resize took: {(time.time() - t_manual_resize_start)*1000:.1f} ms") # Optional timing
                 # Note: display will be original fallback camera size

            # --- Detect Objects (Gets detailed timings now) ---
            detections = []
            try:
                # Pass the correctly sized frame to the detection method
                detections = model.detect(frame_for_model)
            except Exception as detect_error:
                print(f"Error during model detection: {detect_error}")
                traceback.print_exc() # Print full error

            # --- Draw Detections ---
            # Draw on the frame received from cap.read()
            display_frame = frame
            t_start_draw = time.time()
            try:
                 display_frame = model.draw_detections(frame.copy(), detections)
            except Exception as draw_error:
                print(f"Error during drawing detections: {draw_error}")
            t_end_draw = time.time()

            # --- Calculate Timings and FPS ---
            t_end_loop = time.time()
            # Get detailed timings from the model object
            preprocess_ms = model.last_preprocess_ms
            inference_ms = model.last_inference_ms
            postprocess_ms = model.last_postprocess_ms
            draw_time_ms = (t_end_draw - t_start_draw) * 1000
            total_loop_time_ms = (t_end_loop - t_start_loop) * 1000

            # Calculate overall FPS
            frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                fps_start_time = time.time()

            # --- Display Info on Frame ---
            fps_text = f"FPS: {fps:.1f}"
            timing_text1 = f"Pre:{preprocess_ms:.1f} Inf:{inference_ms:.1f}"
            timing_text2 = f"Post:{postprocess_ms:.1f} Draw:{draw_time_ms:.1f}"
            total_text = f"Loop:{total_loop_time_ms:.1f}ms"

            cv2.putText(display_frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, timing_text1, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display_frame, timing_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display_frame, total_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
