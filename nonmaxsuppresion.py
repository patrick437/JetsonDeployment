import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import warnings

# Suppress numpy deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ... (gstreamer_pipeline and classNames remain the same) ...
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

# Simple NMS (you already have this, kept for completeness)
def non_max_suppression_fast(boxes, scores, overlap_thresh):
    # ... (your existing NMS function) ...
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

    area = (x2 - x1) * (y2 - y1) # Use adjusted area calculation if needed
    idxs = np.argsort(scores) # Ascending order

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find intersection
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute width and height of the intersection
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete indices associated with overlaps greater than the threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    # Return only the indices of the picked boxes
    # The actual boxes/scores need to be selected using these indices outside this function
    return pick


class TensorRTDetector:
    def __init__(self, engine_path, conf_threshold=0.5, nms_threshold=0.4): # Added nms_threshold
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold # Store NMS threshold

        # ... (TRT Initialization, Engine Loading, Context Creation) ...
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")

        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # ... (Get Input Binding - improved robustness) ...
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


        # ... (Get Input Shape - handling dynamic batch) ...
        self.input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        print(f"Original engine input shape: {self.input_shape}")
        # Handle dynamic dimensions, assume batch is 1 if dynamic
        if self.input_shape[0] == -1:
             # Check if context already has shape set (might be needed for some engines)
            try:
                 active_shape = self.context.get_binding_shape(self.input_binding_idx)
                 if active_shape[0] != -1:
                     self.input_shape = active_shape
                     print(f"Using active context shape: {self.input_shape}")
                 else:
                      # Set default batch size of 1
                      self.input_shape = (1,) + tuple(self.input_shape[1:])
                      print(f"Input shape has dynamic batch, setting to: {self.input_shape}")
                      self.context.set_binding_shape(self.input_binding_idx, self.input_shape)
            except Exception as e:
                 print(f"Could not get/set context binding shape, setting default: {e}")
                 self.input_shape = (1,) + tuple(self.input_shape[1:])


        self.batch_size, self.channels, self.height, self.width = self.input_shape
        print(f"Using Input Shape (B, C, H, W): {self.batch_size}, {self.channels}, {self.height}, {self.width}")


        # ... (Create Buffers - handling dynamic shapes more carefully) ...
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            # Use shape from context if possible, otherwise from engine (handling dynamics)
            try:
                binding_shape = self.context.get_binding_shape(i)
                # If still dynamic after setting input, default to max batch size or 1
                if binding_shape[0] == -1:
                   binding_shape = (self.batch_size,) + tuple(binding_shape[1:]) # Use determined batch size
                   print(f"Resolved dynamic shape for binding {i} ({binding_name}): {binding_shape}")
            except Exception as e:
                 print(f"Using engine shape for binding {i} ({binding_name}), may need context setting: {e}")
                 binding_shape = self.engine.get_binding_shape(i)
                 if binding_shape[0] == -1:
                     binding_shape = (self.batch_size,) + tuple(binding_shape[1:]) # Use determined batch size
                     print(f"Resolved dynamic shape for binding {i} ({binding_name}): {binding_shape}")


            size = trt.volume(binding_shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape, 'name': binding_name})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': binding_shape, 'name': binding_name})
                print(f"Output binding '{binding_name}' at index {i} with shape {binding_shape} and dtype {dtype}")


        # Check we have at least one output
        if not self.outputs:
             raise ValueError("Could not find any output bindings in the engine")

        print(f"Detector initialized. Input: {self.inputs[0]['name']} {self.inputs[0]['shape']}. Output: {self.outputs[0]['name']} {self.outputs[0]['shape']}")


    def preprocess(self, image):
        """Preprocess image for inference - correctly handling color conversion"""
        # Resize
        input_img = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR) # Use INTER_LINEAR for better quality

        # Convert BGR to RGB (YOLOv8 expects RGB)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        # Convert to float32 and normalize
        input_img = input_img.astype(np.float32) / 255.0

        # Transpose from HWC to CHW format (Height, Width, Channels) -> (Channels, Height, Width)
        input_img = input_img.transpose(2, 0, 1)

        # Add batch dimension
        input_img = np.expand_dims(input_img, axis=0) # Shape: (1, C, H, W)

        # Ensure the input is contiguous in memory
        input_img = np.ascontiguousarray(input_img)

        return input_img

    def detect(self, image):
        """Run inference on an image and return detections"""
        original_height, original_width = image.shape[:2] # Store original image size

        # Preprocess the image
        input_img = self.preprocess(image)

        # Verify input shape matches buffer
        if input_img.shape != self.inputs[0]['shape']:
             print(f"Warning: Preprocessed image shape {input_img.shape} does not match expected input shape {self.inputs[0]['shape']}")
             # Attempt to reshape if only batch size differs and batch size is 1
             if input_img.shape[1:] == self.inputs[0]['shape'][1:] and self.inputs[0]['shape'][0] == 1:
                  print("Adjusting batch size dimension.")
             else:
                  # Resize might be needed if model expects different H/W
                  raise ValueError("Mismatched input shape after preprocessing.")


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
        # Assuming the first output is the main detection output
        output_data = self.outputs[0]['host']
        output_shape = self.outputs[0]['shape'] # Should be like (1, 84, 8400) or similar

        if not hasattr(self, 'debug_printed'):
            print(f"Raw Output shape: {output_shape}") # Crucial debug print
            print(f"Raw Output sample (first 20 elements): {output_data[:20]}")
            self.debug_printed = True

        # Reshape the output if it's flattened (check total elements)
        expected_elements = np.prod(output_shape)
        if output_data.size != expected_elements:
             print(f"Warning: Output data size {output_data.size} doesn't match expected elements {expected_elements} from shape {output_shape}. Trying to reshape anyway.")
             # This might indicate an issue with buffer allocation or engine definition

        # Make sure the output has 3 dimensions (batch, channels, detections)
        # Example target shape: (1, 84, 8400) for YOLOv8n COCO
        num_classes = len(classNames) # 80
        num_coords = 4
        expected_channels = num_classes + num_coords # Should be 84

        # Handle potential flattened output
        if len(output_shape) == 1:
             # Try to infer shape, assuming batch_size=1
             try:
                  inferred_detections = output_shape[0] // expected_channels
                  if output_shape[0] % expected_channels == 0:
                       output_shape = (1, expected_channels, inferred_detections)
                       print(f"Inferred output shape: {output_shape}")
                  else:
                       raise ValueError("Cannot infer shape from flattened output")
             except Exception as e:
                  print(f"Error inferring shape: {e}. Cannot process flattened output of size {output_shape[0]}")
                  return [] # Cannot proceed
        elif len(output_shape) == 2:
             # Maybe [batch, features*detections]? Assume batch=1
             if output_shape[0] == 1:
                   try:
                        inferred_detections = output_shape[1] // expected_channels
                        if output_shape[1] % expected_channels == 0:
                             output_shape = (1, expected_channels, inferred_detections)
                             print(f"Inferred output shape from 2D: {output_shape}")
                        else:
                             raise ValueError("Cannot infer shape from 2D output")
                   except Exception as e:
                        print(f"Error inferring shape from 2D: {e}. Cannot process output shape {output_shape}")
                        return []
             else: # Maybe [detections, features]? Unlikely for standard export
                  print(f"Unexpected 2D output shape {output_shape}. Cannot process.")
                  return []


        # Reshape the flat array according to the (potentially inferred) shape
        try:
             output_data = output_data.reshape(output_shape)
        except ValueError as e:
            print(f"Error reshaping output data with size {output_data.size} to {output_shape}: {e}")
            return [] # Cannot proceed if reshape fails


        # Check channel dimension (should be num_classes + 4)
        if output_shape[1] != expected_channels:
            print(f"Warning: Output channel dimension ({output_shape[1]}) doesn't match expected ({expected_channels}). Processing might fail.")
            # Adjust expected channels if it seems slightly different (e.g. +1 for confidence sometimes)
            if abs(output_shape[1] - expected_channels) < 5: # Allow for slight variations
                 print(f"Adjusting expected channels to {output_shape[1]} for processing.")
                 expected_channels = output_shape[1]
            else:
                 print("Channel mismatch too large. Aborting detection processing.")
                 return []


        # Transpose the output from [batch, channels, detections] to [batch, detections, channels]
        # So from (1, 84, 8400) to (1, 8400, 84)
        try:
            output_data = output_data.transpose(0, 2, 1) # Now shape is (batch, num_detections, num_coords + num_classes)
            print(f"Transposed output shape: {output_data.shape}") # Should be like (1, 8400, 84)
        except ValueError as e:
             print(f"Error transposing output data with shape {output_data.shape}: {e}")
             return []


        # Process detections for the first batch
        detections_batch = output_data[0] # Shape: (num_detections, num_coords + num_classes) e.g., (8400, 84)

        # Extract boxes, confidences, and class scores
        boxes_xywh = detections_batch[:, :num_coords]      # First 4 columns: cx, cy, w, h (normalized)
        all_scores = detections_batch[:, num_coords:] # Remaining columns: class scores

        # Find the highest score for each detection and its class ID
        class_ids = np.argmax(all_scores, axis=1)
        max_confidences = np.max(all_scores, axis=1) # This is the confidence for the predicted class


        # Filter detections based on confidence threshold
        keep = max_confidences >= self.conf_threshold
        filtered_boxes_xywh = boxes_xywh[keep]
        filtered_confidences = max_confidences[keep]
        filtered_class_ids = class_ids[keep]

        if len(filtered_boxes_xywh) == 0:
             #print("No detections above confidence threshold.")
             return [] # No detections passed the confidence threshold

        print(f"Found {len(filtered_boxes_xywh)} detections above confidence threshold {self.conf_threshold}")

        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] (still normalized)
        x_center, y_center, width, height = filtered_boxes_xywh.T
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        filtered_boxes_xyxy = np.stack((x1, y1, x2, y2), axis=1)


        # Apply Non-Maximum Suppression (NMS)
        # Use cv2.dnn.NMSBoxes for potentially better performance if available
        try:
            # OpenCV NMS expects boxes as list of [x, y, w, h] and requires integer coords usually,
            # but let's adapt our normalized [x1, y1, x2, y2]
            # We need to scale to temporary pixel coords for NMS function if it requires it,
            # Or use a custom NMS that works with normalized floats.
            # Using the provided `non_max_suppression_fast` which works on [x1, y1, x2, y2] floats:

            indices_to_keep = non_max_suppression_fast(filtered_boxes_xyxy, filtered_confidences, self.nms_threshold)

            # Alternative using cv2.dnn.NMSBoxes (might need scaling/type adjustments)
            # Note: cv2.dnn.NMSBoxes expects boxes as list of [x_center, y_center, width, height] typically,
            # or sometimes [x1, y1, width, height]. Check documentation. Sticking to custom NMS for now.
            # boxes_for_nms = [[int(c[0]*self.width), int(c[1]*self.height), int(c[2]*self.width), int(c[3]*self.height)] for c in filtered_boxes_xywh] # Example scaling
            # indices_to_keep = cv2.dnn.NMSBoxes(boxes_for_nms, filtered_confidences.tolist(), self.conf_threshold, self.nms_threshold)
            # if isinstance(indices_to_keep, tuple): # Handle older OpenCV versions returning tuples
            #      indices_to_keep = indices_to_keep[0]
            # indices_to_keep = indices_to_keep.flatten() # Ensure it's a flat array/list

        except Exception as e:
            print(f"Error during NMS: {e}. Using detections before NMS.")
            indices_to_keep = list(range(len(filtered_boxes_xyxy))) # Fallback: keep all filtered boxes

        # Prepare final detections list
        final_detections = []
        for idx in indices_to_keep:
             box = filtered_boxes_xyxy[idx]
             confidence = filtered_confidences[idx]
             class_id = filtered_class_ids[idx]

             # Ensure box coordinates are valid (x1 < x2, y1 < y2) and within [0, 1]
             box[0] = max(0.0, box[0]) # x1
             box[1] = max(0.0, box[1]) # y1
             box[2] = min(1.0, box[2]) # x2
             box[3] = min(1.0, box[3]) # y2

             if box[0] < box[2] and box[1] < box[3]: # Check validity after clamping
                  final_detections.append({
                       'box': box.tolist(), # [x1, y1, x2, y2] normalized
                       'confidence': float(confidence),
                       'class_id': int(class_id)
                  })

        # Debug print final detections count
        if len(final_detections) > 0 and not hasattr(self, 'final_det_printed'):
             print(f"Found {len(final_detections)} final detections after NMS.")
             self.final_det_printed = True
        elif len(filtered_boxes_xywh) > 0 and len(final_detections) == 0:
             print("All detections were suppressed by NMS.")


        return final_detections


    def draw_detections(self, image, detections):
        """Draw detections on image - uses normalized coordinates from detect()"""
        img_height, img_width = image.shape[:2]

        for det in detections:
            # Get normalized box coordinates [x1, y1, x2, y2]
            box = det['box']
            x1_norm, y1_norm, x2_norm, y2_norm = box

            # Convert normalized coordinates to pixel coordinates
            x1 = int(x1_norm * img_width)
            y1 = int(y1_norm * img_height)
            x2 = int(x2_norm * img_width)
            y2 = int(y2_norm * img_height)

            # Get class info
            class_id = det['class_id']
            confidence = det['confidence']

            # Skip invalid class IDs
            if class_id < 0 or class_id >= len(classNames):
                print(f"Warning: Skipping detection with invalid class_id {class_id}")
                continue

            class_name = classNames[class_id]

            # Ensure coordinates are valid after scaling
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            # Check if box is valid (sometimes NMS might leave tiny boxes)
            if x2 <= x1 or y2 <= y1:
                continue

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Create label
            label = f"{class_name} {confidence:.2f}"

            # Draw label background and text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_bg_y1 = max(y1 - text_size[1] - 5, 0) # Ensure background doesn't go off-screen top
            label_bg_y2 = y1 - 5
            cv2.rectangle(image, (x1, label_bg_y1), (x1 + text_size[0] + 5, label_bg_y2), (0, 255, 0), -1)
            cv2.putText(image, label, (x1 + 3, label_bg_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image

# ... (gstreamer_pipeline definition) ...
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

# Main function
def main():
    window_name = "YOLOv8-TensorRT Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Loading YOLOv8 TensorRT engine...")
    # Adjust thresholds as needed
    model = TensorRTDetector("yolov8n.engine", conf_threshold=0.3, nms_threshold=0.45)
    print("Model loaded successfully!")

    # Camera settings - match display to model input if possible, or keep lower if needed for performance
    # Note: The model preprocesses to its required size (e.g., 640x640) regardless of capture/display size
    capture_width = 640
    capture_height = 480
    display_width = 640  # Match capture for simplicity here
    display_height = 480 # Match capture for simplicity here
    framerate = 30
    flip_method = 0 # Adjust if your camera is upside down

    pipeline = gstreamer_pipeline(
        sensor_id=0,
        capture_width=capture_width,
        capture_height=capture_height,
        display_width=display_width, # Use the display size here
        display_height=display_height, # Use the display size here
        framerate=framerate,
        flip_method=flip_method
    )
    print("Starting camera with pipeline:", pipeline)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Failed to open camera!")
        # Try falling back to default camera if GStreamer fails
        print("Trying default camera (cv2.VideoCapture(0))...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
             print("Failed to open default camera either.")
             exit()

    print("Camera opened successfully!")

    frame_count = 0
    fps = 0
    fps_start_time = time.time()
    processing_times = []
    frame_skip_counter = 0
    frame_skip_rate = 1 # Process every frame initially

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to get frame or empty frame received.")
                time.sleep(0.1) # Wait a bit before retrying
                # Attempt to re-open capture if it consistently fails
                cap.release()
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if not cap.isOpened():
                    print("Re-opening camera failed.")
                    break
                continue


            # --- Frame Skipping (Optional) ---
            frame_skip_counter += 1
            if frame_skip_counter % frame_skip_rate != 0:
                 # If skipping, still display the *previous* processed frame for smoothness
                 # Or display the current frame without boxes
                 # For simplicity, let's just display the current frame without new boxes
                 # Add previous FPS/time text if desired
                 cv2.putText(frame, f"FPS: {fps:.1f} (Skipped Frame)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                 cv2.imshow(window_name, frame)
                 key = cv2.waitKey(1) & 0xFF
                 if key == 27: # ESC key
                     break
                 continue


            # --- Process Frame ---
            start_time = time.time()
            detections = [] # Reset detections for the frame
            try:
                # Detect objects
                detections = model.detect(frame) # frame is the BGR image from camera

                # Draw detections directly on the frame
                # Make a copy ONLY if you need the original frame later without boxes
                display_frame = model.draw_detections(frame, detections)

            except Exception as infer_error:
                print(f"Error during detection/drawing: {infer_error}")
                display_frame = frame # Show the original frame on error

            process_time = time.time() - start_time
            processing_times.append(process_time)
            if len(processing_times) > 30: # Average over last 30 processed frames
                processing_times.pop(0)
            avg_process_time = sum(processing_times) / len(processing_times) if processing_times else 0

            # --- FPS Calculation ---
            frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                fps_start_time = time.time()

            # --- Display ---
            # Add FPS and processing time to the frame
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f} | Inference: {avg_process_time*1000:.1f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1 # Thickness
            )

            # Display the frame (with detections drawn)
            cv2.imshow(window_name, display_frame)

            # --- Exit Condition ---
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
    finally:
        print("Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        # Clean up CUDA context if necessary (usually auto-handled by pycuda.autoinit)
        print("Done!")

if __name__ == "__main__":
    main()
