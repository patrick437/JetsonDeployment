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
