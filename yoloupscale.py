from ultralytics import YOLO
import cv2
import time
import numpy as np

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
    # ... (rest of your class names)
]

# Create window
window_name = "YOLOv8-TensorRT Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Load the YOLO model
print("Loading YOLOv8 TensorRT engine...")
model = YOLO("yolov8n.engine", task="detect")
print("Model loaded successfully!")

# Camera settings
capture_width = 1280
capture_height = 720
processing_width = 640
processing_height = 640
framerate = 30
flip_method = 0

# Create pipeline
pipeline = gstreamer_pipeline(
    sensor_id=0,
    capture_width=capture_width,
    capture_height=capture_height,
    display_width=processing_width,
    display_height=processing_height,
    framerate=framerate,
    flip_method=flip_method
)

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

# Calculate scaling factors
width_scale = capture_width / processing_width
height_scale = capture_height / processing_height

try:
    while True:
        # Read frame (already resized to 640x640 by GStreamer)
        ret, processed_frame = cap.read()
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
        
        # Create a display frame by upscaling
        display_frame = cv2.resize(processed_frame, (capture_width, capture_height))
        
        # Run inference on the processed frame (640x640)
        try:
            results = model(processed_frame, verbose=False)
            
            if results is not None:
                for result in results:
                    if hasattr(result, 'boxes'):
                        boxes = result.boxes
                        for box in boxes:
                            # Get box coordinates in xyxy format (640x640 space)
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Scale coordinates to display resolution (1280x720)
                            x1 = int(x1 * width_scale)
                            y1 = int(y1 * height_scale)
                            x2 = int(x2 * width_scale)
                            y2 = int(y2 * height_scale)
                            
                            # Get confidence and class ID
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            
                            # Draw if confidence is high enough
                            if conf >= 0.5:
                                # Draw bounding box on display frame
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Create label
                                class_name = classNames[cls_id] if cls_id < len(classNames) else str(cls_id)
                                label = f"{class_name} {conf:.2f}"
                                
                                # Calculate text size
                                (text_width, text_height), _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                                
                                # Draw label background
                                cv2.rectangle(
                                    display_frame,
                                    (x1, y1 - text_height - 10),
                                    (x1 + text_width + 10, y1),
                                    (0, 255, 0),
                                    -1
                                )
                                
                                # Draw text
                                cv2.putText(
                                    display_frame,
                                    label,
                                    (x1 + 5, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    2
                                )
        
        except Exception as infer_error:
            print(f"Inference error: {infer_error}")
            continue
        
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
        
        # Display the upscaled frame with detections
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
