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
        
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Run inference
        try:
            # Run model and get results - handle different formats
            results = model(frame)
            
            # Process results - handle both dictionary and object formats
            if results is not None:
                # Handle results as list of objects
                if isinstance(results, list):
                    for result in results:
                        if hasattr(result, 'boxes'):
                            boxes = result.boxes
                            for box in boxes:
                                try:
                                    # Get box coordinates
                                    if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                                        coords = box.xyxy[0].tolist()
                                        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                                        
                                        # Get confidence and class ID
                                        conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.0
                                        cls_id = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else 0
                                        
                                        # Draw if confidence is high enough
                                        if conf >= 0.5:
                                            # Draw bounding box
                                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            
                                            # Create label
                                            class_name = classNames[cls_id] if cls_id < len(classNames) else f"Class {cls_id}"
                                            label = f"{class_name} {conf:.2f}"
                                            
                                            # Draw label background
                                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                            cv2.rectangle(
                                                display_frame,
                                                (x1, y1 - text_size[1] - 5),
                                                (x1 + text_size[0] + 5, y1),
                                                (0, 255, 0),
                                                -1
                                            )
                                            
                                            # Draw text
                                            cv2.putText(
                                                display_frame,
                                                label,
                                                (x1 + 3, y1 - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5,
                                                (0, 0, 0),
                                                2
                                            )
                                except Exception as box_error:
                                    print(f"Box processing error: {box_error}")
                
                # Alternative: Handle results as a single object
                elif hasattr(results, 'boxes'):
                    boxes = results.boxes
                    for box in boxes:
                        try:
                            # Get box coordinates
                            if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                                coords = box.xyxy[0].tolist()
                                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                                
                                # Get confidence and class ID
                                conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.0
                                cls_id = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else 0
                                
                                # Draw if confidence is high enough
                                if conf >= 0.5:
                                    # Draw bounding box
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Create label
                                    class_name = classNames[cls_id] if cls_id < len(classNames) else f"Class {cls_id}"
                                    label = f"{class_name} {conf:.2f}"
                                    
                                    # Draw label background
                                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                    cv2.rectangle(
                                        display_frame,
                                        (x1, y1 - text_size[1] - 5),
                                        (x1 + text_size[0] + 5, y1),
                                        (0, 255, 0),
                                        -1
                                    )
                                    
                                    # Draw text
                                    cv2.putText(
                                        display_frame,
                                        label,
                                        (x1 + 3, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 0, 0),
                                        2
                                    )
                        except Exception as box_error:
                            print(f"Box processing error: {box_error}")
                
                # Handle numpy array-like results (direct tensor output)
                elif isinstance(results, np.ndarray) or (hasattr(results, '__array__') and callable(results.__array__)):
                    detections = np.array(results)
                    if detections.size > 0:
                        # Assume format [x1, y1, x2, y2, conf, class_id]
                        for detection in detections:
                            if len(detection) >= 6:
                                x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
                                conf = float(detection[4])
                                cls_id = int(detection[5])
                                
                                if conf >= 0.5:
                                    # Draw bounding box
                                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # Create label
                                    class_name = classNames[cls_id] if cls_id < len(classNames) else f"Class {cls_id}"
                                    label = f"{class_name} {conf:.2f}"
                                    
                                    # Draw label
                                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                    cv2.rectangle(
                                        display_frame,
                                        (x1, y1 - text_size[1] - 5),
                                        (x1 + text_size[0] + 5, y1),
                                        (0, 255, 0),
                                        -1
                                    )
                                    cv2.putText(
                                        display_frame,
                                        label,
                                        (x1 + 3, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 0, 0),
                                        2
                                    )
                
        except Exception as infer_error:
            print(f"Inference error: {infer_error}")
        
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
