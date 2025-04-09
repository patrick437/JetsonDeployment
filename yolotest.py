from ultralytics import YOLO
import cv2
import time

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
model = YOLO("yolov8n.engine", task="detect")  # Update with your engine path
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
        results = model(frame, stream=True)
        
        # Process detections
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Confidence and class
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Filter by confidence
                if conf >= 0.5:  # Adjust threshold as needed
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Create label with class name and confidence
                    label = f"{classNames[cls_id]} {conf:.2f}"
                    
                    # Calculate text size for background
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    
                    # Draw filled rectangle for label background
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_height - 5),
                        (x1 + text_width + 5, y1),
                        (0, 255, 0),
                        -1
                    )
                    
                    # Put text
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )
        
        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Display the frame
        cv2.imshow(window_name, frame)
        
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
