from ultralytics import YOLO
import cv2
import torch
import time

# Check CUDA availability
print "CUDA available:", torch.cuda.is_available()

# Model - update path to your engine file
model = YOLO("yolov8n.engine", task="detect")

# Object classes (COCO dataset)
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

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=640,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

if __name__ == "__main__":
    # Camera settings
    capture_width = 1280
    capture_height = 720
    display_width = 640
    display_height = 640
    framerate = 30  # Higher framerate for smoother video
    flip_method = 0  # No rotation
    
    # Print settings
    print "Camera settings: %dx%d -> %dx%d @ %dfps" % (
        capture_width, capture_height, display_width, display_height, framerate
    )
    
    # Create pipeline string
    pipeline = gstreamer_pipeline(
        capture_width,
        capture_height,
        display_width,
        display_height,
        framerate,
        flip_method,
    )
    print "Pipeline:", pipeline
    
    # Restart nvargus daemon (can help with camera access)
    import subprocess
    try:
        print "Restarting nvargus-daemon..."
        subprocess.call(["sudo", "systemctl", "restart", "nvargus-daemon"])
        time.sleep(2)  # Give it time to restart
    except Exception, e:
        print "Failed to restart nvargus-daemon:", e
    
    # Create video capture with the pipeline
    print "Opening camera..."
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print "Failed to open camera!"
        exit()
    
    print "Camera opened successfully!"
    
    # Create display window
    window_name = "YOLOv8 Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Process frames until window is closed
    print "Starting detection loop. Press ESC to exit."
    try:
        while cv2.getWindowProperty(window_name, 0) >= 0:
            # Read frame
            ret_val, img = cap.read()
            if not ret_val:
                print "Failed to read frame!"
                break
            
            # Update FPS calculation
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Run YOLOv8 inference
            results = model(img, stream=True)
            
            # Process detections
            for r in results:
                boxes = r.boxes
                
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    # Only draw high-confidence detections
                    if conf >= 0.5:  # Adjust confidence threshold as needed
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Create label with class name and confidence
                        label = "%s %.2f" % (classNames[cls_id], conf)
                        
                        # Calculate text size and position
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        
                        # Draw filled rectangle for text background
                        cv2.rectangle(
                            img, 
                            (x1, y1 - text_height - 5), 
                            (x1 + text_width + 5, y1), 
                            (0, 255, 0), 
                            -1
                        )
                        
                        # Put text
                        cv2.putText(
                            img, 
                            label, 
                            (x1 + 3, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 0, 0), 
                            2
                        )
            
            # Draw FPS on frame
            cv2.putText(
                img,
                "FPS: %.1f" % fps,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            
            # Display the frame
            cv2.imshow(window_name, img)
            
            # Check for ESC key
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:  # ESC to quit
                break
    
    except KeyboardInterrupt:
        print "Interrupted by user"
    except Exception, e:
        print "Error during processing:", e
    finally:
        # Release resources
        print "Cleaning up..."
        cap.release()
        cv2.destroyAllWindows()
        print "Done!"
