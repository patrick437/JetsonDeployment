from ultralytics import YOLO
import cv2
import numpy as np

try:
    # Create a simple test image
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Load the model
    print("Loading model...")
    model = YOLO("yolov8n.engine", task="detect")
    
    # Run inference
    print("Running inference...")
    results = model(test_img)
    
    print(f"Inference successful! Detected {len(results[0].boxes)} objects")
    
except Exception as e:
    print(f"Error: {e}")
