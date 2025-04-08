from ultralytics import YOLO
import cv2
import math
import torch

print(torch.cuda.is_available())

# model
model = YOLO("deploy.pt")

# object classes
classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


#
def gstreamer_pipeline(
    capture_width=1280,  # captured width
    capture_height=720,  # captured height
    display_width=1280,  # display width
    display_height=720,  # display height
    framerate=10,  # captured fps
    flip_method=0,  # whether rotate
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
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
    capture_width = 1920
    capture_height = 1080

    display_width = 640
    display_height = 640

    framerate = 5  # fps
    flip_method = 0  # direction

    # pipeline
    gp = gstreamer_pipeline(
        capture_width,
        capture_height,
        display_width,
        display_height,
        framerate,
        flip_method,
    )

    # bind pipeline and stream
    cap = cv2.VideoCapture(gp, cv2.CAP_GSTREAMER)

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)

        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            results = model(img, stream=True)
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    )  # convert to int values

                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    # class name
                    cls = int(box.cls[0])
                    print("Class name -->", classNames[cls])

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(
                        img, classNames[cls], org, font, fontScale, color, thickness
                    )

            cv2.imshow("CSI Camera", img)
            keyCode = cv2.waitKey(30) & 0xFF
            if keyCode == 27:  # ESC to quit
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Failed to open camera")
