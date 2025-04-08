import gi
import numpy as np
import cv2
from ultralytics import YOLO

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib

class YoloDetector:
    def __init__(self, model_path, input_size=320):
        # Initialize GStreamer
        Gst.init(None)
        
        # Load the TensorRT engine
        self.model = YOLO(model_path)
        self.input_size = input_size
        
        # Create GStreamer pipeline
        self.create_pipeline()
        
        # Start the pipeline
        self.start_pipeline()
    
    def create_pipeline(self):
        # Pipeline for Raspberry Pi Camera V2 via CSI
        pipeline_str = (
            "nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        
        # Create pipeline
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Get appsink element
        self.appsink = self.pipeline.get_by_name('appsink')
        self.appsink.set_property('emit-signals', True)
        self.appsink.connect('new-sample', self.on_new_sample)
        
        # Connect bus messages to handle errors
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::error', self.on_error)
        bus.connect('message::warning', self.on_warning)
        bus.connect('message::eos', self.on_eos)
    
    def start_pipeline(self):
        # Start playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Failed to set pipeline to playing state")
        else:
            print("Pipeline is playing")
        
        # Create GLib MainLoop
        self.loop = GLib.MainLoop()
    
    def on_new_sample(self, appsink):
        # Pull sample from appsink
        sample = appsink.pull_sample()
        if not sample:
            return Gst.FlowReturn.ERROR
        
        # Get buffer and convert to numpy array
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        # Get dimensions
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')
        
        # Map buffer to get data
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            buffer.unmap(map_info)
            return Gst.FlowReturn.ERROR
        
        # Create numpy array from buffer
        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        )
        
        # Process with YOLO (make a copy to avoid memory issues)
        frame = frame.copy()  
        
        # Preprocess (resize to model's input size)
        orig_shape = frame.shape
        
        # Run YOLO detection
        results = self.model(frame)
        
        # Draw results on frame
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                # Get coordinates and class info
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.quit()
        
        # Cleanup
        buffer.unmap(map_info)
        return Gst.FlowReturn.OK
    
    def on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"GStreamer Error: {err}, {debug}")
        self.quit()
    
    def on_warning(self, bus, msg):
        warn, debug = msg.parse_warning()
        print(f"GStreamer Warning: {warn}, {debug}")
    
    def on_eos(self, bus, msg):
        print("End of stream reached")
        self.quit()
    
    def run(self):
        try:
            print("Running detection loop, press Ctrl+C to quit")
            self.loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.quit()
    
    def quit(self):
        # Stop the pipeline
        self.pipeline.set_state(Gst.State.NULL)
        
        # Quit the loop
        if self.loop.is_running():
            self.loop.quit()
        
        # Cleanup windows
        cv2.destroyAllWindows()

# Load the TensorRT engine and start the detector
detector = YoloDetector("yolov8n.engine", input_size=320)
detector.run()
