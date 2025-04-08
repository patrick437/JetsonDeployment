import gi
import numpy as np
import cv2
import time

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')  # Add GstApp explicitly
from gi.repository import Gst, GLib, GstApp  # Import GstApp

# Initialize GStreamer
Gst.init(None)

# Define callback function using emit instead of direct method call
def on_new_sample(appsink):
    try:
        print("New sample callback triggered")
        sample = appsink.emit("pull-sample")  # Use emit instead of pull_sample
        if not sample:
            print("No sample received")
            return Gst.FlowReturn.OK
        
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')
        
        print(f"Frame dimensions: {width}x{height}")
        
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            print("Failed to map buffer")
            return Gst.FlowReturn.ERROR
        
        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        )
        
        # Display frame
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            loop.quit()
            return Gst.FlowReturn.ERROR
        
        buffer.unmap(map_info)
        return Gst.FlowReturn.OK
    
    except Exception as e:
        print(f"Error in sample processing: {e}")
        return Gst.FlowReturn.ERROR

# Simple pipeline - using a configuration closer to what worked on command line
pipeline_str = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=NV12 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink name=appsink emit-signals=true"
)

print(f"Using pipeline: {pipeline_str}")

# Create and start pipeline
pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name('appsink')

# Setup appsink callbacks
appsink.set_property('emit-signals', True)
appsink.connect('new-sample', on_new_sample)

# Start pipeline
print("Setting pipeline to PLAYING state...")
ret = pipeline.set_state(Gst.State.PLAYING)
print(f"Pipeline state change result: {ret}")

# Create main loop
loop = GLib.MainLoop()
try:
    print("Camera feed started. Press Ctrl+C to exit.")
    loop.run()
except KeyboardInterrupt:
    print("Interrupted by user")
    pass

# Clean up
print("Stopping pipeline...")
pipeline.set_state(Gst.State.NULL)
cv2.destroyAllWindows()
print("Exited cleanly")
