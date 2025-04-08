import gi
import numpy as np
import cv2

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Define callback function first
def on_new_sample(appsink):
    sample = appsink.pull_sample()
    buffer = sample.get_buffer()
    caps = sample.get_caps()
    
    structure = caps.get_structure(0)
    width = structure.get_value('width')
    height = structure.get_value('height')
    
    _, map_info = buffer.map(Gst.MapFlags.READ)
    frame = np.ndarray(
        shape=(height, width, 3),
        dtype=np.uint8,
        buffer=map_info.data
    )
    
    # Display frame
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return Gst.FlowReturn.ERROR
    
    buffer.unmap(map_info)
    return Gst.FlowReturn.OK

# Simple pipeline for just displaying camera feed
pipeline_str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink name=appsink emit-signals=true"
)

# Create and start pipeline
pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name('appsink')

# Setup appsink callbacks
appsink.set_property('emit-signals', True)
appsink.connect('new-sample', on_new_sample)

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

# Create main loop
loop = GLib.MainLoop()
try:
    print("Camera feed started. Press Ctrl+C to exit.")
    loop.run()
except KeyboardInterrupt:
    pass

# Clean up
pipeline.set_state(Gst.State.NULL)
cv2.destroyAllWindows()
