import gi
import numpy as np
import cv2
import time

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Define callback function first
def on_new_sample(appsink):
    try:
        print("New sample callback triggered")
        sample = appsink.pull_sample()
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

# Define message handler
def on_message(bus, message):
    msg_type = message.type
    if msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()
    elif msg_type == Gst.MessageType.WARNING:
        warn, debug = message.parse_warning()
        print(f"Warning: {warn}, {debug}")
    elif msg_type == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif msg_type == Gst.MessageType.STATE_CHANGED:
        if message.src == pipeline:
            old_state, new_state, pending_state = message.parse_state_changed()
            print(f"Pipeline state changed from {Gst.Element.state_get_name(old_state)} to {Gst.Element.state_get_name(new_state)}")

# Simple pipeline for just displaying camera feed
pipeline_str = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink name=appsink emit-signals=true sync=false drop=true"
)

print(f"Using pipeline: {pipeline_str}")

# Create and start pipeline
pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name('appsink')

if not appsink:
    print("ERROR: Could not retrieve appsink element")
    exit(1)

# Setup message bus
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_message)

# Setup appsink callbacks
appsink.set_property('emit-signals', True)
appsink.set_property('sync', False)
appsink.set_property('drop', True)
appsink.set_property('max-buffers', 1)
appsink.connect('new-sample', on_new_sample)

# Start pipeline
print("Setting pipeline to PLAYING state...")
ret = pipeline.set_state(Gst.State.PLAYING)
print(f"Pipeline state change result: {ret}")

if ret == Gst.StateChangeReturn.FAILURE:
    print("Failed to set pipeline to PLAYING state")
    exit(1)

# Give the camera time to initialize
print("Waiting for camera to initialize...")
time.sleep(2)

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
