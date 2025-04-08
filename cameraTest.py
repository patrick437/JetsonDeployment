import gi
import numpy as np
import cv2
import time
import threading

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib, GstApp

# Initialize GStreamer
Gst.init(None)

# Global variables for thread communication
latest_frame = None
frame_ready = False
running = True

# Thread for CV display to avoid Qt threading issues
def display_thread():
    global latest_frame, frame_ready, running
    
    while running:
        if frame_ready:
            # Make a copy to avoid threading issues
            frame = latest_frame.copy()
            
            # Display the frame
            cv2.imshow("Camera Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
                break
                
            frame_ready = False
        else:
            # Short sleep to avoid CPU spinning
            time.sleep(0.01)
    
    cv2.destroyAllWindows()

# Define callback function
def on_new_sample(appsink):
    global latest_frame, frame_ready, running
    
    if not running:
        return Gst.FlowReturn.EOS
    
    try:
        sample = appsink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK
        
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')
        
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            buffer.unmap(map_info)
            return Gst.FlowReturn.ERROR
        
        # Create numpy array from buffer
        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        ).copy()  # Make a copy to avoid memory issues
        
        # Update the global frame
        latest_frame = frame
        frame_ready = True
        
        buffer.unmap(map_info)
        return Gst.FlowReturn.OK
    
    except Exception as e:
        print(f"Error in sample processing: {e}")
        return Gst.FlowReturn.ERROR

# Pipeline
pipeline_str = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=NV12 ! "
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

# Start the display thread
display_thread = threading.Thread(target=display_thread)
display_thread.daemon = True
display_thread.start()

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

# Create main loop
loop = GLib.MainLoop()
try:
    print("Camera feed started. Press 'q' in the video window to exit.")
    loop.run()
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    # Clean up
    running = False
    pipeline.set_state(Gst.State.NULL)
    display_thread.join(timeout=1.0)
    print("Exited cleanly")
