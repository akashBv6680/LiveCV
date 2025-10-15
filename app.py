import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av  # PyAV for handling video frames
import numpy as np
import cv2 # OpenCV is still used internally by Ultralytics/PyAV

# --- Configuration ---
MODEL_PATH = 'yolov8n.pt'
st.set_page_config(
    page_title="YOLOv8 Live Camera Detector",
    layout="centered"
)

# --- Model Loading with Caching ---
# Caching the model is essential for performance
@st.cache_resource
def load_yolo_model(path):
    """Loads the YOLOv8 model."""
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Failed to load the model from {path}.")
        st.error(f"Error: {e}")
        return None

# Load the model outside the class
MODEL = load_yolo_model(MODEL_PATH)
if MODEL is None:
    st.stop() # Stop the app if model loading failed

# --- Video Processing Class for WebRTC ---

class YOLOv8LiveTransformer(VideoTransformerBase):
    """
    This class is the core of the real-time processing.
    The 'recv' method is called for every frame from the webcam.
    """
    def __init__(self, model):
        self.model = model
        # Default confidence threshold
        self.confidence_threshold = 0.25 
    
    # Method to update confidence from Streamlit UI (must be thread-safe)
    def set_confidence(self, threshold):
        self.confidence_threshold = threshold

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the PyAV frame to a NumPy array in BGR format
        img = frame.to_ndarray(format="bgr24")
        
        # Run YOLOv8 inference on the frame
        # The source is the NumPy array (img)
        # verbose=False suppresses console output for cleaner logs
        results = self.model.predict(
            source=img, 
            conf=self.confidence_threshold, 
            verbose=False,
            # Adjust these for deployment. 
            # Lowering the resolution can increase FPS.
            # imgsz=320 
        )

        # Get the annotated frame (NumPy array with boxes/labels drawn)
        annotated_frame = results[0].plot()

        # Convert the annotated NumPy array back to a PyAV VideoFrame
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- Streamlit UI ---
st.title("Live Camera Object Detection (YOLOv8)")
st.caption("Powered by Streamlit, YOLOv8, and streamlit-webrtc.")

# Sidebar for controls
with st.sidebar:
    st.header("Detection Settings")
    
    # Create a unique key in session state to hold the confidence value
    if 'live_conf' not in st.session_state:
        st.session_state.live_conf = 0.25

    # Slider for confidence threshold
    confidence = st.slider(
        "Confidence Threshold", 
        min_value=0.01, 
        max_value=1.0, 
        value=st.session_state.live_conf, 
        step=0.01,
        key="confidence_slider"
    )
    # Update the session state
    st.session_state.live_conf = confidence

# Initialize the WebRTC stream
ctx = webrtc_streamer(
    key="yolo-live-detection",
    mode=WebRtcMode.SENDRECV,
    # Pass the VideoTransformer class to the factory
    video_transformer_factory=lambda: YOLOv8LiveTransformer(MODEL), 
    # Constraints to request video but not audio
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    # Configuration for STUN server to work across different networks
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    async_processing=True # Allows frames to be processed in a separate thread
)

# Logic to pass the confidence threshold to the transformer
if ctx.video_transformer:
    # Safely update the confidence on the transformer instance
    ctx.video_transformer.set_confidence(st.session_state.live_conf)

    # Display real-time frame rate in the app (optional)
    if st.checkbox("Show Performance Info"):
        st.info("Performance stats can be displayed here, but require more complex thread-safe state management.")

st.markdown("---")
st.write("Click **START** to begin streaming from your webcam.")
