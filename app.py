import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time

# --- Configuration ---
# Dictionary of available YOLO models and their corresponding weight file names.
# These weights are automatically downloaded by the ultralytics library if not present locally.
MODEL_OPTIONS = {
    'YOLOv8 Nano (Fastest)': 'yolov8n.pt',
    'YOLOv8 Small (Default)': 'yolov8s.pt',
    'YOLOv8 Medium (Balanced)': 'yolov8m.pt',
    'YOLOv8 Large (High Accuracy)': 'yolov8l.pt',
    'YOLOv8 Extra Large (Highest Accuracy)': 'yolov8x.pt',
    # You could also add custom models or other versions like 'yolov5s.pt' here.
}

# --- Utility Functions ---

# Use Streamlit's caching to load the model only once
@st.cache_resource
def load_model(model_path):
    """Loads the YOLO model and caches it."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {e}")
        return None

def process_video(uploaded_file, model, conf_threshold):
    """Handles video processing and object detection."""
    # Save the uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    tfile_path = tfile.name
    
    cap = cv2.VideoCapture(tfile_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create placeholders for displaying video and statistics
    st.subheader("Detected Video Stream")
    video_placeholder = st.empty()
    fps_placeholder = st.empty()
    
    # Process video frame by frame
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO inference
        results = model.predict(
            source=frame, 
            conf=conf_threshold, 
            stream=False, 
            verbose=False
        )
        
        # Plot results on the frame (using BGR format from OpenCV)
        annotated_frame = results[0].plot()

        # Convert the frame from BGR (OpenCV default) to RGB for Streamlit display
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
        
        frame_count += 1
        
        # Update FPS display every 10 frames
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            fps_placeholder.metric(label="FPS (Inference Speed)", value=f"{current_fps:.2f}", delta_label="Real-time performance", delta_color="normal")
            
    cap.release()
    tfile.close()
    st.success("Video processing complete!")


# --- Streamlit App Layout ---
def main():
    """Main function to run the Streamlit application."""
    st.title("Object Detection Platform")
    st.subheader("Select Model Version and Upload Video for Inference")
    
    # --- Sidebar for Configuration ---
    st.sidebar.header("Configuration")
    
    # 1. Model Selection Dropdown
    model_name = st.sidebar.selectbox(
        'Select YOLO Model Version:',
        options=list(MODEL_OPTIONS.keys()),
        index=1, # Default to YOLOv8 Small
        help="Different models offer varying trade-offs between speed (FPS) and accuracy (mAP)."
    )
    
    model_path = MODEL_OPTIONS[model_name]
    
    # 2. Confidence Slider
    conf_threshold = st.sidebar.slider(
        'Detection Confidence Threshold',
        min_value=0.01,
        max_value=1.0,
        value=0.25,
        step=0.01,
        help="Lowering the threshold detects more objects, but increases false positives."
    )
    
    # 3. Load the selected model
    st.sidebar.markdown("---")
    st.sidebar.info(f"Loading weights for: **{model_name}**")
    model = load_model(model_path)
    
    if model is None:
        st.stop()
        
    # --- Main Content Area ---
    
    st.header("Upload Video or Image")

    # 4. File Uploader
    uploaded_file = st.file_uploader(
        "Upload a video or image file (.mp4, .mov, .jpg, .png)",
        type=['mp4', 'mov', 'avi', 'jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == 'video':
            st.video(uploaded_file)
            if st.button('Start Object Detection'):
                with st.spinner(f"Running detection on video using {model_name}..."):
                    process_video(uploaded_file, model, conf_threshold)
        
        elif file_type == 'image':
            # Display Image logic (simplified for the scope of this request)
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            
            # Since the model is loaded, we can run inference directly
            if st.button('Detect Objects'):
                with st.spinner(f"Running detection on image using {model_name}..."):
                    
                    # Convert uploaded file to OpenCV image format
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1) # BGR
                    
                    # Run inference
                    results = model.predict(source=image, conf=conf_threshold, verbose=False)
                    annotated_image = results[0].plot()
                    
                    # Convert to RGB for Streamlit
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    
                    st.subheader("Detected Image")
                    st.image(annotated_image_rgb, caption='Detected Objects', use_column_width=True)
                    st.success("Detection complete!")
        else:
            st.warning("Unsupported file type. Please upload a video or image.")

if __name__ == '__main__':
    main()
