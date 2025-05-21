import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os
from military_utils import download_yolov5_if_needed, create_dummy_model_if_needed
from PIL import Image
import time

st.set_page_config(
    page_title="Military Soldier Safety Monitoring",
    page_icon="🛡️",
    layout="wide"
)

# Create the models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Set up the page header
st.title("Military Soldier Safety Monitoring System")
st.markdown("### Powered by YOLOv5 Object Detection")

# Sidebar configuration
st.sidebar.title("Settings")

# Model selection
model_option = st.sidebar.selectbox(
    "Select Detection Model",
    ["Custom Model (best.pt)", "YOLOv5s (Default Fallback)"]
)

# Confidence threshold
conf_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05
)

# IOU threshold for NMS
iou_threshold = st.sidebar.slider(
    "IOU Threshold for NMS",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05
)

# Initialize session state for the model
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_loaded = False
    st.session_state.class_names = []

# Initialize or Load Model
def load_model():
    with st.spinner("Loading detection model... This may take a moment."):
        try:
            # First, make sure YOLOv5 repository is available
            download_yolov5_if_needed()
            
            # If user selects custom model, use that; otherwise use YOLOv5s
            if model_option == "Custom Model (best.pt)":
                # Check if custom model exists, if not create a dummy model for demonstration
                model_path = 'models/best.pt'
                if not os.path.exists(model_path):
                    st.warning("Custom model file not found. Creating a dummy model for demonstration.")
                    create_dummy_model_if_needed()
                
                if os.path.exists(model_path):
                    try:
                        # Try loading as a custom model
                        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                        st.success("Custom model loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading custom model: {e}")
                        # Fallback to using the standard YOLOv5s model
                        st.warning("Falling back to standard YOLOv5s model.")
                        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                else:
                    st.error("Model file could not be created. Falling back to standard YOLOv5s model.")
                    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            else:
                # Load the default YOLOv5s model
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                st.success("YOLOv5s model loaded successfully!")
            
            # Get class names from the model
            class_names = model.names
            
            # Store in session state
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.session_state.class_names = class_names
            
            return model, class_names
        
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, []

# Run detection on image
def detect_objects(img, model):
    # Convert from BGR (OpenCV) to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(img)
    
    # Process predictions with confidence threshold
    predictions = results.pred[0]
    boxes = predictions[:, :4].cpu().numpy()  # x1, y1, x2, y2
    scores = predictions[:, 4].cpu().numpy()
    categories = predictions[:, 5].cpu().numpy().astype(int)
    
    # Return only predictions above the confidence threshold
    mask = scores >= conf_threshold
    return boxes[mask], categories[mask], scores[mask]

# Display detection results
def display_detection_results(img, boxes, categories, scores, class_names):
    # Create a copy of the image to draw on
    img_draw = img.copy()
    
    # Draw boxes and labels
    for box, category, score in zip(boxes, categories, scores):
        x1, y1, x2, y2 = box.astype(int)
        label = f"{class_names[category]}: {score:.2f}"
        
        # Different colors for different categories
        color = (0, 255, 0)  # Default green
        
        # Use different colors for different types of detections
        if "threat" in class_names[category].lower() or "danger" in class_names[category].lower():
            color = (0, 0, 255)  # Red for threats
        elif "soldier" in class_names[category].lower() or "personnel" in class_names[category].lower():
            color = (255, 165, 0)  # Orange for soldiers
        
        # Draw rectangle
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        # Draw filled rectangle for text background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img_draw, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        
        # Draw text
        cv2.putText(img_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img_draw

# Load the model button
if st.sidebar.button("Load Detection Model"):
    model, class_names = load_model()

# Once model is loaded, enable the main UI
if st.session_state.model_loaded:
    model = st.session_state.model
    class_names = st.session_state.class_names
    
    # Display class names
    st.sidebar.markdown("### Detection Classes")
    class_list = ", ".join([f"{i}: {name}" for i, name in enumerate(class_names)])
    st.sidebar.text_area("Available Classes", class_list, height=100)
    
    # Media source selection
    media_source = st.radio("Select Input Source", ["Upload Image", "Upload Video", "Webcam"])
    
    if media_source == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Process the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display the original image
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
            
            # Run detection when requested
            if st.button("Run Detection"):
                # Process the image
                boxes, categories, scores = detect_objects(img, model)
                
                # Display results if any detections
                if len(boxes) > 0:
                    img_result = display_detection_results(img, boxes, categories, scores, class_names)
                    st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), caption="Detection Results", use_column_width=True)
                    
                    # Display detection stats
                    st.markdown("### Detection Results")
                    
                    # Create columns for detected classes and counts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Detected Objects")
                        for cat in np.unique(categories):
                            count = np.sum(categories == cat)
                            st.markdown(f"- **{class_names[cat]}**: {count}")
                    
                    with col2:
                        st.markdown("#### Confidence Scores")
                        for cat in np.unique(categories):
                            cat_scores = scores[categories == cat]
                            avg_score = np.mean(cat_scores)
                            st.markdown(f"- **{class_names[cat]}**: {avg_score:.2f} avg")
                else:
                    st.warning("No objects detected in the image.")
    
    elif media_source == "Upload Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_video is not None:
            # Save the uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            
            # Video properties
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            st.text(f"Video FPS: {fps}, Total Frames: {frame_count}")
            
            # Video processing options
            st.markdown("### Video Processing Options")
            process_every_n_frames = st.slider("Process every N frames", 1, 10, 5)
            
            if st.button("Process Video"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize frame counter
                frame_idx = 0
                processed_frames = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every Nth frame to improve performance
                    if frame_idx % process_every_n_frames == 0:
                        # Update status
                        status_text.text(f"Processing frame {frame_idx}/{frame_count}")
                        progress_bar.progress(frame_idx / frame_count)
                        
                        # Run detection
                        boxes, categories, scores = detect_objects(frame, model)
                        
                        # Draw detection results
                        if len(boxes) > 0:
                            result_frame = display_detection_results(frame, boxes, categories, scores, class_names)
                            processed_frames.append(result_frame)
                        else:
                            processed_frames.append(frame)
                    
                    frame_idx += 1
                
                cap.release()
                
                # Display processed frames as a video
                if processed_frames:
                    status_text.text("Displaying processed video...")
                    
                    # Create a container for the video display
                    video_container = st.empty()
                    
                    # Display frames with proper timing
                    for frame in processed_frames:
                        video_container.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                              caption="Processed Video", 
                                              use_column_width=True)
                        time.sleep(1 / (fps / process_every_n_frames))
                    
                    status_text.text("Video processing complete!")
                else:
                    st.warning("No frames were processed from the video.")
    
    elif media_source == "Webcam":
        st.warning("This is a simulation of webcam functionality. In a deployed environment, this would access your camera.")
        
        # Simulating webcam input with a "Capture" button
        if st.button("Capture from Webcam"):
            # For demonstration, we'll use a blank image (would be webcam frame in real use)
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img.fill(200)  # Light gray background
            
            # Add some text to the simulated webcam image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "Webcam Simulation", (150, 240), font, 1, (0, 0, 0), 2)
            
            st.image(img, caption="Simulated Webcam Feed", use_column_width=True)
            st.info("In a real deployment, this would show live webcam feed and process frames in real-time.")
else:
    # If model not loaded, show instructions
    st.info("Please click 'Load Detection Model' in the sidebar to start.")
    
    # Show some information about the system while waiting
    st.markdown("""
    ## About Military Soldier Safety Monitoring System
    
    This system uses advanced AI-powered object detection to identify potential safety threats and 
    monitor soldier presence in various environments. The system can detect:
    
    - Military personnel/soldiers
    - Potential threats (weapons, hazards)
    - Safety equipment (helmets, vests)
    - Environmental dangers
    
    ### How to use:
    1. Click "Load Detection Model" in the sidebar
    2. Select your input source (image, video, or webcam)
    3. Adjust detection settings as needed
    4. Run detection to analyze the scene
    
    The system will display detection results with bounding boxes and confidence scores.
    """)
