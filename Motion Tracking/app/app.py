import streamlit as st
import tempfile
import os
from motion_tracker import process_video
from utils import save_frames_as_video

st.title("Drone Video Motion Tracking using FAST + ORB")

video_file = st.file_uploader("Upload a drone video", type=["mp4", "avi", "mov"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(video_file.read())
        input_path = temp_input.name

    with st.spinner("Processing video..."):
        frames = process_video(input_path)
        output_path = input_path.replace(".mp4", "_tracked.mp4")
        save_frames_as_video(frames, output_path)

    st.success("Processing complete!")
    st.video(output_path)

    with open(output_path, "rb") as file:
        st.download_button("Download Processed Video", file, "motion_tracked.mp4")
