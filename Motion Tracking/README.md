# 🎯 Motion Tracking in Drone Videos using FAST and ORB

This project implements a real-time motion tracking system using computer vision techniques. It uses **FAST (Features from Accelerated Segment Test)** for keypoint detection and **ORB (Oriented FAST and Rotated BRIEF)** for descriptor extraction and matching. The motion between video frames is visualized using black lines and dots for clarity.

A **Streamlit web application** is provided to make the system interactive and easy to use.

---

## 🚀 Features

- FAST-based keypoint detection
- ORB-based descriptor extraction and matching
- Motion path visualization using black lines and dots
- Streamlit web app for uploading and processing drone videos
- Downloadable output with motion tracks

---

## 📁 Project Structure

motion-tracking/
├── app.py # Streamlit UI for uploading and displaying videos
├── motion_tracker.py # Motion tracking logic using OpenCV
├── utils.py # Helper to save output frames as a video
├── requirements.txt # List of required Python libraries
└── README.md # Project documentation