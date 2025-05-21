# Military Soldier Safety Monitoring System

A system that uses YOLOv5 object detection to monitor military soldier safety by analyzing images and videos for potential threats and safety hazards.

## Features

- Real-time object detection using YOLOv5
- Military-specific detection classes (soldiers, threats, safety equipment, etc.)
- Support for image, video, and webcam input
- Adjustable detection confidence thresholds
- Visual display of detection results with bounding boxes
- Summary statistics for detected objects

## Installation

1. Unzip the project files
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required dependencies:
   ```
   pip install -r requirements_readme.txt
   ```
4. Optional: Place your custom trained YOLOv5 model file (`best.pt`) in the `models/` directory
   (If no model is available, the system will create a dummy model or use the standard YOLOv5s model)

## Usage

Run the Streamlit application:
```
streamlit run app.py
```

## Instructions

1. Click "Load Detection Model" in the sidebar
2. Select your input source (image, video, or webcam)
3. Adjust detection settings as needed
4. Run detection to analyze the scene

The system will display detection results with bounding boxes and confidence scores.

