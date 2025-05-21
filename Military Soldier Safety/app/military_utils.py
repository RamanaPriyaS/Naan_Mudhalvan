import os
import torch
import torchvision
import streamlit as st
from pathlib import Path
import shutil
import sys
import subprocess
import platform
import contextlib
import threading

def emojis(str=""):
    """Returns an emoji-safe version of a string, stripped of emojis on Windows platforms."""
    return str.encode().decode("ascii", "ignore") if platform.system() == "Windows" else str

def threaded(func):
    """Decorator @threaded to run a function in a separate thread, returning the thread instance."""
    def wrapper(*args, **kwargs):
        """Runs the decorated function in a separate daemon thread and returns the thread instance."""
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread
    return wrapper

class TryExcept(contextlib.ContextDecorator):
    """A context manager and decorator for error handling that prints an optional message on exception."""

    def __init__(self, msg=""):
        """Initializes TryExcept with an optional message, used as a decorator or context manager for error handling."""
        self.msg = msg

    def __enter__(self):
        """Enter the runtime context related to this object for error handling with an optional message."""
        pass

    def __exit__(self, exc_type, value, traceback):
        """Context manager exit method that prints an error message if an exception occurred, always returns True."""
        if value:
            print(f"{self.msg}: {value}")
        return True

def download_yolov5_if_needed():
    """
    Ensure YOLOv5 repository is available for loading models.
    This function will check if YOLOv5 is already installed via torch hub, 
    and if not, will download it.
    """
    try:
        # Check if YOLOv5 is already in the torch hub cache
        torch.hub.list('ultralytics/yolov5')
    except Exception as e:
        st.info("Downloading YOLOv5 repository. This may take a moment...")
        
        # Try to force reload the YOLOv5 repository
        try:
            torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
        except Exception as inner_e:
            st.error(f"Failed to download YOLOv5: {inner_e}")
            raise

def create_dummy_model_if_needed():
    """
    Create a dummy PyTorch model file for demonstration purposes
    when the actual trained model is not available.
    """
    model_path = 'models/best.pt'
    
    # Check if model file already exists
    if os.path.exists(model_path):
        return
    
    try:
        # Create a basic YOLOv5 compatible model
        st.info("Creating a dummy model for demonstration...")
        
        # Try one of these approaches, if one fails, try the next
        try:
            # Try to load YOLOv5s as a base first
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
            
            # Modify the class names to simulate military-specific classes
            model.names = [
                'soldier', 'officer', 'vehicle', 'weapon', 'threat',
                'civilian', 'helmet', 'vest', 'hazard', 'safe-zone', 
                'danger-zone', 'explosive', 'shelter', 'radio', 'medical'
            ]
            
            # Save the modified model
            torch.save(model, model_path)  # Save the entire model, not just state_dict
            st.success(f"Created dummy model at {model_path}")
            
        except Exception as e1:
            st.warning(f"First method failed: {e1}. Trying alternative approach...")
            
            # Create a very simple dummy model as a fallback
            import torch.nn as nn
            
            class DummyYOLOModel(nn.Module):
                def __init__(self):
                    super(DummyYOLOModel, self).__init__()
                    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                    self.relu = nn.ReLU()
                    self.names = [
                        'soldier', 'officer', 'vehicle', 'weapon', 'threat',
                        'civilian', 'helmet', 'vest', 'hazard', 'safe-zone', 
                        'danger-zone', 'explosive', 'shelter', 'radio', 'medical'
                    ]
                
                def forward(self, x):
                    return self.relu(self.conv1(x))
                
            dummy_model = DummyYOLOModel()
            torch.save(dummy_model, model_path)
            st.success(f"Created simple dummy model at {model_path}")
        
    except Exception as e:
        st.error(f"Error creating dummy model: {e}")
        st.error("Failed to create any model file. Application may not function correctly.")

def install_dependencies():
    """
    Install required dependencies if they're not already installed.
    This is a fallback function in case the environment doesn't have all dependencies.
    """
    try:
        # Check for required packages and install if missing
        required_packages = ['torch', 'torchvision', 'opencv-python', 'numpy', 'Pillow']
        installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode('utf-8').split('\n')
        installed_packages = [pkg.split('==')[0].lower() for pkg in installed_packages if pkg]
        
        missing_packages = [pkg for pkg in required_packages 
                           if pkg.lower() not in installed_packages and 
                           pkg.lower().split('-')[0] not in installed_packages]
        
        if missing_packages:
            st.warning(f"Installing missing dependencies: {', '.join(missing_packages)}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            st.success("Dependencies installed successfully!")
    
    except Exception as e:
        st.error(f"Error installing dependencies: {e}")
        st.info("You may need to install the required packages manually.")
