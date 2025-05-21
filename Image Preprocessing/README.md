# 🚗 Damaged Car Image Preprocessing Pipeline

This project automates the preprocessing of damaged car images using **OpenCV** and provides a user-friendly web interface via **Streamlit**. It prepares car images for further analysis like damage detection or classification.

---

## 📌 Features

- Resize and crop damaged car images
- Convert images to multiple color spaces (RGB, HSV, GRAY)
- Apply different thresholding techniques (Binary, Adaptive, Otsu)
- Interactive web interface with Streamlit
- Exploratory Data Analysis (EDA) on images
- Option to download preprocessed images

---

## 🧑‍💻 Technologies Used

- Python
- OpenCV
- Streamlit
- NumPy
- PIL (Python Imaging Library)
- Matplotlib (for EDA)

---
## 📁 Project Structure

damaged-car-preprocessing/
│
├── app.py                 # Main Streamlit app
├── preprocessing.py       # Preprocessing functions
├── eda.py                 # EDA functions
├── requirements.txt       # Required Python packages
├── README.md              # Project documentation
└── sample_images/         # Folder for test images
