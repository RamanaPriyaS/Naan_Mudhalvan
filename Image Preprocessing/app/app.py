import streamlit as st
import cv2
import numpy as np
from preprocessing import resize_image, crop_center, convert_color_spaces, apply_thresholding
from PIL import Image

st.title("🚗 Damaged Car Image Preprocessing")
st.markdown("Upload a damaged car image and apply preprocessing techniques.")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("Original Image")
    st.image(image, channels="BGR")

    # Preprocessing
    resized = resize_image(image)
    cropped = crop_center(resized)
    color_spaces = convert_color_spaces(cropped)
    thresholds = apply_thresholding(color_spaces["GRAY"])

    st.subheader("Resized & Cropped")
    st.image(cropped, channels="BGR")

    st.subheader("Color Spaces")
    st.image(color_spaces["RGB"], caption="RGB")
    st.image(color_spaces["HSV"], caption="HSV")
    st.image(color_spaces["GRAY"], caption="Grayscale", use_column_width=True, clamp=True)

    st.subheader("Thresholding")
    st.image(thresholds["Binary"], caption="Binary Threshold", use_column_width=True, clamp=True)
    st.image(thresholds["Adaptive"], caption="Adaptive Threshold", use_column_width=True, clamp=True)
    st.image(thresholds["Otsu"], caption="Otsu's Threshold", use_column_width=True, clamp=True)

    # Download processed images
    st.download_button("Download Cropped Image", Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).tobytes(), file_name="cropped.jpg")
