import streamlit as st
import cv2
import numpy as np
from src import ocr
from src import preprocessing


st.title("Vehicle Number Plate Detection")

uploaded_file = st.file_uploader("Upload a vehicle image")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    text = ocr.extract_text(img)
    st.write("Detected Plate Text:", text)
