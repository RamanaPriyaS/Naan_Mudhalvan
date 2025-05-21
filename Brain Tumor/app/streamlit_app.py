import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.preprocessing import normalize_image, apply_gaussian_blur
from src.segmentation import simple_threshold_segmentation

st.title("Brain Tumor Segmentation")

uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)
    
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    norm_img = normalize_image(image_np)
    blur_img = apply_gaussian_blur(norm_img)
    segmented_img = simple_threshold_segmentation(blur_img)

    st.subheader("Segmented Tumor Region")
    st.image(segmented_img, use_column_width=True, clamp=True)


# Additional visualization logic to highlight the tumor with a green circle
import cv2
import numpy as np

def highlight_tumor(image, mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out small areas
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image, center, radius, (0, 255, 0), 2)
    return image

# Assuming you have a mask and image, here’s an example of using the function:
# processed_image = highlight_tumor(original_image.copy(), predicted_mask)
