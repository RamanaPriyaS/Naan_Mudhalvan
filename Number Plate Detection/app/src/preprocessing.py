import cv2
import numpy as np

def resize_image(image, width=640, height=480):
    return cv2.resize(image, (width, height))

def enhance_contrast(image):
    return cv2.convertScaleAbs(image, alpha=1.5, beta=0)

def binarize_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh
