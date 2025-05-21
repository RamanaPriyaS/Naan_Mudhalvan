import cv2
import numpy as np

def load_and_resize(image_path, size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(image, size)

def normalize_image(image):
    return image / 255.0

def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)
