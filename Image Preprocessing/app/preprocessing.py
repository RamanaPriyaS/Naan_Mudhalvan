import cv2
import numpy as np

def resize_image(image, width=400):
    aspect_ratio = width / image.shape[1]
    dim = (width, int(image.shape[0] * aspect_ratio))
    return cv2.resize(image, dim)

def crop_center(image, cropx=200, cropy=200):
    y, x, _ = image.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return image[starty:starty+cropy, startx:startx+cropx]

def convert_color_spaces(image):
    return {
        "RGB": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        "HSV": cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
        "GRAY": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    }

def apply_thresholding(image_gray):
    _, binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    adaptive = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    _, otsu = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {
        "Binary": binary,
        "Adaptive": adaptive,
        "Otsu": otsu
    }
