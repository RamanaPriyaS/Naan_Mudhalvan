import cv2
import numpy as np

def simple_threshold_segmentation(image):
    _, thresh = cv2.threshold(image, 0.6, 1.0, cv2.THRESH_BINARY)
    return thresh

def edge_based_segmentation(image):
    edges = cv2.Canny((image * 255).astype(np.uint8), 100, 200)
    return edges
