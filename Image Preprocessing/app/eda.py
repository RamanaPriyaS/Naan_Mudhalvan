import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

def analyze_image_dimensions(image_paths):
    dimensions = [cv2.imread(p).shape[:2] for p in image_paths]
    heights, widths = zip(*dimensions)
    return heights, widths

def plot_color_histogram(image):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
    plt.title("Color Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()
