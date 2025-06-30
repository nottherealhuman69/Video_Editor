import cv2
import numpy as np
from functools import partial

def grayscale(img):
    """Convert frame to grayscale"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def invert_colors(img):
    """Invert colors of the frame"""
    return cv2.bitwise_not(img)

def sepia(img):
    """Apply sepia tone effect"""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(img, kernel)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

def pencil_sketch(img):
    """Create pencil sketch effect"""
    gray, sketch = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return sketch  # Already 3-channel, no need to convert

def cartoon_effect(img):
    """Apply cartoon-like effect"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def edge_detection(img):
    """Apply Canny edge detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def sobel_x(img):
    """Apply Sobel X edge detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    return cv2.cvtColor(abs_sobelx, cv2.COLOR_GRAY2BGR)

def sobel_y(img):
    """Apply Sobel Y edge detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    abs_sobely = cv2.convertScaleAbs(sobely)
    return cv2.cvtColor(abs_sobely, cv2.COLOR_GRAY2BGR)

def laplacian(img):
    """Apply Laplacian edge detection"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    abs_lap = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(abs_lap, cv2.COLOR_GRAY2BGR)

def gaussian_blur(img):
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(img, (7, 7), 0)

def median_blur(img):
    """Apply median blur"""
    return cv2.medianBlur(img, 5)

def bilateral_filter(img):
    """Apply bilateral filter for noise reduction while preserving edges"""
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

def motion_blur(img):
    """Apply motion blur effect"""
    kernel_size = 15
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(img, -1, kernel)

def emboss_filter(img):
    """Apply emboss effect"""
    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    emboss = cv2.filter2D(gray, -1, kernel) + 128
    # Ensure values are in valid range
    emboss = np.clip(emboss, 0, 255).astype(np.uint8)
    return cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)

def sharpen(img):
    """Sharpen the image"""
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    return cv2.filter2D(img, -1, kernel)

def hsv_filter(img):
    """Convert to HSV and back (for HSV color space visualization)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def negative_hsv(img):
    """Apply negative effect in HSV color space"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (180 - hsv[:, :, 0]) % 180  # Hue wrap-around
    hsv[:, :, 1] = 255 - hsv[:, :, 1]  # Invert saturation
    hsv[:, :, 2] = 255 - hsv[:, :, 2]  # Invert value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def color_tint(img, color=(255, 0, 0), alpha=0.3):
    """Apply color tint overlay (default: blue tint)"""
    overlay = np.full(img.shape, color, dtype=np.uint8)
    return cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

def thresholding(img):
    """Apply binary thresholding"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def adaptive_threshold(img):
    """Apply adaptive thresholding"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adapt = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)

def dilation(img):
    """Apply morphological dilation"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)