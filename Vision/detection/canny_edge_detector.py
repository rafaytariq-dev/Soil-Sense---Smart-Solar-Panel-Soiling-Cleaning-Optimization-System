"""Canny edge detection (Gaussian blur + thresholds 50/150)."""

import cv2
import numpy as np


def apply_canny(img: np.ndarray) -> np.ndarray:
    """Return a binary edge map (uint8, 0 or 255) from a BGR image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, 50, 150)
