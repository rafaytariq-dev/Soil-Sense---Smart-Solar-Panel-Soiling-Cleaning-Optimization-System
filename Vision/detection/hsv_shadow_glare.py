"""HSV-based shadow and glare detection for solar panel images."""

import cv2
import numpy as np


def shadow_check(img: np.ndarray) -> dict:
    """Flag dark grey pixels (V<40 and S<=50) as shadow. Returns {shadowed, ratio}."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = (v < 40) & (s <= 50)
    ratio = float(mask.sum()) / mask.size
    return {"shadowed": bool(ratio > 0.05), "ratio": round(ratio, 4)}


def glare_check(img: np.ndarray) -> dict:
    """Flag overexposed pixels (V>240) as glare. Returns {glare, ratio}."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    mask = v > 240
    ratio = float(mask.sum()) / mask.size
    return {"glare": bool(ratio > 0.05), "ratio": round(ratio, 4)}
