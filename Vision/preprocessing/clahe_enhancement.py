"""CLAHE on the L channel only — boosts local contrast without touching hue."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """BGR -> Lab, equalise L channel adaptively, return BGR."""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    except Exception as exc:
        logger.error("apply_clahe failed: %s", exc)
        return img
