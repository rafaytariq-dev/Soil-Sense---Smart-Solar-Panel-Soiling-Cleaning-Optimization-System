"""Edge-preserving denoise. Runs after white balance, before CLAHE."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_bilateral(img: np.ndarray) -> np.ndarray:
    """Smooth flat regions, preserve edges. Returns BGR uint8 unchanged on error."""
    try:
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    except Exception as exc:
        logger.error("apply_bilateral failed: %s", exc)
        return img
