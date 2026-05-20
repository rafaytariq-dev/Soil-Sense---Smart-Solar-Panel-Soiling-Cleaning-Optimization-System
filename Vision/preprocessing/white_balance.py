"""Gray-World white balance via Lab a/b channel re-centring."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def white_balance(img: np.ndarray) -> np.ndarray:
    """Shift Lab a and b channel means to 128 so the average colour is neutral grey."""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        a -= (np.mean(a) - 128.0)
        b -= (np.mean(b) - 128.0)
        lab = np.clip(cv2.merge([l, a, b]), 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except Exception as exc:
        logger.error("white_balance failed: %s", exc)
        return img
