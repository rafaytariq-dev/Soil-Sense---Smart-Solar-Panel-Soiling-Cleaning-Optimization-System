"""Resize to 224x224 (model input size). INTER_AREA for downscale, INTER_LINEAR for upscale."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def resize_to_model_input(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize BGR image to target_size (w, h)."""
    try:
        h, w = image.shape[:2]
        tw, th = target_size
        interp = cv2.INTER_AREA if (h > th or w > tw) else cv2.INTER_LINEAR
        return cv2.resize(image, (tw, th), interpolation=interp)
    except Exception as exc:
        logger.error("resize_to_model_input failed: %s", exc)
        return image
