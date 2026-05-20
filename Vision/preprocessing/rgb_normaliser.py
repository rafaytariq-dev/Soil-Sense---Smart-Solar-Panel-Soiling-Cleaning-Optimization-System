"""BGR uint8 -> float32 RGB tensor with ImageNet mean/std normalisation."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalise_for_mobilenet(image: np.ndarray) -> np.ndarray:
    """BGR -> RGB -> [0,1] -> ImageNet-normalised -> (1, H, W, 3) float32."""
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalised = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.expand_dims(normalised, axis=0)
    except Exception as exc:
        logger.error("normalise_for_mobilenet failed: %s", exc)
        return np.expand_dims(image.astype(np.float32), axis=0)
