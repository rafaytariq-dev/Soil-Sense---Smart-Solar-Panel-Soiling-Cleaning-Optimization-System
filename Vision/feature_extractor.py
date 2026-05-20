"""Classical features (GLCM + HSV + LBP + frequency) for the dashboard."""

import logging

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from Vision.preprocessing.frequency_analyser import extract_frequency_features

logger = logging.getLogger(__name__)

_GLCM_DISTANCES = [1]
_GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
_LBP_POINTS     = 8
_LBP_RADIUS     = 1


def extract_features(img: np.ndarray) -> dict:
    """Return GLCM (contrast/energy/correlation), HSV stats, LBP histogram, and 5 frequency features."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        glcm = graycomatrix(gray, distances=_GLCM_DISTANCES, angles=_GLCM_ANGLES,
                            levels=256, symmetric=True, normed=True)
        contrast    = float(graycoprops(glcm, "contrast").mean())
        energy      = float(graycoprops(glcm, "energy").mean())
        correlation = float(graycoprops(glcm, "correlation").mean())

        h_ch, s_ch, v_ch = cv2.split(hsv)
        hsv_stats = {
            "hsv_h_mean": float(h_ch.mean()), "hsv_h_std": float(h_ch.std()),
            "hsv_s_mean": float(s_ch.mean()), "hsv_s_std": float(s_ch.std()),
            "hsv_v_mean": float(v_ch.mean()), "hsv_v_std": float(v_ch.std()),
        }

        lbp = local_binary_pattern(gray, P=_LBP_POINTS, R=_LBP_RADIUS, method="uniform")
        n_bins = _LBP_POINTS + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        freq_features = extract_frequency_features(img)

        return {
            "glcm_contrast":    round(contrast, 6),
            "glcm_energy":      round(energy, 6),
            "glcm_correlation": round(correlation, 6),
            **{k: round(v, 4) for k, v in hsv_stats.items()},
            "lbp_histogram": [round(float(x), 6) for x in hist],
            **freq_features,
        }
    except Exception as exc:
        logger.error("extract_features failed: %s", exc)
        return {}
