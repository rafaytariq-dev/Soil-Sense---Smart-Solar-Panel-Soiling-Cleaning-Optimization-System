import cv2
import numpy as np

from Vision.detection.canny_edge_detector import apply_canny

# Minimum fraction of edge pixels required to accept an image as a panel photo.
_MIN_EDGE_DENSITY = 0.05

# Acceptable width-to-height aspect ratio range for a panel photo.
_MIN_ASPECT = 0.25
_MAX_ASPECT = 4.0


def validate_panel(image_path: str) -> dict:
    """Decide whether an uploaded image contains a solar panel.

    Checks two conditions (Section 2.5):
        1. Aspect ratio — must be in [0.25, 4.0] (landscape or portrait panel).
        2. Edge density — Canny edge pixels / total pixels must be >= 0.05.
           Non-panel images (sky, blank walls, etc.) have very sparse edge maps.

    Args:
        image_path: Path to the input image file.

    Returns:
        {
            "valid": bool,
            "reason": str,      # human-readable explanation
            "confidence": float # [0, 1] — how strongly the image resembles a panel
        }
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"valid": False, "reason": "Cannot read image file.", "confidence": 0.0}

    h, w = img.shape[:2]
    aspect = w / h

    if not (_MIN_ASPECT <= aspect <= _MAX_ASPECT):
        return {
            "valid": False,
            "reason": (
                f"Aspect ratio {aspect:.2f} is outside the accepted range "
                f"[{_MIN_ASPECT}, {_MAX_ASPECT}] — image is unlikely to be a panel photo."
            ),
            "confidence": 0.0,
        }

    edges = apply_canny(img)
    edge_density = float(np.count_nonzero(edges)) / edges.size

    if edge_density < _MIN_EDGE_DENSITY:
        # Partial confidence: how far it is from passing the threshold.
        confidence = round(edge_density / _MIN_EDGE_DENSITY, 4)
        return {
            "valid": False,
            "reason": (
                f"Edge density {edge_density:.4f} is below the minimum "
                f"{_MIN_EDGE_DENSITY} — image lacks panel grid structure."
            ),
            "confidence": confidence,
        }

    # Scale confidence: 0.0 at the threshold, 1.0 at four-times the threshold.
    confidence = round(min((edge_density - _MIN_EDGE_DENSITY) / (3 * _MIN_EDGE_DENSITY) + 0.25, 1.0), 4)
    return {
        "valid": True,
        "reason": "Sufficient edge structure detected — image accepted as a panel photo.",
        "confidence": confidence,
    }
