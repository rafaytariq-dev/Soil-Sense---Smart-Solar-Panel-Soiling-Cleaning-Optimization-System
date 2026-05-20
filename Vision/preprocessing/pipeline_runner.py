"""Run the full DIP pipeline on one image: resize -> WB -> denoise -> CLAHE -> normalise."""

import logging

import cv2
import numpy as np

from Vision.preprocessing.image_resizer import resize_to_model_input
from Vision.preprocessing.white_balance import white_balance
from Vision.preprocessing.bilateral_filter import apply_bilateral
from Vision.preprocessing.clahe_enhancement import apply_clahe
from Vision.preprocessing.rgb_normaliser import normalise_for_mobilenet
from Vision.preprocessing.frequency_analyser import extract_frequency_features
from Vision.preprocessing.preprocessing_visualiser import (
    plot_pipeline_stages,
    plot_rgb_histograms,
    plot_3d_surface,
    plot_dust_heatmap,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    image_path: str,
    mode: str = "inference",
    skip_visualiser: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Load and preprocess an image.

    Returns (display_img, model_input_tensor, frequency_features, visualisations).
    display_img is BGR uint8 224x224; model_input_tensor is (1, 224, 224, 3) float32.
    """
    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    try:
        img_resized   = resize_to_model_input(raw)
        img_wb        = white_balance(img_resized)
        img_denoised  = apply_bilateral(img_wb)
        img_equalised = apply_clahe(img_denoised)
        display_img   = img_equalised.copy()
        model_input_tensor = normalise_for_mobilenet(img_equalised)

        augmented_images: list[np.ndarray] = []
        if mode == "training":
            try:
                from Vision.preprocessing.data_augmentor import augment
                augmented_images = augment(img_equalised)
            except ImportError:
                logger.warning("data_augmentor not available — skipping augmentation")

        frequency_features = extract_frequency_features(display_img)

        if skip_visualiser:
            visualisations: dict = {}
        else:
            norm_squeezed = np.squeeze(model_input_tensor, axis=0)
            visualisations = {
                "pipeline_stages": plot_pipeline_stages(
                    raw, img_resized, img_denoised, img_equalised, norm_squeezed
                ),
                "rgb_histograms": plot_rgb_histograms(img_denoised, img_equalised),
                "surface_3d":     plot_3d_surface(display_img),
                "dust_heatmap":   plot_dust_heatmap(display_img),
            }
        if mode == "training":
            visualisations["augmented_count"] = len(augmented_images)

        return display_img, model_input_tensor, frequency_features, visualisations

    except Exception as exc:
        logger.error("run_pipeline failed for '%s': %s", image_path, exc)
        raise
