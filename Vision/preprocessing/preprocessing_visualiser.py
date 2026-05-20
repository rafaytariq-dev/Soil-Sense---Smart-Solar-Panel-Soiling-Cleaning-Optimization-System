"""DIP-report figures (pipeline stages, RGB hists, 3D surfaces, dust heatmap) as numpy arrays."""

import io
import logging

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _fig_to_array(fig: plt.Figure) -> np.ndarray:
    """Render a Figure to a BGR uint8 array and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    plt.close(fig)
    buf.close()
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _to_display_bgr(img: np.ndarray) -> np.ndarray:
    """Convert any image variant to uint8 BGR (denormalising float32 if needed)."""
    if img is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if img.dtype != np.uint8:
        denorm = np.clip(img * _IMAGENET_STD + _IMAGENET_MEAN, 0.0, 1.0)
        return (denorm * 255).astype(np.uint8)
    return img


def plot_pipeline_stages(
    original: np.ndarray,
    resized: np.ndarray,
    denoised: np.ndarray,
    equalised: np.ndarray,
    normalised: np.ndarray,
) -> np.ndarray:
    """Side-by-side figure showing all 5 preprocessing stages."""
    try:
        titles = ["Original", "Resized\n224x224", "Denoised\n(Bilateral)", "Equalised\n(CLAHE)", "Normalised\n(ImageNet)"]
        images = [original, resized, denoised, equalised, normalised]

        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle("Preprocessing Pipeline Stages", fontsize=13, fontweight="bold")

        for ax, title, img in zip(axes, titles, images):
            rgb = cv2.cvtColor(_to_display_bgr(img), cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        return _fig_to_array(fig)
    except Exception as exc:
        logger.error("plot_pipeline_stages failed: %s", exc)
        return np.zeros((100, 800, 3), dtype=np.uint8)


def plot_rgb_histograms(image_before: np.ndarray, image_after: np.ndarray) -> np.ndarray:
    """R/G/B intensity histograms before vs after CLAHE."""
    try:
        fig, axes = plt.subplots(2, 3, figsize=(14, 6))
        fig.suptitle("RGB Channel Histograms — Before vs After CLAHE", fontsize=12, fontweight="bold")

        channel_names = ("R", "G", "B")
        colors        = ("red", "green", "royalblue")
        bgr_indices   = (2, 1, 0)

        for col, (ch, color, idx) in enumerate(zip(channel_names, colors, bgr_indices)):
            for row, (img, label) in enumerate([(image_before, "Before"), (image_after, "After")]):
                ax = axes[row, col]
                hist = cv2.calcHist([img], [idx], None, [256], [0, 256]).ravel()
                ax.fill_between(range(256), hist, color=color, alpha=0.5)
                ax.plot(hist, color=color, linewidth=0.8)
                ax.set_title(f"{ch} — {label}", fontsize=9)
                ax.set_xlabel("Intensity", fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                ax.set_xlim(0, 255)

        plt.tight_layout()
        return _fig_to_array(fig)
    except Exception as exc:
        logger.error("plot_rgb_histograms failed: %s", exc)
        return np.zeros((100, 600, 3), dtype=np.uint8)


def plot_3d_surface(image: np.ndarray) -> np.ndarray:
    """3D surface plots of R/G/B intensity (green=dusty, red=clean)."""
    try:
        small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(16, 5))
        fig.suptitle("3-D Channel Intensity Surfaces\n(green = dusty, red = clean)", fontsize=12, fontweight="bold")

        x = np.arange(64); y = np.arange(64)
        X, Y = np.meshgrid(x, y)

        for i, ch_name in enumerate(("R", "G", "B")):
            ax = fig.add_subplot(1, 3, i + 1, projection="3d")
            Z = rgb_small[:, :, i].astype(np.float32)
            ax.plot_surface(X, Y, Z, cmap="RdYlGn", vmin=0, vmax=255)
            ax.set_title(f"{ch_name} channel", fontsize=10)
            ax.set_zlim(0, 255)
            ax.set_xlabel("X", fontsize=8); ax.set_ylabel("Y", fontsize=8); ax.set_zlabel("Intensity", fontsize=8)
            ax.tick_params(labelsize=6)

        plt.tight_layout()
        return _fig_to_array(fig)
    except Exception as exc:
        logger.error("plot_3d_surface failed: %s", exc)
        return np.zeros((100, 600, 3), dtype=np.uint8)


def plot_dust_heatmap(image: np.ndarray) -> np.ndarray:
    """Green-channel heatmap with contour at 50% intensity to mark dusty regions."""
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        green = rgb[:, :, 1].astype(np.float32) / 255.0

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.set_title("Dust Heatmap — Green Channel\n(low intensity = dusty region)", fontsize=11)

        im = ax.imshow(green, cmap="YlGn", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, label="Relative Intensity [0–1]")

        contour = ax.contour(green, levels=[0.5], colors="red", linewidths=1.5)
        ax.clabel(contour, fmt="0.5", fontsize=8)
        ax.set_xlabel("X (pixels)", fontsize=9); ax.set_ylabel("Y (pixels)", fontsize=9)

        plt.tight_layout()
        return _fig_to_array(fig)
    except Exception as exc:
        logger.error("plot_dust_heatmap failed: %s", exc)
        return np.zeros((100, 400, 3), dtype=np.uint8)
