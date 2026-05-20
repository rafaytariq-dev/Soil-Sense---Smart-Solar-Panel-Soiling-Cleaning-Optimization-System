"""Frequency-domain dust analysis: Butterworth filters and 5 spectral features."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_fft_spectrum(image: np.ndarray) -> dict:
    """Return {'spectrum', 'spectrum_img'} — log-magnitude FFT after Hanning window."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        fft_shift = np.fft.fftshift(np.fft.fft2(gray * win))
        spectrum = np.log1p(np.abs(fft_shift)).astype(np.float32)
        norm = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX)
        return {"spectrum": spectrum, "spectrum_img": norm.astype(np.uint8)}
    except Exception as exc:
        logger.error("compute_fft_spectrum failed: %s", exc)
        return {"spectrum": None, "spectrum_img": None}


def _butterworth_mask(h: int, w: int, cutoff_ratio: float, order: int,
                      high_pass: bool = False) -> np.ndarray:
    """Centred Butterworth frequency mask of shape (H, W)."""
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    d = np.sqrt((yy - cy) ** 2.0 + (xx - cx) ** 2.0)
    cutoff = cutoff_ratio * min(h, w) / 2.0
    d_safe = np.where(d == 0, 1e-9, d)
    if high_pass:
        return (1.0 / (1.0 + (cutoff / d_safe) ** (2 * order))).astype(np.float32)
    return (1.0 / (1.0 + (d_safe / cutoff) ** (2 * order))).astype(np.float32)


def _fft_filter_reconstruct(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a frequency mask and reconstruct a uint8 spatial image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    fft_shift = np.fft.fftshift(np.fft.fft2(gray * win))
    recon = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_shift * mask))).astype(np.float32)
    return cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def apply_butterworth_lpf(image: np.ndarray, cutoff_ratio: float = 0.10, order: int = 2) -> np.ndarray:
    """Butterworth low-pass — keeps haze, large-scale brightness."""
    try:
        h, w = image.shape[:2]
        return _fft_filter_reconstruct(image, _butterworth_mask(h, w, cutoff_ratio, order, False))
    except Exception as exc:
        logger.error("apply_butterworth_lpf failed: %s", exc)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_butterworth_hpf(image: np.ndarray, cutoff_ratio: float = 0.30, order: int = 2) -> np.ndarray:
    """Butterworth high-pass — keeps edges and fine texture."""
    try:
        h, w = image.shape[:2]
        return _fft_filter_reconstruct(image, _butterworth_mask(h, w, cutoff_ratio, order, True))
    except Exception as exc:
        logger.error("apply_butterworth_hpf failed: %s", exc)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_butterworth_bpf(image: np.ndarray, low_ratio: float = 0.10,
                          high_ratio: float = 0.30, order: int = 2) -> np.ndarray:
    """Butterworth band-pass — mid-frequency texture (coarse dust grains)."""
    try:
        h, w = image.shape[:2]
        lpf = _butterworth_mask(h, w, high_ratio, order, False)
        hpf = _butterworth_mask(h, w, low_ratio,  order, True)
        return _fft_filter_reconstruct(image, lpf * hpf)
    except Exception as exc:
        logger.error("apply_butterworth_bpf failed: %s", exc)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_notch_filter(image: np.ndarray, notch_radius: int = 8, num_peaks: int = 6) -> np.ndarray:
    """Suppress periodic spectral peaks from the panel cell grid."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        cy, cx = h // 2, w // 2

        win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        fft_shift = np.fft.fftshift(np.fft.fft2(gray * win))
        magnitude = np.abs(fft_shift)

        # Don't suppress the DC neighbourhood — that's signal, not periodic artefact
        dc_radius = int(min(h, w) * 0.10)
        searchable = magnitude.copy()
        yy, xx = np.ogrid[:h, :w]
        searchable[(yy - cy) ** 2 + (xx - cx) ** 2 <= dc_radius ** 2] = 0.0

        notch_mask = np.ones((h, w), dtype=np.float32)
        for _ in range(num_peaks):
            py, px = np.unravel_index(int(np.argmax(searchable)), searchable.shape)
            for ry, rx in [(py, px), (2 * cy - py, 2 * cx - px)]:
                ry = int(np.clip(ry, 0, h - 1))
                rx = int(np.clip(rx, 0, w - 1))
                circle = (yy - ry) ** 2 + (xx - rx) ** 2 <= notch_radius ** 2
                notch_mask[circle] = 0.0
                searchable[circle] = 0.0

        recon = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_shift * notch_mask))).astype(np.float32)
        return cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    except Exception as exc:
        logger.error("apply_notch_filter failed: %s", exc)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def extract_frequency_features(image: np.ndarray) -> dict:
    """5 spectral features: LPF/HPF/BPF energy ratios, HPF/LPF ratio, spectral entropy."""
    _zero = {
        "lpf_energy_ratio": 0.0, "hpf_energy_ratio": 0.0, "bpf_energy_ratio": 0.0,
        "hpf_lpf_ratio": 0.0, "spectral_entropy": 0.0,
    }
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        fft_shift = np.fft.fftshift(np.fft.fft2(gray * win))
        power = (np.abs(fft_shift) ** 2).astype(np.float64)
        total = float(power.sum()) + 1e-12

        lpf_mask = _butterworth_mask(h, w, 0.10, 2, False).astype(np.float64)
        hpf_mask = _butterworth_mask(h, w, 0.30, 2, True).astype(np.float64)
        bpf_mask = (
            _butterworth_mask(h, w, 0.30, 2, False).astype(np.float64)
            * _butterworth_mask(h, w, 0.10, 2, True).astype(np.float64)
        )

        lpf_e = float((power * lpf_mask).sum())
        hpf_e = float((power * hpf_mask).sum())
        bpf_e = float((power * bpf_mask).sum())

        prob = power / total
        nz = prob.ravel()
        nz = nz[nz > 0]
        entropy = float(-np.sum(nz * np.log2(nz + 1e-12)))
        entropy /= float(np.log2(h * w + 1e-12))

        return {
            "lpf_energy_ratio": round(lpf_e / total, 6),
            "hpf_energy_ratio": round(hpf_e / total, 6),
            "bpf_energy_ratio": round(bpf_e / total, 6),
            "hpf_lpf_ratio":    round(hpf_e / (lpf_e + 1e-12), 6),
            "spectral_entropy": round(float(np.clip(entropy, 0.0, 1.0)), 6),
        }
    except Exception as exc:
        logger.error("extract_frequency_features failed: %s", exc)
        return _zero
