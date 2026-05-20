"""Wrap the trained classifier into a single predict() call.

Falls back to mock predictions when the model/scaler files are missing,
so the rest of the system can be developed without a trained model.
"""

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import SYSTEM_KW, FINAL_MODEL_PATH, SCALER_PATH  # noqa: E402

CLASSES = ["Clean", "Low Dust", "Heavy Dust"]

# pkl trained ordering is [Clean, Heavy_Dust, Low_Dust]; we display by severity
# [Clean, Low Dust, Heavy Dust], so pkl idx 2 (Low) -> 1, pkl idx 1 (Heavy) -> 2.
_PKL_TO_DEPLOY = [0, 2, 1]

# pipeline_runner uses ImageNet mean/std; EfficientNet wants raw [0, 255] RGB.
_PIPELINE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_PIPELINE_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Per-class soiling loss ranges (fraction of peak output).
# Source: typical PV soiling studies — Clean ~0-6%, light dust ~10-30%, heavy ~35-65%.
_LOSS_RANGES: dict[str, tuple[float, float]] = {
    "Clean":      (0.00, 0.06),
    "Low Dust":   (0.10, 0.30),
    "Heavy Dust": (0.35, 0.65),
}
_LOSS_MIDPOINTS = {c: (lo + hi) / 2 for c, (lo, hi) in _LOSS_RANGES.items()}


def load_model(head_path: str = None, scaler_path: str = None):
    """Build the image-to-softmax pipeline as one flat Keras model.

    Layers: undo-pipeline-norm -> EfficientNetB0 -> GAP -> baked scaler -> MLP head.
    Conv layers stay accessible by name (e.g. "top_conv") so Grad-CAM works.

    Returns a dict bundle, or None if files are missing / load fails.
    """
    head_path   = head_path   or FINAL_MODEL_PATH
    scaler_path = scaler_path or SCALER_PATH

    if not os.path.exists(head_path) or not os.path.exists(scaler_path):
        return None

    try:
        import pickle
        import tensorflow as tf
        from tensorflow.keras import Input, Model, layers
        from tensorflow.keras.applications import EfficientNetB0

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        head = tf.keras.models.load_model(head_path)

        inp = Input(shape=(224, 224, 3), dtype="float32", name="pipeline_input")
        eff = layers.Lambda(
            lambda t: (t * _PIPELINE_STD + _PIPELINE_MEAN) * 255.0,
            name="undo_pipeline_norm",
        )(inp)
        base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=eff)
        base.trainable = False
        x = layers.GlobalAveragePooling2D(name="gap")(base.output)
        x = layers.Normalization(
            mean=scaler.mean_, variance=scaler.var_, name="bake_scaler",
        )(x)
        out = head(x)
        full_model = Model(inp, out, name="soilsense_unified")

        return {"full_model": full_model, "scaler": scaler, "head": head}

    except Exception as exc:
        print(f"[dual_head_predictor] Could not load bundle: {exc}")
        return None


def predict(model, image_tensor: np.ndarray, system_kw: float = SYSTEM_KW) -> dict:
    """Classify the image and convert the result to watt figures for `system_kw`.

    Args:
        model:        Bundle from load_model(), or None for mock mode.
        image_tensor: (1, 224, 224, 3) float32 from pipeline_runner.
        system_kw:    Installed system size in kW (drives expected/predicted watts).

    Returns a dict with class_label, confidence, expected_watts, predicted_watts,
    watt_loss, loss_pct.
    """
    if model is None:
        return _mock_predict(system_kw)

    try:
        pkl_probs = np.array(
            model["full_model"].predict(image_tensor, verbose=0)
        ).flatten()
        class_probs = pkl_probs[_PKL_TO_DEPLOY]

        top_idx = int(np.argmax(class_probs))
        class_label = CLASSES[top_idx]
        return _build_result(
            class_label=class_label,
            confidence=float(class_probs[top_idx]),
            loss_pct=_LOSS_MIDPOINTS[class_label],
            system_kw=system_kw,
        )

    except Exception as exc:
        print(f"[dual_head_predictor] Inference failed ({exc}), using mock mode.")
        return _mock_predict(system_kw)


def _mock_predict(system_kw: float) -> dict:
    """Random realistic prediction for development without a trained model."""
    rng = np.random.default_rng()
    class_label = str(rng.choice(CLASSES, p=[0.30, 0.50, 0.20]))

    lo, hi = _LOSS_RANGES[class_label]
    loss_pct = float(rng.uniform(lo, hi))
    confidence = float(rng.uniform(0.55, 0.85)) if class_label == "Dusty" else float(rng.uniform(0.75, 0.98))

    result = _build_result(class_label, confidence, loss_pct, system_kw)
    result["mock"] = True
    return result


def _build_result(class_label: str, confidence: float,
                  loss_pct: float, system_kw: float) -> dict:
    """Compose the response dict from class + loss fraction + system size."""
    expected_watts  = max(system_kw, 0.0) * 1000.0
    loss_pct        = max(0.0, min(loss_pct, 1.0))
    watt_loss       = expected_watts * loss_pct
    predicted_watts = expected_watts - watt_loss

    return {
        "class_label":     class_label,
        "confidence":      round(confidence, 4),
        "predicted_watts": round(predicted_watts, 1),
        "expected_watts":  round(expected_watts, 1),
        "watt_loss":       round(watt_loss, 1),
        "loss_pct":        round(loss_pct * 100, 2),
    }
