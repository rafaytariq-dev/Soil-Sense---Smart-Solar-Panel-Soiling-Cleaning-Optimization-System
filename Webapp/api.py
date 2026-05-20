"""
SoilSense FastAPI backend
Run: uvicorn Webapp.api:app --reload --port 8000
"""

import base64
import os
import sys
import tempfile

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CLEANING_COST, LOCATION, PKR_TARIFF, SYSTEM_KW
from Analytics.anomaly_detector import detect_anomalies
from Analytics.cleaning_scheduler import recommend
from Analytics.pkr_loss_calculator import calculate_loss
from Analytics.seasonal_baseline import compute_baseline
from Analytics.weather_forecast import get_forecast
from Vision.dual_head_predictor import load_model, predict
from Vision.gradcam_explainer import generate_gradcam, overlay_heatmap
from Vision.ood_validator import validate_panel
from Vision.preprocessing.pipeline_runner import run_pipeline

app = FastAPI(title="SoilSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def _img_to_b64(img_bgr) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr)
    return base64.b64encode(buf).decode()


# Cross-check tuning knobs
_CROSS_CHECK_WINDOW_DAYS = 14
_INVERTER_LOSS_THRESHOLD = 0.95   # below this efficiency_ratio counts as real loss


def _compute_recent_efficiency() -> tuple[float | None, int]:
    """Mean efficiency over the last N FoxESS days; zero-generation days are dropped
    so load-shedding / outages don't poison the average."""
    try:
        csv_path = os.path.join(ROOT, "Data", "Fox", "solar_report_6months.csv")
        if not os.path.exists(csv_path):
            return None, 0
        df = compute_baseline(csv_path).sort_values("date").tail(_CROSS_CHECK_WINDOW_DAYS)
        df = df[df["generation"] > 0]
        if df.empty:
            return None, 0
        return float(df["efficiency_ratio"].mean()), int(len(df))
    except Exception:
        return None, 0


def _cross_check(class_label: str, efficiency_ratio: float | None,
                 window_days: int) -> dict:
    """Verdict from {image dust?} × {inverter loss?}."""
    image_dusty = class_label in ("Low Dust", "Heavy Dust")

    if efficiency_ratio is None:
        return {
            "status": "unknown",
            "image_dusty": image_dusty,
            "inverter_efficiency_pct": None,
            "inverter_loss_pct": None,
            "window_days": 0,
            "message": "Inverter data not available — soiling cannot be confirmed at the system level.",
        }

    eff_pct = round(efficiency_ratio * 100, 1)
    loss_pct = round((1 - efficiency_ratio) * 100, 1)
    inverter_loss = efficiency_ratio < _INVERTER_LOSS_THRESHOLD

    if image_dusty and inverter_loss:
        status, message = "confirmed", (
            f"Confirmed soiling. Dust visible on this panel AND inverter shows "
            f"{loss_pct}% underperformance over the last {window_days} days — "
            f"the whole array is likely soiled. Cleaning is justified."
        )
    elif image_dusty and not inverter_loss:
        status, message = "localised", (
            f"Localised dust. This panel shows dust but inverter is at "
            f"{eff_pct}% efficiency over the last {window_days} days. "
            f"Soiling may be limited to a few panels — inspect others before cleaning."
        )
    elif not image_dusty and inverter_loss:
        status, message = "non_dust_loss", (
            f"Loss not caused by dust. This panel appears clean, but inverter shows "
            f"{loss_pct}% underperformance over the last {window_days} days. "
            f"Check for shading, wiring, or inverter faults — cleaning will not help."
        )
    else:
        status, message = "healthy", (
            f"All clear. Panel appears clean and inverter is at {eff_pct}% efficiency "
            f"over the last {window_days} days."
        )

    return {
        "status": status,
        "image_dusty": image_dusty,
        "inverter_efficiency_pct": eff_pct,
        "inverter_loss_pct": loss_pct,
        "window_days": window_days,
        "message": message,
    }


@app.get("/api/config")
def get_config():
    return {
        "system_kw": SYSTEM_KW,
        "tariff": PKR_TARIFF,
        "cleaning_cost": CLEANING_COST,
        "location": LOCATION,
        "model_ready": _get_model() is not None,
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    system_kw: float = Form(SYSTEM_KW),
    tariff: float = Form(PKR_TARIFF),
    cleaning_cost: float = Form(CLEANING_COST),
    is_my_panel: bool = Form(True),
):
    model = _get_model()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        validation = validate_panel(tmp_path)
        if not validation["valid"]:
            return {
                "valid": False,
                "reason": validation["reason"],
                "confidence": validation["confidence"],
            }

        display_img, model_input_tensor, _freq_feats, _visuals = run_pipeline(tmp_path)
    finally:
        os.unlink(tmp_path)

    prediction = predict(model, model_input_tensor, system_kw=system_kw)

    loss_info = calculate_loss(
        watt_loss=prediction["watt_loss"],
        system_kw=system_kw,
        tariff=tariff,
        cleaning_cost=cleaning_cost,
    )

    preprocessed_b64 = _img_to_b64(display_img)

    gradcam_b64 = None
    if model is not None and not prediction.get("mock"):
        try:
            heatmap = generate_gradcam(model["full_model"], model_input_tensor, "top_conv")
            overlay = overlay_heatmap(display_img, heatmap)
            overlay_bgr = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
            gradcam_b64 = _img_to_b64(overlay_bgr)
        except Exception as exc:
            print(f"[api] Grad-CAM failed: {exc}")

    try:
        forecast = get_forecast(days=7)
    except Exception:
        forecast = []

    rec = recommend(forecast, loss_info["daily_loss_pkr"])

    if is_my_panel:
        recent_eff, days_used = _compute_recent_efficiency()
        cross_check = _cross_check(prediction["class_label"], recent_eff, days_used)
    else:
        cross_check = {
            "status": "skipped",
            "image_dusty": prediction["class_label"] in ("Low Dust", "Heavy Dust"),
            "inverter_efficiency_pct": None,
            "inverter_loss_pct": None,
            "window_days": 0,
            "message": "Cross-check skipped — image is not of your own system, so inverter data does not apply.",
        }

    return {
        "valid": True,
        "mock": prediction.get("mock", False),
        "validation_confidence": validation["confidence"],
        "class_label": prediction["class_label"],
        "confidence": prediction["confidence"],
        "expected_watts": prediction["expected_watts"],
        "predicted_watts": prediction["predicted_watts"],
        "watt_loss": prediction["watt_loss"],
        "loss_pct": prediction["loss_pct"],
        "daily_loss_pkr": loss_info["daily_loss_pkr"],
        "monthly_loss_pkr": loss_info["monthly_loss_pkr"],
        "days_to_breakeven": loss_info["days_to_breakeven"],
        "clean_recommended": loss_info["clean_recommended"],
        "preprocessed_image": preprocessed_b64,
        "gradcam_image": gradcam_b64,
        "recommendation": {
            "action": rec["action"],
            "reason": rec["reason"],
            "suppress_loss": rec.get("suppress_loss", False),
            "rain_5day_mm": rec.get("rain_5day_mm", 0),
            "today_cloud_pct": rec.get("today_cloud_pct", 0),
        },
        "cross_check": cross_check,
        "forecast": forecast[:7],
    }


@app.get("/api/weather")
def weather():
    try:
        data = get_forecast(days=7)
        return {"data": data}
    except Exception as exc:
        return {"data": [], "error": str(exc)}


@app.get("/api/dashboard")
def dashboard(system_kw: float = SYSTEM_KW, tariff: float = PKR_TARIFF):
    try:
        csv_path = os.path.join(ROOT, "Data", "Fox", "solar_report_6months.csv")
        df = compute_baseline(csv_path)
        df = detect_anomalies(df)

        df["watt_loss"] = (1.0 - df["efficiency_ratio"]).clip(lower=0) * system_kw * 1000
        df["daily_loss_pkr"] = (df["watt_loss"] / 1000.0) * 6.0 * tariff
        df["cumulative_loss_pkr"] = df["daily_loss_pkr"].cumsum()

        return {
            "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
            "generations": df["generation"].round(2).tolist(),
            "baselines": df["expected_generation"].round(2).tolist(),
            "is_anomaly": df["is_anomaly"].tolist(),
            "daily_losses": df["daily_loss_pkr"].round(0).tolist(),
            "cumulative_losses": df["cumulative_loss_pkr"].round(0).tolist(),
            "total_loss_pkr": float(df["daily_loss_pkr"].sum()),
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "dates": [], "generations": [], "baselines": [],
            "is_anomaly": [], "daily_losses": [], "cumulative_losses": [],
            "total_loss_pkr": 0,
        }


# Serve built React app in production
_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/assets", StaticFiles(directory=os.path.join(_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        return FileResponse(os.path.join(_dist, "index.html"))
