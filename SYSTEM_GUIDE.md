# SoilSense — System Guide

A walkthrough of the SoilSense pipeline from "what is this thing" to "how does every piece work." Read top to bottom for the full story; jump to any section by name when you just need that part.

---

## 1. What SoilSense does

You upload a photo of your solar panel. The system:

1. Decides whether the photo is actually a solar panel.
2. Cleans the image (white-balance, denoise, contrast-boost).
3. Classifies the panel as **Clean / Low Dust / Heavy Dust** using a deep-learning model.
4. Highlights which pixels drove the decision (Grad-CAM).
5. Estimates the daily power loss in Watts and the financial loss in PKR.
6. Cross-checks the image verdict against your real inverter generation data.
7. Recommends an action: **CLEAN NOW / WAIT / MONITOR**, factoring in the weather forecast.

Two semester courses are integrated: **Digital Image Processing** (the preprocessing + Grad-CAM) and **Data Science** (the classifier, the analytics, the cross-check).

---

## 2. How to run it

Two terminals from the repo root.

**Backend:**
```powershell
uvicorn Webapp.api:app --reload --port 8000
```

**Frontend (dev mode):**
```powershell
cd Webapp/frontend
npm run dev
```

Open http://localhost:5173 in a browser. Vite proxies `/api/*` requests to the FastAPI backend on port 8000.

**One-port production build:**
```powershell
cd Webapp/frontend; npm run build; cd ../..
uvicorn Webapp.api:app --port 8000
```
Then http://localhost:8000 serves both UI and API.

**Liveness check:** `curl http://localhost:8000/api/config` returns `{"model_ready": true, ...}` once the model has loaded.

---

## 3. The big picture

```
                      Browser  (React + Recharts)
                          ↑
                          | HTTP
                          ↓
              ┌─────────────────────────────┐
              │   FastAPI backend (api.py)  │
              └─────────────────────────────┘
                /          |          \
       ┌────────┐  ┌──────────────┐  ┌────────────┐
       │ Vision │  │  Analytics   │  │ Open-Meteo │
       │ stack  │  │  (CSV math)  │  │  weather   │
       └────────┘  └──────────────┘  └────────────┘
            |             |
   image → DIP →       FoxESS
   model → result    solar_report
                      _6months.csv
```

Three pillars, all wired through one FastAPI process:

| Pillar | What it owns |
|---|---|
| **Vision** | DIP preprocessing, OOD check, EfficientNetB0 + MLP classifier, Grad-CAM |
| **Analytics** | PKR loss math, seasonal baseline, anomaly detection, cleaning scheduler |
| **External APIs** | Open-Meteo (weather forecast), FoxESS (already-fetched CSVs in `Data/Fox/`) |

---

## 4. End-to-end workflow — what happens when you upload a photo

```
1.  User drops a JPG into the React UploadSection
        ↓
2.  POST /api/analyze   (multipart: file + system_kw + tariff + is_my_panel)
        ↓
3.  ood_validator.validate_panel(path)
        ├─ aspect ratio in [0.25, 4.0]?
        └─ Canny edge density ≥ 0.05?
        ↓  (reject early if either fails)
4.  pipeline_runner.run_pipeline(path)
        ├─ Step 1  image_resizer       → 224×224
        ├─ Step 2  white_balance       → remove colour cast
        ├─ Step 3  bilateral_filter    → edge-preserving denoise
        ├─ Step 4  clahe_enhancement   → local contrast boost
        ├─ Step 5  rgb_normaliser      → float32 ImageNet-normalised tensor
        ├─ Step 7  frequency_analyser  → 5 frequency-domain features
        └─ Step 8  preprocessing_visualiser → debug plots (skip in production)
        ↓
5.  dual_head_predictor.predict(model, tensor, system_kw)
        ├─ Lambda layer: undo ImageNet norm → [0, 255] RGB
        ├─ EfficientNetB0 backbone (frozen)
        ├─ GlobalAveragePooling → 1280-d feature vector
        ├─ Normalization layer (StandardScaler baked in)
        └─ MLP head → softmax → class + confidence
        ↓
6.  gradcam_explainer.generate_gradcam(...)
        Heatmap of which pixels mattered, overlaid on the preprocessed image
        ↓
7.  pkr_loss_calculator.calculate_loss(watt_loss, system_kw, tariff)
        daily_pkr, monthly_pkr, days_to_breakeven
        ↓
8.  weather_forecast.get_forecast(lat, lon, 7)
        Next 7 days from Open-Meteo (free, keyless)
        ↓
9.  cleaning_scheduler.recommend(forecast, daily_pkr)
        CLEAN NOW | WAIT | MONITOR
        ↓
10. _compute_recent_efficiency()  +  _cross_check()    (if is_my_panel)
        Compares image diagnosis with 14-day inverter efficiency
        → CONFIRMED / LOCALISED / NON-DUST LOSS / ALL CLEAR
        ↓
11. JSON response → React renders preprocessed image, Grad-CAM, metrics,
    cross-check verdict, cleaning recommendation
```

---

## 5. Components in detail

### 5.1 Data sources

| Source | Where | How used |
|---|---|---|
| **Deep Solar Eye** (45,754 images) | Kaggle dataset, downloaded locally to `Data/raw/` then preprocessed to `Data/processed/{Clean,Low_Dust,Heavy_Dust}/` via `batch_preprocess.py` | Train the classifier (offline, on Kaggle GPU) |
| **FoxESS CSVs** (245 days) | `Data/Fox/solar_report_6months.csv` + `solar_history_6months.csv` | Power your dashboard, anomaly detection, and the cross-check |
| **Open-Meteo** | `https://api.open-meteo.com/v1/forecast` (no key) | 7-day weather forecast for cleaning recommendation |

The two data sources never mix during training. Image data trains the model, FoxESS data drives analytics at inference time.

### 5.2 DIP preprocessing pipeline — `Vision/preprocessing/`

Five image-transforming steps run in fixed order. Order matters — denoise before contrast-boost, contrast-boost before normalisation.

| Step | File | What it does | How it does it |
|---|---|---|---|
| 1 | `image_resizer.py` | Standardise to 224×224 | OpenCV `cv2.resize` with bilinear interpolation |
| 2 | `white_balance.py` | Remove yellow/blue colour cast from outdoor lighting and dust | Convert BGR→Lab, normalise the **a** (green-red) and **b** (blue-yellow) channels around 128, convert back |
| 3 | `bilateral_filter.py` | Smooth noise inside flat dusty regions without blurring cell grid lines | `cv2.bilateralFilter(d=9, sigmaColor=75, sigmaSpace=75)` — weights neighbours by *both* spatial distance and intensity similarity, so sharp edges survive |
| 4 | `clahe_enhancement.py` | Boost local contrast so dust texture stands out | Convert BGR→Lab, apply `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` on the **L** channel only (preserves colour), convert back |
| 5 | `rgb_normaliser.py` | Convert uint8 BGR → float32 RGB tensor in ImageNet's stats range | BGR→RGB swap, divide by 255, subtract `[0.485, 0.456, 0.406]`, divide by `[0.229, 0.224, 0.225]`, add batch dim → `(1, 224, 224, 3)` |

After step 5: also runs `frequency_analyser.py` (5 FFT-domain features for the DIP report) and optionally `preprocessing_visualiser.py` (matplotlib figures). Neither changes the tensor sent to the model.

### 5.3 DIP detection — `Vision/detection/`

These run inside `ood_validator` and are **not** part of the model pipeline.

| File | Purpose |
|---|---|
| `canny_edge_detector.py` | Gaussian blur (5×5) → `cv2.Canny(50, 150)`. Returns a binary edge map used to compute edge density |
| `hsv_shadow_glare.py` | Two checks: `shadow_check` flags pixels with low HSV value in the grey hue band; `glare_check` flags overexposed pixels (V > 240). Used in extended OOD checks (currently optional) |

### 5.4 OOD validator — `Vision/ood_validator.py`

First line of defence. Decides whether the uploaded image is even worth running the model on.

```
valid ← (0.25 ≤ width/height ≤ 4.0)  AND  (edge_density ≥ 0.05)
```

The `confidence` it returns scales from 0 at the threshold to 1.0 at four-times-threshold — a soft signal of "how panel-like" the image looks.

Limitation: this catches obvious non-panel images (sky, blank wall, portrait) but not anything with rectangular edges (brick wall, parking lot). See Section 7.

### 5.5 The classifier — what was trained, where, how

**Backbone**: `EfficientNetB0(include_top=False, weights="imagenet")`. Frozen — no gradients flow through it.

**Why transfer learning, not training from scratch?** Solar-panel dust photos are visually close to ImageNet's general object/texture distribution. The pretrained backbone already knows edges, textures, gradients, surfaces. We don't have 14 million images of panels — we have ~45k. Training from scratch would overfit; freezing the backbone and learning only a small head is the standard recipe.

**Feature extraction (offline, on Kaggle)** — [notebooks/solar_preprocessing_pipeline.ipynb](notebooks/solar_preprocessing_pipeline.ipynb):
- Loop every image: `EfficientNetB0(preprocess_input(img))` → GAP → 1280-d feature vector
- Save to `solar_features.pkl` (~1.5 GB) along with a fitted `StandardScaler`

**Head training (offline, on Kaggle)** — [notebooks/solar_model_training.ipynb](notebooks/solar_model_training.ipynb):
- MLP: `Dense(256, relu) → Dropout(0.4) → Dense(64, relu) → Dropout(0.3) → Dense(3, softmax)`
- Loss: sparse categorical cross-entropy
- Regularisation: L2 (`1e-4`), dropout, EarlyStopping on val_loss
- Result: 97.83% reported test accuracy (with caveat — see Section 7)

**Logistic Regression baseline** is also trained as a sanity check (89.73% test acc).

**Deployed artifacts** in `Models/`:
- `mlp_head.keras` — the trained head, 4.2 MB
- `scaler.pkl` — the fitted StandardScaler, 31 KB

### 5.6 Inference predictor — `Vision/dual_head_predictor.py`

`load_model()` builds **one unified Keras model** with everything wired in:

```
Input(1, 224, 224, 3)                      ← pipeline-normalised
 ↓
Lambda: undo_pipeline_norm  → [0, 255] RGB ← EfficientNet expects this
 ↓
EfficientNetB0 (ImageNet, frozen)
 ↓
GlobalAveragePooling2D                     → (1, 1280)
 ↓
Normalization (mean & var from scaler)     ← bakes StandardScaler in
 ↓
mlp_head  (Dense → Dropout → Dense → Softmax)
 ↓
Output(1, 3)
```

The pkl was trained on `[Clean, Heavy_Dust, Low_Dust]` indices `[0, 1, 2]`. The UI shows them in severity order `[Clean, Low Dust, Heavy Dust]`, so we permute `[0, 2, 1]` via `_PKL_TO_DEPLOY`.

`predict()` returns class label, confidence, and watt figures. Watt loss is computed as `system_kw × 1000 × midpoint_loss_pct`, where the midpoints come from `_LOSS_RANGES`:

| Class | Loss range | Midpoint |
|---|---|---|
| Clean | 0 – 6 % | 3 % |
| Low Dust | 10 – 30 % | 20 % |
| Heavy Dust | 35 – 65 % | 50 % |

These ranges come from typical PV soiling studies. Because they're percentages, changing the SettingsBar slider (e.g., 7.5 kW vs 10 kW) scales the displayed watts correctly.

**Mock mode**: if `mlp_head.keras` or `scaler.pkl` is missing, `load_model()` returns `None` and `predict()` returns randomised but plausible numbers so the webapp keeps working during development.

### 5.7 Explainability — `Vision/gradcam_explainer.py`

Grad-CAM produces a heatmap showing which spatial regions of the image drove the predicted class.

How it works:
1. Build a secondary Keras model that exposes both `top_conv`'s feature maps AND the final softmax.
2. Compute the gradient of the top-class score with respect to those feature maps.
3. Spatially average those gradients per channel → channel importance weights.
4. Multiply each feature map by its weight and sum → 2-D heatmap.
5. ReLU + min-max normalise to [0, 1].
6. Resize to image size, colour-map with JET, blend over the preprocessed image.

In SoilSense, the target layer is `top_conv` (EfficientNetB0's last convolutional layer before pooling). At inference the unified model is passed in so all layers are visible by name.

### 5.8 Analytics — `Analytics/`

| File | Function | What it does |
|---|---|---|
| `seasonal_baseline.py` | `compute_baseline(csv_path)` | Loads `solar_report_6months.csv`, groups by month, computes monthly **median** generation, returns DataFrame annotated with `expected_generation` and `efficiency_ratio = actual / expected` |
| `anomaly_detector.py` | `detect_anomalies(df)` | Flags rows where `efficiency_ratio < 0.85` as candidate soiling days |
| `pkr_loss_calculator.py` | `calculate_loss(watt_loss, system_kw, tariff, cleaning_cost)` | `daily_pkr = (watt_loss/1000) × 6 × tariff` (6 = average peak-sun hours in Rawalpindi). Also returns monthly, breakeven, and a `clean_recommended` boolean |
| `cleaning_scheduler.py` | `recommend(forecast, daily_loss)` | Decision tree: rain in next 5 days > 5 mm → WAIT; daily loss > PKR 200 → CLEAN NOW; else MONITOR. Also flags `suppress_loss=True` when today's cloud cover > 60% |
| `weather_forecast.py` | `get_forecast(lat, lon, days)` | One HTTPS GET to Open-Meteo, returns list of `{date, rain_mm, cloud_cover_pct, description}`. WMO weather codes are mapped to human-readable text |

### 5.9 Cross-check — the cleanest novel contribution

[Webapp/api.py](Webapp/api.py) — `_compute_recent_efficiency()` + `_cross_check()`.

Takes two **independent** signals:

1. **Image classifier**: "is there dust on this one panel?" — answer ∈ {Clean, Low Dust, Heavy Dust}
2. **Inverter data**: "is the *whole system* actually losing power right now?" — mean efficiency_ratio over the last 14 days (dropping zero-generation days so load shedding doesn't corrupt the average)

Cross-tabulates them:

| | Inverter shows real loss | Inverter looks fine |
|---|---|---|
| **Image says dust** | CONFIRMED SOILING — clean it | LOCALISED DUST — inspect other panels first |
| **Image says clean** | NON-DUST LOSS — check shading / wiring | ALL CLEAR |

A fifth state (**CROSS-CHECK SKIPPED**) is used when the user explicitly unchecks "This is my own solar system" in the UI — sensible when uploading a stranger's panel or a demo image. A sixth (**CROSS-CHECK UNAVAILABLE**) covers the case where the FoxESS CSV is missing.

### 5.10 Webapp

**Backend** — [Webapp/api.py](Webapp/api.py) (FastAPI):

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/config` | GET | System constants + `model_ready` flag |
| `/api/analyze` | POST (multipart) | Full analysis of an uploaded image |
| `/api/weather` | GET | 7-day Open-Meteo forecast |
| `/api/dashboard` | GET | Historical generation, baseline, anomalies, PKR loss series for the charts |

**Frontend** — [Webapp/frontend/](Webapp/frontend/) (React + Vite + Recharts):

| Component | What it shows |
|---|---|
| `SettingsBar` | System capacity (kW, 0.5 step), tariff (PKR/kWh), cleaning cost (PKR), location |
| `UploadSection` | Drag-drop upload, "is my panel" toggle, progress UI, result card with badge + confidence bar + preprocessed image + Grad-CAM + metric tiles |
| `RecommendationCard` | Cross-check verdict badge + grid (this panel vs inverter) + cleaning action block + weather tags |
| `GenerationChart` | Daily actual vs baseline kWh, anomaly days highlighted |
| `LossChart` | Daily PKR loss bars + cumulative loss line |
| `WeatherChart` | 7-day rain (bars) + cloud cover (line) |

---

## 6. Algorithms and techniques (mastery section)

### Why CLAHE not standard histogram equalisation?
A global histogram equalisation would brighten the whole image uniformly — fine for a posterised scene, bad for a panel where some cells are in shadow. CLAHE divides the image into 8×8 tiles, equalises each separately, then bilinearly interpolates at tile boundaries. `clipLimit=2.0` caps any one bin from dominating, which suppresses noise amplification. Run on the **L** channel of Lab so saturation and hue are untouched.

### Why bilateral filter not Gaussian blur?
Gaussian blur is a low-pass filter — it averages all neighbours, blurring everything including the cell-grid edges that Canny needs later. The bilateral filter weights each neighbour by **both** spatial proximity (Gaussian on distance) **and** intensity similarity (Gaussian on value difference). A neighbour across a sharp edge has very different intensity → near-zero weight → the edge survives.

### Why standardise features after GAP, not before?
The 1280 outputs of GAP have very different scales — some channels respond strongly to ImageNet's mean-textured images, others barely. A small MLP head learns much faster on standardised features (each feature ~ N(0, 1)) than on raw ones. StandardScaler is fit on the **train split only** to avoid leakage, then applied to val and test.

### Why bake the scaler into the model graph at deployment?
If the scaler lives outside the Keras model, deployment needs *both* the .keras file *and* the .pkl file *and* code that runs them in the right order. By converting `scaler.mean_` and `scaler.var_` into a `tf.keras.layers.Normalization` layer at load time, the entire pipeline becomes one Keras graph — Grad-CAM works, single `model.predict()` call works, easier to ship.

### Why use class-percentage midpoints for watt loss instead of training a regression head?
Regression on a continuous L value (Deep Solar Eye's power-loss target) was tried during exploration and abandoned: the labels are noisy at low-irradiance times, and the 3-class binning matches what users actually care about ("should I clean it?"). Watt loss is then computed as `system_kw × 1000 × midpoint_of_class`, which scales correctly with the user's real system size and is interpretable in viva.

### Why drop zero-generation days from the 14-day efficiency window?
Soiling never drops a working panel to **absolute zero** generation — only outages (load shedding, inverter offline, grid-tied disconnect) do that. Including those days makes the 14-day mean efficiency artificially low and triggers false "non-dust loss" verdicts. Filtering `generation > 0` removes the bias without needing a separate "was the grid up?" signal.

### Why monthly **medians** for baselines, not means?
Means are dragged by outliers: a single cloudy week pulls the monthly mean down even if the rest of the month was normal. The median is the middle value — it ignores the tails and represents a typical day for that month. Robust to both unusually bad days and unusually good days.

### Why Open-Meteo over OpenWeatherMap?
- Keyless (no `.env` setup, no rate-limit panic at viva)
- Returns ready-aggregated **daily** values (precipitation_sum, cloud_cover_mean, weather_code)
- Free for non-commercial use
- Historical archive endpoint exists (`/v1/archive`) if you ever want to enrich the analytics

---

## 7. Known limitations and how the system handles them

| Limitation | Mitigation |
|---|---|
| Single photo → whole-system inference assumes uniform soiling | Cross-check against FoxESS inverter data; "this is my panel" toggle |
| 97.83% test accuracy may be inflated by oversampling-before-split data leakage | Acknowledged; honest re-evaluation is queued |
| Softmax classifier has no "I don't know" output | OOD validator filters obvious non-panel images; confidence is reported but not promoted as calibrated probability |
| OOD validator only checks aspect ratio + edge density | Can be fooled by brick walls or rectangular structures; future work: Mahalanobis distance to training embeddings |
| `_WATT_MIDPOINTS` are heuristic class midpoints, not regression | Honest in the report — a regression head was traded for simplicity |
| Hard-coded Rawalpindi coordinates | OK for the project demo; would come from a user profile in deployment |
| Load shedding days corrupt cross-check window | Filtered out (`generation > 0` requirement) |
| Cloudy spells affect cross-check window | Partially mitigated by 14-day average; full fix would weight by Open-Meteo historical cloud cover |

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **Soiling** | Dust/dirt accumulating on panel surface, reducing power output |
| **OOD** | Out-of-distribution — input that does not match the training distribution |
| **GAP** | GlobalAveragePooling — averages each feature map to one scalar |
| **CLAHE** | Contrast-Limited Adaptive Histogram Equalisation |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping — heatmap of where the model "looked" |
| **Efficiency ratio** | Actual generation ÷ monthly median generation; 1.0 = normal, < 0.85 = anomaly |
| **WMO code** | World Meteorological Organization weather code (0 = clear, 95 = thunderstorm, etc.) |
| **Tariff** | Electricity unit price in PKR per kWh |
| **Breakeven** | `cleaning_cost ÷ daily_pkr_loss` — days a cleaning pays for itself |

---

## 9. Files at a glance

```
.
├── config.py                      System constants (kW, tariff, paths)
├── batch_preprocess.py            One-off: raw images → Data/processed/{class}/
├── requirements.txt
├── PROJECT_PLAN.md                Original semester plan
├── SYSTEM_GUIDE.md                This file
├── Data/
│   ├── Fox/                       FoxESS inverter CSVs (refreshed via Analytics/fox/)
│   └── processed/                 (gitignored) Preprocessed training images
├── Models/                        Trained artifacts (mlp_head.keras + scaler.pkl)
├── notebooks/
│   ├── solar_preprocessing_pipeline.ipynb   Feature extraction on Kaggle
│   └── solar_model_training.ipynb           MLP head training + comparison
├── Analytics/                     PKR math, baseline, anomaly, scheduler, weather
├── Vision/
│   ├── preprocessing/             5-step DIP pipeline + freq features + viz
│   ├── detection/                 Canny edge detector + HSV shadow/glare
│   ├── ood_validator.py           Aspect-ratio + edge-density gate
│   ├── dual_head_predictor.py     Unified Keras model + predict()
│   ├── gradcam_explainer.py       Grad-CAM heatmap + overlay
│   └── feature_extractor.py       Classical features (GLCM, LBP, HSV) for dashboard
└── Webapp/
    ├── api.py                     FastAPI: /api/config, /analyze, /weather, /dashboard
    └── frontend/                  React + Vite + Recharts
```

---

*Author's note for examiners: SoilSense's distinguishing contribution is the **cross-check** between an image classifier and inverter measurements. Either signal alone is incomplete — an image cannot prove there is power loss, and an inverter cannot diagnose *why* there is loss. Combining them gives an actionable verdict that respects what each data source can and cannot claim.*
