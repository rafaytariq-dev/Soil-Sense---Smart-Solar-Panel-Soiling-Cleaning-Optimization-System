. Project Overview
SoilSense merges two semester courses — Digital Image Processing (DIP) and Data Science (DS) — into one applied system. A user uploads a solar panel photo. The DIP pipeline enhances it. A CNN classifies soiling and predicts watt drop. The analytics engine converts watt drop to daily PKR loss, checks the weather forecast, and recommends whether to clean now or wait for rain.

1.1 Two data sources — they never mix
Source	What it is	Used for
Kaggle dataset (45k images)	Panel photos labelled clean / dusty / very dusty + watt value	Train CNN classifier and watt regression head
FoxESS CSV (180 days)	Abu Bakar's inverter data: daily kWh, 5-min power intervals	Seasonal baseline, anomaly detection, PKR loss calculation

No panel photos from the last 180 days are needed. The image classifier and the generation CSV are completely separate. Section 7 explains this in full.


2. Complete Folder Structure
Every file is named after its technique or responsibility. No two files share the same name anywhere in the project. The __init__.py files are a Python requirement — they cannot be renamed (see Section 8). They are always empty.

2.1 Root
File	Owner	Syllabus	What to implement
config.py	Abu Bakar	—	Central constants: PKR_TARIFF=65, CLEANING_COST=800, SYSTEM_KW=10, Rawalpindi coords, all folder paths, model paths, API base URLs
requirements.txt	Abu Bakar	—	All pip packages: tensorflow, opencv-python, streamlit, plotly, pandas, numpy, requests, python-dotenv, pillow
.gitignore	Abu Bakar	—	Ignore: .env, *.h5, data/raw/, data/fox/*.csv, __pycache__/, venv/, output/
README.md	Abu Bakar	—	Project title, team names, setup steps, how to run: streamlit run webapp/streamlit_app.py

2.2 data/
File	Owner	Syllabus	What to implement
data/raw/	Done	—	45k Kaggle images. In .gitignore — never commit. Download manually to this folder
data/processed/	Done	—	DIP pipeline output images. Auto-created by pipeline_runner.py
data/fox/solar_report_6months.csv	Done	—	PASTE YOUR FILE HERE. 182 rows. Columns: date, generation, feedin, gridConsumption
data/fox/solar_history_6months.csv	Done	—	PASTE YOUR FILE HERE. 51k rows. 5-min interval power data

2.3 vision/preprocessing/ — Rafay (4 technique files)
Rafay owns all 4 files. Each file is one standalone function plus a test block. All 4 are called in sequence by pipeline_runner.py.

File	Owner	Syllabus	What to implement
__init__.py	Rafay	—	Empty file. Required by Python. Do not edit.
white_balance.py	Rafay	Week 3	Function white_balance(img). Convert to Lab color space. Normalize a and b channels. Returns corrected BGR image
clahe_enhancement.py	Rafay	Week 6	Function apply_clahe(img). Adaptive histogram eq. on L channel only. clipLimit=2.0, tileGridSize=(8,8). Returns enhanced BGR image
bilateral_filter.py	Rafay	Week 6	Function apply_bilateral(img). Edge-preserving smooth: d=9, sigmaColor=75, sigmaSpace=75. Returns smoothed BGR image
pipeline_runner.py	Rafay	Week 3+6	Function run_pipeline(image_path). Calls white_balance -> apply_clahe -> apply_bilateral in order. Resizes to 224x224. Normalizes to ImageNet mean/std. Returns (display_img, model_input_tensor)

2.4 vision/detection/ — Aimen (4 technique files)
Aimen owns all 4 detection files. This is Aimen's DIP contribution. These run after preprocessing to locate the panel and flag image quality issues.

File	Owner	Syllabus	What to implement
__init__.py	Aimen	—	Empty file. Required by Python. Do not edit.
canny_edge_detector.py	Aimen	Week 7	Function apply_canny(img). Apply Gaussian blur (5x5) then cv2.Canny(50, 150). Returns edge map. Also used by ood_validator.py for edge density check
hough_panel_locator.py	Aimen	Week 12	Function locate_panel(edge_map). Apply HoughLinesP. Find dominant rectangle from line intersections. Return bounding box dict {x, y, w, h} or None if not found
morphological_closing.py	Aimen	Week 16	Function close_edges(edge_map). Apply cv2.morphologyEx with MORPH_CLOSE and a 5x5 kernel. Fills small gaps. Returns clean binary mask
hsv_shadow_glare.py	Aimen	Week 6	Two functions in one file. shadow_check(img): HSV value < 40 in gray hue range -> returns {shadowed: bool, ratio: float}. glare_check(img): value > 240 -> returns {glare: bool, ratio: float}

2.5 vision/ — Abu Bakar (model logic)
File	Owner	Syllabus	What to implement
__init__.py	Abu Bakar	—	Empty file. Required by Python. Do not edit.
ood_validator.py	Abu Bakar	—	Function validate_panel(image_path). Uses canny_edge_detector to compute edge_density. If edge_density < 0.05 or aspect ratio is wrong -> reject. Returns {valid: bool, reason: str, confidence: float}
gradcam_explainer.py	Abu Bakar	—	Function generate_gradcam(model, image_tensor, target_layer). Returns heatmap array. Function overlay_heatmap(original, heatmap). Returns PIL Image with colored overlay
feature_extractor.py	Abu Bakar	—	Function extract_features(img). Returns dict of classical features: GLCM contrast/energy/correlation, HSV mean and std per channel, LBP histogram. For dashboard display only — not used for classification
dual_head_predictor.py	Abu Bakar	—	Function predict(model, image_tensor). Returns {class_label, confidence, predicted_watts, expected_watts, watt_loss, loss_pct}. Feeds into Aimen's pkr_loss_calculator.py

2.6 analytics/fox/ — paste existing files
File	Owner	Syllabus	What to implement
fox_config.py	Done	—	Already complete. API auth, MD5 signature, rate limiting
fox_data_extract.py	Done	—	Already complete. Pulls 180-day daily report + 5-min history from FoxESS API
fox_device_diagnostics.py	Done	—	Renamed from fox_extract.py to avoid confusion. Device discovery and API connection test
fox_visualize.py	Done	—	Already complete. 8 chart types saved to output/

2.7 analytics/ — Aimen (DS pipeline)
File	Owner	Syllabus	What to implement
__init__.py	Aimen	—	Empty file. Required by Python. Do not edit.
seasonal_baseline.py	Aimen	—	Function compute_baseline(csv_path). Load solar_report_6months.csv. Group by month. Compute median daily generation per month. Add efficiency_ratio = actual / expected column. Return annotated DataFrame
anomaly_detector.py	Aimen	—	Function detect_anomalies(df). Flag rows where efficiency_ratio < 0.85. These are candidate soiling days. Used to validate that classifier results align with real generation drops
pkr_loss_calculator.py	Aimen	—	Function calculate_loss(watt_loss, system_kw=10, tariff=65, cleaning_cost=800). Formula: daily_loss = watt_loss_kw x 6 x tariff. Returns {daily_loss_pkr, monthly_loss_pkr, days_to_breakeven, clean_recommended}
weather_forecast.py	Aimen	—	Function get_forecast(lat=33.6007, lon=73.0679, days=7). OpenWeatherMap free API. Returns list of dicts: [{date, rain_mm, cloud_cover_pct, description}]
cleaning_scheduler.py	Aimen	—	Function recommend(forecast, daily_loss). Logic: rain_mm > 5 in next 5 days -> WAIT. daily_loss > 200 -> CLEAN NOW. Else -> MONITOR. Also sets suppress_loss = True if today cloud_cover > 60%

2.8 notebooks/
File	Owner	Syllabus	What to implement
01_baseline_cnn_training.ipynb	Abu Bakar	—	Train 3-layer CNN from scratch on Kaggle GPU. Record accuracy and loss curves
02_mobilenet_transfer.ipynb	Abu Bakar	—	Fine-tune MobileNetV2 (freeze base, train head, then partial unfreeze). Compare with baseline
03_gradcam_visualization.ipynb	Abu Bakar	—	Load trained model. Generate Grad-CAM heatmaps on test set. Save to docs/figures/
04_power_regression_head.ipynb	Abu Bakar	—	Add linear regression head to MobileNetV2. Train dual loss: softmax classification + watt regression
05_fox_data_analysis.ipynb	Aimen	—	Exploratory analysis of solar_report_6months.csv. Generation trends, seasonality, outliers, missing days
06_seasonal_baseline_fit.ipynb	Aimen	—	Fit and validate monthly median baseline. Plot efficiency_ratio over 6 months. Identify soiling event candidates

2.9 webapp/ — Aimen
File	Owner	Syllabus	What to implement
__init__.py	Aimen	—	Empty file. Required by Python. Do not edit.
streamlit_app.py	Aimen	—	Entry point. Run with: streamlit run webapp/streamlit_app.py. Sidebar: system_kw slider, tariff input, location. Two tabs: Analysis and Energy Dashboard
upload_panel.py	Aimen	—	Image uploader widget. Calls ood_validator -> pipeline_runner -> dual_head_predictor in sequence. Returns prediction dict
results_display.py	Aimen	—	Show classification label, confidence score, Grad-CAM heatmap side by side with original image
generation_chart.py	Aimen	—	Plotly line chart of daily generation. Color segments by efficiency_ratio: green = normal, red = anomaly days
loss_chart.py	Aimen	—	Plotly bar chart of estimated daily PKR loss from dust over the 6-month period
weather_chart.py	Aimen	—	Plotly 7-day forecast chart. Rain bars (mm) + cloud cover line. Mark recommended clean date
recommendation_card.py	Aimen	—	Display final action card: CLEAN NOW (red), WAIT (blue), or MONITOR (green) with reasoning text and PKR figures

2.10 models/ and docs/
File	Owner	Syllabus	What to implement
models/baseline_cnn.h5	Abu Bakar	—	Output of notebook 01. In .gitignore — never commit
models/mobilenet_transfer.h5	Abu Bakar	—	Output of notebooks 02 and 04. Final deployed model
docs/figures/	Shared	—	Grad-CAM outputs, charts, report screenshots
docs/dip_technique_report.md	Rafay + Aimen	—	Rafay: preprocessing section. Aimen: detection section. Each writes their own part
docs/ds_analysis_report.md	Aimen	—	Baseline curve methodology, anomaly detection results, PKR loss validation
docs/model_results_report.md	Abu Bakar	—	Baseline vs transfer comparison, Grad-CAM analysis, regression head accuracy

3. Workload Split

Member	Files owned	Load
Abu Bakar	config.py, requirements.txt, .gitignore, README.md, ood_validator.py, gradcam_explainer.py, feature_extractor.py, dual_head_predictor.py, 4 Kaggle notebooks, models/	12 files + GPU training (heaviest)
Aimen	4 DIP detection files + 5 DS analytics files + 8 webapp files + 2 notebooks = 19 files	19 files (many are short)
Rafay	white_balance.py, clahe_enhancement.py, bilateral_filter.py, pipeline_runner.py + report section	4 core DIP files + report

Why this split makes sense: Aimen has more files but most webapp files are short (20-50 lines each). Rafay has fewer files but each requires deep OpenCV knowledge and careful testing. Abu Bakar carries the model training burden which runs on Kaggle GPU and cannot be parallelized.


4. Final 8 DIP Techniques (Trimmed from 12)
Four techniques were removed as redundant or low-impact: adaptive median filter (bilateral filter covers this better), standalone Gaussian blur (folded into canny_edge_detector.py), contrast stretching (CLAHE is stronger and covers this), and gamma correction (less relevant for panel soiling — panels are usually well-lit).

Technique	File	Owner	Week	Why it matters for solar panels
White balance	white_balance.py	Rafay	Wk 3	Removes the yellowish cast caused by dust and outdoor lighting. Makes dusty patches more distinct from clean silicon
CLAHE	clahe_enhancement.py	Rafay	Wk 6	Adaptive histogram eq. boosts local contrast between clean and dusty cells, especially in flat-lit or overcast conditions
Bilateral filter	bilateral_filter.py	Rafay	Wk 6	Edge-preserving smooth removes noise inside dusty regions without blurring the sharp grid lines between solar cells
Canny edges	canny_edge_detector.py	Aimen	Wk 7	Detects the panel boundary. Also used for OOD check: non-panel images have weak or random edge structure
Hough transform	hough_panel_locator.py	Aimen	Wk 12	Converts Canny lines into a detected rectangle. Locates the panel frame even when the photo is not perfectly aligned
Morphological closing	morphological_closing.py	Aimen	Wk 16	Fills small gaps in the detected panel edge map. Produces a clean binary mask for cropping before classification
HSV shadow detection	hsv_shadow_glare.py	Aimen	Wk 6	Detects shadow regions using gray hue + low HSV value. Prevents shadow from being misclassified as very dusty
HSV glare detection	hsv_shadow_glare.py	Aimen	Wk 6	Detects overexposed regions (value > 240). Prevents a washed-out photo from being misclassified as clean
