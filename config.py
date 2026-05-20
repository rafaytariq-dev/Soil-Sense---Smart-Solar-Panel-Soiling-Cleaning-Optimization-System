import os

# System Parameters 
PKR_TARIFF = 65
CLEANING_COST = 800
SYSTEM_KW = 10
LOCATION = {"lat": 33.6007, "lon": 73.0679}

# Folder Paths 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "Models")

# Model Paths [cite: 17, 39]
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_head.keras")
SCALER_PATH      = os.path.join(MODEL_DIR, "scaler.pkl")