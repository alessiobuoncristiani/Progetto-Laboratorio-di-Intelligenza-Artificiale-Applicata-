from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "diabetes.csv"
MODEL_PATH = ROOT_DIR / "models" / "diabetes_model.joblib"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

FEATURE_COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age",
]
TARGET_COLUMN = "outcome"
INVALID_ZERO_COLUMNS = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
