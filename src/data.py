"""Dataset download and validation utilities."""

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from src.config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN

DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
SOURCE_COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
    "bmi", "diabetes_pedigree", "age", "outcome",
]


def get_dataset(path: Path = DATA_PATH, download: bool = True) -> pd.DataFrame:
    """Return a validated dataset, downloading it only when it is absent."""
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Dataset not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading PIMA Indians Diabetes dataset...")
        try:
            with urlopen(DATASET_URL, timeout=30) as response:  # nosec B310: fixed public dataset URL
                path.write_bytes(response.read())
        except OSError as exc:
            raise ConnectionError(
                "Unable to download the dataset. Check the internet connection or place the CSV at "
                f"{path}."
            ) from exc

    frame = pd.read_csv(path, header=None, names=SOURCE_COLUMNS)
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    if set(frame.columns) != expected or frame.empty:
        raise ValueError("The dataset has an unexpected schema or contains no rows.")
    if not frame[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Outcome must be binary (0 or 1).")
    return frame
