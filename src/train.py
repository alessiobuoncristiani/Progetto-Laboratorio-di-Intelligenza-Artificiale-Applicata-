"""Train, select and persist diabetes prediction models."""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.config import FEATURE_COLUMNS, INVALID_ZERO_COLUMNS, METRICS_PATH, MODEL_PATH, TARGET_COLUMN
from src.data import get_dataset


def replace_invalid_zeros(frame):
    """Convert impossible clinical zero values to missing values."""
    frame = frame.copy()
    frame[INVALID_ZERO_COLUMNS] = frame[INVALID_ZERO_COLUMNS].replace(0, np.nan)
    return frame


def make_pipeline(estimator, scale: bool) -> Pipeline:
    steps = [
        ("invalid_zeros", FunctionTransformer(replace_invalid_zeros, validate=False)),
        ("preprocess", ColumnTransformer([("numeric", SimpleImputer(strategy="median"), FEATURE_COLUMNS)])),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def candidate_models() -> dict:
    return {
        "logistic_regression": (
            make_pipeline(LogisticRegression(max_iter=2000, random_state=42), scale=True),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            make_pipeline(RandomForestClassifier(random_state=42, class_weight="balanced"), scale=False),
            {"classifier__n_estimators": [200, 400], "classifier__max_depth": [None, 5, 10]},
        ),
        "svm": (
            make_pipeline(SVC(probability=True, random_state=42, class_weight="balanced"), scale=True),
            {"classifier__C": [0.1, 1.0, 10.0], "classifier__gamma": ["scale", "auto"]},
        ),
    }


def train_and_save() -> dict:
    data = get_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS], data[TARGET_COLUMN], test_size=0.2, stratify=data[TARGET_COLUMN], random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    searches = {}
    for name, (pipeline, parameters) in candidate_models().items():
        search = GridSearchCV(pipeline, parameters, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(x_train, y_train)
        searches[name] = search

    best_name, best_search = max(searches.items(), key=lambda item: item[1].best_score_)
    probabilities = best_search.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    comparison = {
        name: {"cv_roc_auc": round(float(search.best_score_), 4), "best_parameters": search.best_params_}
        for name, search in searches.items()
    }
    metrics = {
        "selected_model": best_name,
        "test_metrics": {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        },
        "model_comparison": comparison,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": len(data),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_search.best_estimator_, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train_and_save()
