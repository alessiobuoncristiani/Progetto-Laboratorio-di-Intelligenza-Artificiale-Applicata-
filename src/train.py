"""Train and persist the model used by the Flask application.

The experiment is deliberately limited to the two algorithms documented in
the machine-learning notebook: Logistic Regression and KNN.
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    INVALID_ZERO_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
)
from src.data import get_dataset


def replace_invalid_zeros(frame):
    """Convert clinically implausible zero measurements into missing values."""
    frame = frame.copy()
    frame[INVALID_ZERO_COLUMNS] = frame[INVALID_ZERO_COLUMNS].replace(0, np.nan)
    return frame


def make_pipeline(estimator) -> Pipeline:
    """Build the shared preprocessing + estimator pipeline."""
    return Pipeline(
        [
            ("invalid_zeros", FunctionTransformer(replace_invalid_zeros, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


def candidate_models() -> dict[str, Pipeline]:
    """Return the two models compared in the notebook."""
    return {
        "logistic_regression": make_pipeline(
            LogisticRegression(max_iter=2000, random_state=42)
        ),
        "knn": make_pipeline(KNeighborsClassifier(n_neighbors=15)),
    }


def evaluate_model(model, x_test, y_test) -> dict[str, float]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


def train_and_save() -> dict:
    """Evaluate both candidates and save the documented final model."""
    data = get_dataset()
    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    models = candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    comparison = {}
    for name, model in models.items():
        cv_results = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring)
        model.fit(x_train, y_train)
        comparison[name] = {
            "cv_mean": {metric: round(float(cv_results[f"test_{metric}"].mean()), 4) for metric in scoring},
            "cv_std": {metric: round(float(cv_results[f"test_{metric}"].std()), 4) for metric in scoring},
            "test": evaluate_model(model, x_test, y_test),
        }

    # This is the model selected and documented in 02_Machine_Learning.ipynb.
    selected_name = "logistic_regression"
    final_model = models[selected_name]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    metrics = {
        "selected_model": selected_name,
        "comparison": comparison,
        "dataset_rows": len(data),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train_and_save()
