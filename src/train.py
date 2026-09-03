"""Train and persist the final model used by the Flask application.

The comparison and tuning of Logistic Regression and KNN are documented in the
optimization notebook. This operational script trains only the selected
polynomial Logistic Regression pipeline.
"""

import json
from datetime import datetime, timezone

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
)
from src.data import get_dataset
from src.preprocessing import replace_invalid_zeros


def make_pipeline(estimator, polynomial_degree: int | None = None) -> Pipeline:
    """Build the shared preprocessing + estimator pipeline."""
    steps = [
        ("invalid_zeros", FunctionTransformer(replace_invalid_zeros, validate=False)),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if polynomial_degree is not None:
        steps.append(("polynomial", PolynomialFeatures(degree=polynomial_degree, include_bias=False)))
    steps.extend([
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ])
    return Pipeline(steps)


def final_model() -> Pipeline:
    """Build the tuned model selected in the optimization notebook."""
    return make_pipeline(
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, random_state=42),
        polynomial_degree=2,
    )


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
    """Evaluate, refit and save the selected final model."""
    data = get_dataset()
    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    selected_name = "logistic_regression_polynomial"
    model = final_model()
    model.fit(x_train, y_train)
    test_metrics = evaluate_model(model, x_test, y_test)

    # Refit only after evaluation, using all labelled data available.
    model.fit(x, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "selected_model": selected_name,
        "configuration": {
            "polynomial_degree": 2,
            "C": 0.1,
            "class_weight": "balanced",
            "decision_threshold": 0.5,
        },
        "test": test_metrics,
        "dataset_rows": len(data),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train_and_save()
