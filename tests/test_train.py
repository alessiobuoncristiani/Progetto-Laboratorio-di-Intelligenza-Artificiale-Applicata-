import numpy as np

from src.train import evaluate_model, final_model


def test_final_model_uses_selected_tuned_configuration():
    model = final_model()

    assert list(model.named_steps) == ["invalid_zeros", "imputer", "polynomial", "scaler", "classifier"]
    assert model.named_steps["polynomial"].degree == 2
    classifier = model.named_steps["classifier"]
    assert classifier.C == 0.1
    assert classifier.class_weight == "balanced"


def test_evaluate_model_uses_probability_threshold_and_metrics():
    class DummyModel:
        def predict_proba(self, features):
            return np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]])

    metrics = evaluate_model(DummyModel(), [[0], [1], [2], [3]], [0, 1, 1, 0])

    assert metrics == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
    }
