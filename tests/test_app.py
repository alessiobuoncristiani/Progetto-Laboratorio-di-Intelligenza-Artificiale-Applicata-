import pytest
import numpy as np

from app.app import app, parse_payload


def valid_payload():
    return {"pregnancies": 2, "glucose": 120, "blood_pressure": 70, "skin_thickness": 20, "insulin": 80, "bmi": 28.5, "diabetes_pedigree": 0.3, "age": 35}


def test_parse_payload_converts_numbers():
    parsed = parse_payload(valid_payload())
    assert parsed["glucose"] == 120.0
    assert set(parsed) == set(valid_payload())


@pytest.mark.parametrize("field,value", [("age", -1), ("bmi", -2), ("pregnancies", -1)])
def test_parse_payload_rejects_negative_values(field, value):
    payload = valid_payload()
    payload[field] = value
    with pytest.raises(ValueError):
        parse_payload(payload)


def test_parse_payload_requires_all_fields():
    payload = valid_payload()
    payload.pop("age")
    with pytest.raises(ValueError, match="Campo obbligatorio"):
        parse_payload(payload)


def test_home_page_is_available():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert "Stima del rischio di diabete" in response.get_data(as_text=True)


def test_api_predict_returns_prediction(monkeypatch):
    class DummyModel:
        def predict_proba(self, frame):
            return np.array([[0.8, 0.2]])

    monkeypatch.setattr("app.app.load_model", lambda path: DummyModel())
    response = app.test_client().post("/api/predict", json=valid_payload())
    data = response.get_json()

    assert response.status_code == 200
    assert data["prediction"] == 0
    assert data["probability"] == 0.2


def test_api_predict_rejects_missing_json():
    response = app.test_client().post("/api/predict", json={})
    assert response.status_code == 400
    assert "Campo obbligatorio" in response.get_json()["error"]


def test_health_reports_model_status():
    response = app.test_client().get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
