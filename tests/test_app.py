import numpy as np
import pytest

from app.app import app, load_model, parse_payload, unusual_fields


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


def test_load_model_is_cached(monkeypatch, tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    calls = []

    def fake_joblib_load(path):
        calls.append(path)
        return object()

    load_model.cache_clear()
    monkeypatch.setattr("app.app.joblib.load", fake_joblib_load)
    first = load_model(model_path)
    second = load_model(model_path)

    assert first is second
    assert calls == [model_path]
    load_model.cache_clear()


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


def test_prediction_page_includes_what_if_controls(monkeypatch):
    class DummyModel:
        def predict_proba(self, frame):
            return np.array([[0.8, 0.2]])

    monkeypatch.setattr("app.app.load_model", lambda path: DummyModel())
    response = app.test_client().post("/", data=valid_payload())
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="what-if-glucose"' in html
    assert 'id="what-if-bmi"' in html


def test_api_predict_rejects_missing_json():
    response = app.test_client().post("/api/predict", json={})
    assert response.status_code == 400
    assert "Campo obbligatorio" in response.get_json()["error"]


def test_api_predict_rejects_non_numeric_value():
    payload = valid_payload()
    payload["glucose"] = "non valido"
    response = app.test_client().post("/api/predict", json=payload)

    assert response.status_code == 400
    assert "deve essere un numero" in response.get_json()["error"]


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_api_predict_rejects_non_finite_value(value):
    payload = valid_payload()
    payload["glucose"] = value
    response = app.test_client().post("/api/predict", json=payload)

    assert response.status_code == 400
    assert "numero finito" in response.get_json()["error"]


def test_api_predict_rejects_boolean_value():
    payload = valid_payload()
    payload["glucose"] = True
    response = app.test_client().post("/api/predict", json=payload)

    assert response.status_code == 400
    assert "deve essere un numero" in response.get_json()["error"]


def test_unusual_fields_warns_about_outlier_and_imputed_zero():
    values = valid_payload()
    values["glucose"] = 280
    values["insulin"] = 0

    warnings = unusual_fields(values)

    assert len(warnings) == 2
    assert "Glucosio" in warnings[0]
    assert "dato mancante" in warnings[1]


def test_health_reports_model_status():
    response = app.test_client().get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
