"""Flask UI and REST API for diabetes-risk predictions."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.config import FEATURE_COLUMNS, MODEL_PATH

FIELD_LABELS = {
    "pregnancies": "Numero di gravidanze", "glucose": "Glucosio (mg/dL)",
    "blood_pressure": "Pressione arteriosa (mm Hg)", "skin_thickness": "Spessore cutaneo (mm)",
    "insulin": "Insulina (mu U/ml)", "bmi": "BMI (kg/m²)",
    "diabetes_pedigree": "Diabetes Pedigree Function", "age": "Età (anni)",
}
FIELD_HINTS = {
    "pregnancies": "Numero di gravidanze effettuate. Lo zero è un valore valido.",
    "glucose": "Concentrazione di glucosio nel sangue; lo zero indica un dato mancante.",
    "blood_pressure": "Pressione diastolica; lo zero indica un dato mancante.",
    "skin_thickness": "Spessore della plica cutanea; lo zero indica un dato mancante.",
    "insulin": "Valore dell'insulina nel sangue; lo zero indica un dato mancante.",
    "bmi": "Indice di massa corporea; lo zero indica un dato mancante.",
    "diabetes_pedigree": "Indicatore della familiarità per il diabete.",
    "age": "Età della persona esaminata, in anni.",
}
FIELD_PLACEHOLDERS = {
    "pregnancies": "es. 2",
    "glucose": "es. 120",
    "blood_pressure": "es. 70",
    "skin_thickness": "es. 20",
    "insulin": "es. 79",
    "bmi": "es. 28.5",
    "diabetes_pedigree": "es. 0.3",
    "age": "es. 35",
}
FIELD_GROUPS = [
    {
        "title": "Dati personali",
        "description": "Informazioni anagrafiche e storia delle gravidanze.",
        "fields": ["pregnancies", "age"],
    },
    {
        "title": "Misure cliniche",
        "description": "Valori rilevati durante la visita o l'esame diagnostico.",
        "fields": ["glucose", "blood_pressure", "bmi"],
    },
    {
        "title": "Altri indicatori",
        "description": "Parametri aggiuntivi utilizzati dal modello predittivo.",
        "fields": ["skin_thickness", "insulin", "diabetes_pedigree"],
    },
]
NON_NEGATIVE_FIELDS = set(FEATURE_COLUMNS)
TYPICAL_RANGES = {
    "pregnancies": (0, 17),
    "glucose": (50, 200),
    "blood_pressure": (40, 120),
    "skin_thickness": (5, 60),
    "insulin": (10, 600),
    "bmi": (10, 60),
    "diabetes_pedigree": (0.08, 2.0),
    "age": (18, 90),
}


def parse_payload(payload: dict) -> dict:
    """Validate a prediction request and return numeric feature values."""
    if not isinstance(payload, dict):
        raise ValueError("Il corpo della richiesta deve essere un oggetto JSON.")
    values = {}
    for field in FEATURE_COLUMNS:
        raw_value = payload.get(field)
        if raw_value is None or raw_value == "":
            raise ValueError(f"Campo obbligatorio mancante: {FIELD_LABELS[field]}.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{FIELD_LABELS[field]} deve essere un numero.") from exc
        if field in NON_NEGATIVE_FIELDS and value < 0:
            raise ValueError(f"{FIELD_LABELS[field]} non può essere negativo.")
        values[field] = value
    return values


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("Modello non disponibile. Esegui prima: python -m src.train")
    return joblib.load(path)


def unusual_fields(values: dict) -> list[str]:
    """Return non-blocking warnings for values outside typical dataset ranges."""
    warnings = []
    for field, (lower, upper) in TYPICAL_RANGES.items():
        value = values[field]
        if value != 0 and not lower <= value <= upper:
            warnings.append(
                f"{FIELD_LABELS[field]} ({value:g}) è fuori dall'intervallo tipico "
                f"osservato nel dataset ({lower:g}–{upper:g})."
            )
    return warnings


def create_app(model_path: Path = MODEL_PATH) -> Flask:
    app = Flask(__name__)
    app.config["MODEL_PATH"] = model_path

    def predict(values: dict) -> dict:
        model = load_model(app.config["MODEL_PATH"])
        probability = float(model.predict_proba(pd.DataFrame([values], columns=FEATURE_COLUMNS))[0, 1])
        predicted_class = int(probability >= 0.5)
        return {
            "prediction": predicted_class,
            "label": "Rischio elevato" if predicted_class else "Rischio non elevato",
            "probability": round(probability, 4),
            "warnings": unusual_fields(values),
            "disclaimer": "Risultato didattico: non costituisce una diagnosi medica.",
        }

    @app.get("/")
    def index():
        return render_template("index.html", groups=FIELD_GROUPS, labels=FIELD_LABELS, hints=FIELD_HINTS, placeholders=FIELD_PLACEHOLDERS)

    @app.post("/")
    def form_prediction():
        try:
            result = predict(parse_payload(request.form.to_dict()))
            return render_template("index.html", groups=FIELD_GROUPS, labels=FIELD_LABELS, hints=FIELD_HINTS, placeholders=FIELD_PLACEHOLDERS, result=result, values=request.form)
        except (ValueError, FileNotFoundError) as exc:
            return render_template("index.html", groups=FIELD_GROUPS, labels=FIELD_LABELS, hints=FIELD_HINTS, placeholders=FIELD_PLACEHOLDERS, error=str(exc), values=request.form), 400

    @app.post("/api/predict")
    def api_prediction():
        try:
            return jsonify(predict(parse_payload(request.get_json(silent=True)))), 200
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "model_available": app.config["MODEL_PATH"].exists()})

    return app


app = create_app()
