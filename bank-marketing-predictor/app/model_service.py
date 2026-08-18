# app/model_service.py
from pathlib import Path

import joblib
import pandas as pd

FEATURE_ORDER = [
    "age",
    "job",
    "marital",
    "education",
    "balance",
    "housing",
    "loan",
    "campaign",
]

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "bank_marketing_pipeline.joblib"
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Run scripts/train_model.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(payload: dict) -> dict:
    model = get_model()

    row = {k: payload[k] for k in FEATURE_ORDER}
    sample = pd.DataFrame([row], columns=FEATURE_ORDER)

    proba_yes = float(model.predict_proba(sample)[0, 1])
    pred_int = int(proba_yes >= 0.5)
    prediction = "yes" if pred_int == 1 else "no"

    if proba_yes >= 0.5:
        classification = "Potencialmente interesado"
    elif proba_yes >= 0.2:
        classification = "Interés moderado"
    else:
        classification = "Poco probable"

    return {
        "probability": round(proba_yes, 4),
        "prediction": prediction,
        "classification": classification,
    }