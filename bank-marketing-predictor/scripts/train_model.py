# scripts/train_model.py
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "bank.csv"
MODELS_DIR = BASE_DIR / "models"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")  # UCI usa ; como separador
    return df


def train_and_save() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # Features recomendadas
    features = [
        "age",
        "job",
        "marital",
        "education",
        "balance",
        "housing",
        "loan",
        "campaign",
    ]
    target = "y"

    df = df[features + [target]]

    # y: yes/no -> 1/0
    y = df[target].map({"no": 0, "yes": 1}).astype(int)
    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_features = ["age", "balance", "campaign"]
    categorical_features = ["job", "marital", "education", "housing", "loan"]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
        "n_rows": int(len(df)),
        "features": features,
    }

    model_path = MODELS_DIR / "bank_marketing_pipeline.joblib"
    metrics_path = MODELS_DIR / "metrics.json"

    joblib.dump(pipeline, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    train_and_save()