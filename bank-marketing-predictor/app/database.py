# app/database.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictions.db"


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                age INTEGER NOT NULL,
                job TEXT NOT NULL,
                marital TEXT NOT NULL,
                education TEXT NOT NULL,
                balance REAL NOT NULL,
                housing TEXT NOT NULL,
                loan TEXT NOT NULL,
                campaign INTEGER NOT NULL,
                probability REAL NOT NULL,
                prediction TEXT NOT NULL,
                classification TEXT NOT NULL
            )
            """
        )


def save_prediction(features: dict, result: dict) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                age,
                job,
                marital,
                education,
                balance,
                housing,
                loan,
                campaign,
                probability,
                prediction,
                classification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                features["age"],
                features["job"],
                features["marital"],
                features["education"],
                features["balance"],
                features["housing"],
                features["loan"],
                features["campaign"],
                result["probability"],
                result["prediction"],
                result["classification"],
            ),
        )


def list_recent_predictions(limit: int = 20) -> list[dict]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                id,
                created_at,
                age,
                job,
                marital,
                education,
                balance,
                housing,
                loan,
                campaign,
                probability,
                prediction,
                classification
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]