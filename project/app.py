"""
app.py
------
Flask backend for the Student Performance Prediction System.

Flow:
  User -> Frontend (HTML form) -> Flask -> locally trained ML model
       -> prediction -> SQLite (history) -> result shown to user
"""

import os
import sqlite3
from datetime import datetime

import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

# -----------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "student-performance-prediction-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "trained_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
DB_PATH = os.path.join(BASE_DIR, "database", "app.db")

FEATURES = ["studytime", "absences", "failures", "G1", "G2"]

# -----------------------------------------------------------------
# Load the trained model and scaler ONCE at startup
# (the app must not retrain on every request)
# -----------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Trained model not found. Please run 'python train_model.py' first."
    )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# -----------------------------------------------------------------
# Database helpers
# -----------------------------------------------------------------
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_hours TEXT NOT NULL,
            absences INTEGER NOT NULL,
            failures INTEGER NOT NULL,
            g1 INTEGER NOT NULL,
            g2 INTEGER NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def save_prediction(record):
    conn = get_db()
    conn.execute(
        """
        INSERT INTO prediction
            (study_hours, absences, failures, g1, g2, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["study_hours_label"],
            record["absences"],
            record["failures"],
            record["g1"],
            record["g2"],
            record["prediction"],
            record["confidence"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def fetch_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM prediction ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


STUDY_TIME_LABELS = {
    1: "Less than 2 hours/week",
    2: "2 to 5 hours/week",
    3: "5 to 10 hours/week",
    4: "More than 10 hours/week",
}


# -----------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------
def validate_input(form):
    errors = []
    values = {}

    def parse_int(field, min_v, max_v, label):
        raw = form.get(field, "").strip()
        if raw == "":
            errors.append(f"{label} is required.")
            return None
        try:
            val = int(raw)
        except ValueError:
            errors.append(f"{label} must be a whole number.")
            return None
        if val < min_v or val > max_v:
            errors.append(f"{label} must be between {min_v} and {max_v}.")
            return None
        return val

    values["studytime"] = parse_int("studytime", 1, 4, "Study hours category")
    values["absences"] = parse_int("absences", 0, 100, "Absences")
    values["failures"] = parse_int("failures", 0, 4, "Past class failures")
    values["g1"] = parse_int("g1", 0, 20, "Previous marks (G1)")
    values["g2"] = parse_int("g2", 0, 20, "Internal marks (G2)")

    return values, errors


# -----------------------------------------------------------------
# Routes
# -----------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", study_labels=STUDY_TIME_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    values, errors = validate_input(request.form)

    if errors:
        for e in errors:
            flash(e)
        return redirect(url_for("home"))

    try:
        features = np.array(
            [[
                values["studytime"],
                values["absences"],
                values["failures"],
                values["g1"],
                values["g2"],
            ]]
        )
        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = round(float(max(proba)) * 100, 2)

        result_label = "Pass" if pred == 1 else "Fail"

        record = {
            "study_hours_label": STUDY_TIME_LABELS[values["studytime"]],
            "absences": values["absences"],
            "failures": values["failures"],
            "g1": values["g1"],
            "g2": values["g2"],
            "prediction": result_label,
            "confidence": confidence,
        }
        save_prediction(record)

        return render_template("result.html", record=record)

    except Exception as exc:
        flash(f"An error occurred while making the prediction: {exc}")
        return redirect(url_for("home"))


@app.route("/history")
def history():
    rows = fetch_history()
    return render_template("history.html", rows=rows)


# -----------------------------------------------------------------
# Error handlers
# -----------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", study_labels=STUDY_TIME_LABELS), 404


@app.errorhandler(500)
def server_error(e):
    flash("Something went wrong on the server. Please try again.")
    return redirect(url_for("home"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
