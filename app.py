"""
app.py — Flask REST API serving the heart disease prediction model.

Endpoints:
    GET  /            -> simple status page
    POST /predict      -> accepts patient details as JSON, returns prediction as JSON
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

@app.route("/")
def home():
    return "Heart Disease Prediction API is running. POST patient data to /predict"


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        missing = [col for col in FEATURE_COLUMNS if col not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        input_df = pd.DataFrame([{col: data[col] for col in FEATURE_COLUMNS}])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": result,
            "probability": round(float(probability), 4),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
