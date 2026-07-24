# src/predict_api.py
"""
Flask API to serve churn predictions.
POST /predict  with JSON body (single customer fields) returns {"churn": 0/1, "probability": float}
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # allow cross-origin (useful for local frontend)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "..", "models", "churn_pipeline.joblib")
model = None

def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model

@app.route("/")
def index():
    return "Churn Prediction API is up. POST JSON to /predict"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON:
    { "gender":"Female", "SeniorCitizen":0, "Partner":"Yes", ... }
    Must include the same raw feature column names as dataset (see README for list)
    """
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({"error": "Expected JSON object for a single customer"}), 400

        model = load_model()
        df = pd.DataFrame([data])

        # Predict
        proba = float(model.predict_proba(df)[:,1][0])
        pred = int(model.predict(df)[0])

        return jsonify({"churn": pred, "probability": proba})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Loading model...")
    load_model()
    # Run on localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
