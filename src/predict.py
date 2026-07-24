# src/predict.py
"""
Simple CLI to predict single customer from JSON or inline values.
Example:
python src/predict.py '{"gender":"Female","SeniorCitizen":0, ... }'
"""

import os
import sys
import json
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "churn_pipeline.joblib")

def load_model():
    return joblib.load(MODEL_PATH)

def predict_from_dict(model, data_dict):
    # model expects a dataframe with the same feature columns
    df = pd.DataFrame([data_dict])
    proba = model.predict_proba(df)[:,1][0]
    pred = model.predict(df)[0]
    return pred, proba

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py '{\"gender\":\"Female\",\"SeniorCitizen\":0, ... }'")
        sys.exit(1)
    input_json = sys.argv[1]
    try:
        data = json.loads(input_json)
    except json.JSONDecodeError:
        print("Invalid JSON")
        sys.exit(1)

    model = load_model()
    pred, proba = predict_from_dict(model, data)
    print("Prediction:", "Churn" if pred==1 else "No Churn")
    print("Probability of churn:", round(float(proba), 4))

if __name__ == "__main__":
    main()
