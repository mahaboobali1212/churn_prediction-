# src/evaluate.py
"""
Load saved model and run evaluation on test split from dataset.
Generates simple text metrics. Useful to re-check after training.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(ROOT, "models", "churn_pipeline.joblib")

def load_and_clean(path):
    df = pd.read_csv(path)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["Churn"])
    if "TotalCharges" in df.columns:
        df = df.dropna(subset=["TotalCharges"])
    df["Churn"] = df["Churn"].map({"Yes":1,"No":0})
    return df

def main():
    df = load_and_clean(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=42)
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
