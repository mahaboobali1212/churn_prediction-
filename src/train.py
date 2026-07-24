# src/train.py
"""
Train script for Telco Customer Churn.
Saves trained pipeline to ../models/churn_pipeline.joblib
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(ROOT, "models", "churn_pipeline.joblib")

def load_and_clean(path):
    df = pd.read_csv(path)
    # drop ID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
    # Convert TotalCharges to numeric (some rows are empty strings)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Drop rows with missing target or totalcharges
    df = df.dropna(subset=["Churn"])
    if "TotalCharges" in df.columns:
        df = df.dropna(subset=["TotalCharges"])
    # Map target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def build_pipeline(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # automatic numeric / categorical detection
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()

    # Preprocessing pipelines
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])

    return pipeline, X, y

def main():
    print("Loading dataset...")
    df = load_and_clean(DATA_PATH)
    print("Dataset shape:", df.shape)

    pipeline, X, y = build_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)
    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Ensure models dir exists
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    main()
