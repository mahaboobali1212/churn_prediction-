# src/streamlit_app.py
"""
Streamlit UI for churn prediction.
It will attempt to POST to backend at http://localhost:5000/predict.
If backend is not available, it will try to load local model from ../models/churn_pipeline.joblib
"""

import streamlit as st
import requests
import joblib
import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "..", "models", "churn_pipeline.joblib")
API_URL = "http://localhost:5000/predict"

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("Customer Churn Prediction")
st.write("Fill the customer details and click Predict. The app will call the backend API (if running) or use local model.")

# Define fields (common Telco dataset fields)
with st.form("customer_form"):
    gender = st.selectbox("gender", ["Female","Male"])
    SeniorCitizen = st.selectbox("SeniorCitizen", [0,1])
    Partner = st.selectbox("Partner", ["Yes","No"])
    Dependents = st.selectbox("Dependents", ["Yes","No"])
    tenure = st.number_input("tenure (months)", min_value=0, max_value=200, value=12)
    PhoneService = st.selectbox("PhoneService", ["No","Yes"])
    MultipleLines = st.selectbox("MultipleLines", ["No phone service","No","Yes"])
    InternetService = st.selectbox("InternetService", ["DSL","Fiber optic","No"])
    OnlineSecurity = st.selectbox("OnlineSecurity", ["No","Yes","No internet service"])
    OnlineBackup = st.selectbox("OnlineBackup", ["No","Yes","No internet service"])
    DeviceProtection = st.selectbox("DeviceProtection", ["No","Yes","No internet service"])
    TechSupport = st.selectbox("TechSupport", ["No","Yes","No internet service"])
    StreamingTV = st.selectbox("StreamingTV", ["No","Yes","No internet service"])
    StreamingMovies = st.selectbox("StreamingMovies", ["No","Yes","No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes","No"])
    PaymentMethod = st.selectbox("PaymentMethod", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, max_value=10000.0, value=50.0, format="%.2f")
    TotalCharges = st.number_input("TotalCharges", min_value=0.0, max_value=100000.0, value=600.0, format="%.2f")

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges)
    }

    # Try backend first
    try:
        resp = requests.post(API_URL, json=payload, timeout=3)
        if resp.status_code == 200:
            result = resp.json()
            churn = result.get("churn")
            prob = result.get("probability")
            st.success(f"Prediction from API: {'Churn' if churn==1 else 'No churn'} (prob={prob:.3f})")
        else:
            st.error(f"API returned status {resp.status_code}: {resp.text}")
            raise Exception("Bad API response")
    except Exception:
        st.info("API not reachable — trying local model...")
        try:
            model = joblib.load(MODEL_PATH)
            df = pd.DataFrame([payload])
            prob = float(model.predict_proba(df)[:,1][0])
            churn = int(model.predict(df)[0])
            st.success(f"Local model prediction: {'Churn' if churn==1 else 'No churn'} (prob={prob:.3f})")
        except Exception as e:
            st.error(f"Local model prediction failed: {e}")
            st.write("Make sure you trained the model and models/churn_pipeline.joblib exists.")
