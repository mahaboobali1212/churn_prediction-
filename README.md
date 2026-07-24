# Churn Prediction Project

## Setup
1. Create & activate virtual environment:
   python -m venv venv
   # On Linux/Mac:
   source venv/bin/activate
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1

2. Install dependencies:
   pip install -r requirements.txt

3. Put dataset:
   Place `Telco-Customer-Churn.csv` in `data/`

## Train model
From project root:
python src/train.py

This trains and saves pipeline to `models/churn_pipeline.joblib`.

## Run backend API (Flask)
python src/predict_api.py
This starts server at: http://localhost:5000
Example curl:
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":5,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}'

## Run frontend (Streamlit)
streamlit run src/streamlit_app.py

The Streamlit app will attempt to call the backend at http://localhost:5000/predict; if backend not running it will try local model file.

## Extras
- Use `src/evaluate.py` to re-evaluate a saved model.
- Use `src/predict.py` to predict a single sample from CLI:
  python src/predict.py '{"gender":"Female","SeniorCitizen":0,...}'

