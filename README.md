# 📊 Telco Customer Churn Prediction & Retention Intelligence

An end-to-end Machine Learning solution to predict customer churn, evaluate financial risk, identify key churn drivers, and provide automated customer retention recommendations for telecommunications and subscription businesses.

---

## 🏗️ Project Architecture

```
churn_prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv       # Historical dataset (demographics, services, charges)
├── models/
│   └── churn_pipeline.joblib          # Scikit-learn Pipeline (Imputer + Scaler + OHE + Tuned RF)
├── src/
│   ├── train.py                       # Data preprocessing, pipeline building & model training
│   ├── evaluate.py                    # Model evaluation (Accuracy, ROC-AUC, PR-AUC, Confusion Matrix)
│   ├── predict_api.py                 # Flask REST API backend (/predict, /health endpoints)
│   ├── streamlit_app.py               # Interactive Single-Customer Prediction & Retention UI
│   └── predict.py                     # CLI tool for single-customer churn scoring
├── app.py                             # All-in-one Segment Analytics & Batch Prediction Dashboard
├── churn_results.csv                  # Sample exported predictions
├── model.pkl                          # Model artifact for standalone app compatibility
└── requirements.txt                   # Project dependencies
```

---

## ⚡ Quick Start

### 1. Setup Environment & Install Dependencies
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Machine Learning Pipeline
```powershell
python src/train.py
```
*Trains an end-to-end ColumnTransformer + balanced Random Forest pipeline and saves it to `models/churn_pipeline.joblib`.*

### 3. Evaluate the Model
```powershell
python src/evaluate.py
```
*Outputs Accuracy, ROC-AUC, PR-AUC, Precision, Recall, F1-Score, and Confusion Matrix on a 20% stratified test split.*

---

## 🖥️ Running the Applications

### Option A: Interactive Retention Web Dashboard (Streamlit UI + Flask API)

1. **Start the Flask Backend API:**
   ```powershell
   python src/predict_api.py
   ```
   *Runs at `http://localhost:5000`*

2. **Start the Streamlit Frontend:**
   ```powershell
   streamlit run src/streamlit_app.py
   ```
   *Features: Organized 3-column input form, churn probability gauge, risk classification (Low / Medium / High), risk factors, and automated retention action plans.*

### Option B: Standalone Segment Analytics & Batch Prediction App
```powershell
streamlit run app.py
```
*Features: Segment filtering (contract type, tenure, charges), model retraining in-app, risk breakdown pie charts, top churn drivers bar chart, and CSV export.*

### Option C: Command-Line Interface (CLI) Single Prediction
```powershell
python src/predict.py "{gender: Female, SeniorCitizen: 0, tenure: 2, MonthlyCharges: 95.0, Contract: Month-to-month, InternetService: Fiber optic, TechSupport: No, PaymentMethod: Electronic check}"
```

---

## 🔌 REST API Endpoints

### `POST /predict`
**Request Payload (JSON):**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.0,
  "TotalCharges": 190.0
}
```

**Response (JSON):**
```json
{
  "churn": 1,
  "probability": 0.9031,
  "risk_level": "High",
  "risk_factors": [
    "Month-to-month contract (high cancellation flexibility)",
    "Short tenure (2 months) - high early lifecycle risk",
    "High monthly bill ($95.00/month)",
    "Fiber optic subscription without technical support",
    "No online security add-on",
    "Payment via Electronic Check (statistically higher churn rate)"
  ]
}
```


