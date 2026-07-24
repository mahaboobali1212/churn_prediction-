import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

# 📂 Paths
DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_PATH = "model.pkl"

# --- Load dataset ---
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

# --- Load model ---
def load_model() -> BaseEstimator | None:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

# --- Train model ---
def train_model(df: pd.DataFrame) -> BaseEstimator | None:
    df = df.copy()
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    for col in df.select_dtypes(include="object").columns:
        if col != "Churn":
            df[col] = LabelEncoder().fit_transform(df[col])
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes":1, "No":0})
    X = df.drop(columns=["Churn"], errors="ignore")
    y = df["Churn"] if "Churn" in df.columns else None
    model: BaseEstimator = RandomForestClassifier(n_estimators=100, random_state=42)
    if y is not None:
        model.fit(X, y)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        return model
    return None

# --- Generate reason for churn ---
def churn_reason(row):
    reasons = []
    if "tenure" in row and row["tenure"] < 12:
        reasons.append("Low tenure")
    if "MonthlyCharges" in row and row["MonthlyCharges"] > 80:
        reasons.append("High monthly charges")
    if "SeniorCitizen" in row and row["SeniorCitizen"] == 1:
        reasons.append("Senior citizen")
    if not reasons:
        reasons.append("Other factors")
    return ", ".join(reasons)

# --- Streamlit App ---
st.title("🔮 Customer Churn Prediction App")

df = load_data()
st.info(f"📂 Loaded dataset with {len(df)} rows")

# Train button
model: BaseEstimator | None = None
if st.button("🛠️ Train Model"):
    with st.spinner("Training model..."):
        model = train_model(df)
    if model:
        st.success("✅ Model trained and saved successfully!")
    else:
        st.error("❌ Training failed. Check dataset format.")

if os.path.exists(MODEL_PATH) and model is None:
    model = load_model()

# --- Input fields ---
gender = st.selectbox(
    "Gender", ["Any"] + list(df["gender"].unique())
) if "gender" in df.columns else None

senior = st.selectbox(
    "Senior Citizen", ["Any"] + list(df["SeniorCitizen"].unique())
) if "SeniorCitizen" in df.columns else None

# Tenure input
if "tenure" in df.columns:
    tenure_min = int(st.number_input("Tenure Min", int(df["tenure"].min()), int(df["tenure"].max()), int(df["tenure"].min())))
    tenure_max = int(st.number_input("Tenure Max", int(df["tenure"].min()), int(df["tenure"].max()), int(df["tenure"].max())))
else:
    tenure_min, tenure_max = None, None

# Monthly Charges input
if "MonthlyCharges" in df.columns:
    monthly_min = float(st.number_input("Monthly Charges Min", float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()), float(df["MonthlyCharges"].min())))
    monthly_max = float(st.number_input("Monthly Charges Max", float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()), float(df["MonthlyCharges"].max())))
else:
    monthly_min, monthly_max = None, None

# Predict button
if st.button("🔍 Predict Churn") and model is not None:
    matched = df.copy()
    if gender and gender != "Any":
        matched = matched[matched["gender"] == gender]
    if senior and senior != "Any":
        matched = matched[matched["SeniorCitizen"] == int(senior)]
    if tenure_min is not None and tenure_max is not None:
        matched = matched[(matched["tenure"] >= tenure_min) & (matched["tenure"] <= tenure_max)]
    if monthly_min is not None and monthly_max is not None:
        matched = matched[(matched["MonthlyCharges"] >= monthly_min) & (matched["MonthlyCharges"] <= monthly_max)]

    if matched.empty:
        st.warning("⚠️ No customers found with those inputs.")
    else:
        # Prepare features
        X = matched.copy()
        if "customerID" in X.columns:
            X = X.drop(columns=["customerID"])
        X = X.drop(columns=["Churn"], errors="ignore")
        categorical_cols = X.select_dtypes(include="object").columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0
            X = X[model.feature_names_in_]

        # Predict
        preds = model.predict(X)
        matched["Churn Prediction"] = preds
        matched["Reason"] = matched.apply(lambda row: churn_reason(row) if row["Churn Prediction"] == 1 else "", axis=1)

        st.success(f"✅ Found {len(matched)} matching customer(s).")
        st.dataframe(matched)

        churn_counts = matched["Churn Prediction"].value_counts()
        churned_reasons = matched[matched["Churn Prediction"] == 1]["Reason"].value_counts()

        # --- Pie Chart (smaller) ---
        st.subheader("📊 Churn Distribution (Pie Chart)")
        fig1, ax1 = plt.subplots(figsize=(3, 3))  # smaller size
        wedges, texts, autotexts = ax1.pie(
            churn_counts,
            labels=["No Churn (0)", "Churn (1)"],
            autopct="%1.1f%%",
            colors=["#66b3ff", "#ff6666"],
            startangle=90
        )
        ax1.axis("equal")
        plt.tight_layout()
        st.pyplot(fig1)

        # --- Churn Reasons Bar Chart (with legend only) ---
        st.subheader("📈 Churn Reasons (Bar Chart)")
        if not churned_reasons.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 4))

            # Assign different colors for reasons
            colors = sns.color_palette("Set2", len(churned_reasons))
            bars = ax2.bar(range(len(churned_reasons)), churned_reasons.values, color=colors)

            # Remove reason names from x-axis
            ax2.set_xticks([])  
            ax2.set_ylabel("Number of Customers")

            # Add legend mapping color → reason
            ax2.legend(
                bars,
                churned_reasons.index,
                title="Reasons",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                ncol=2
            )

            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.write("No churners found, so no reasons to display.")

        # Download results
        csv = matched.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Results as CSV",
            csv,
            "matching_customers_predictions.csv",
            "text/csv",
            key="download-csv"
        )
