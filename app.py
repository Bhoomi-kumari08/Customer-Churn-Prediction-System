import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configure the page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the trained model and encoders
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "customer_churn_model.pkl")
encoders = joblib.load(BASE_DIR / "models" / "label_encoders.pkl")

st.markdown(
    "<h1 style='text-align: center;'>📊 Customer Churn Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align: center; font-size: 18px; color: gray;'>
    This application predicts whether a telecom customer is likely to churn
    using a trained Random Forest Machine Learning model.
    </p>
    """,
    unsafe_allow_html=True
)

st.header("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=1)

    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.number_input("Monthly Charges", min_value=0.0)
    total = st.number_input("Total Charges", min_value=0.0)

predict = st.button("Predict Churn")
if predict:

    # Convert inputs to the same format as the training data
    input_data = {
        "gender": encoders["gender"].transform([gender])[0],
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": encoders["Partner"].transform([partner])[0],
        "Dependents": encoders["Dependents"].transform([dependents])[0],
        "tenure": tenure,
        "PhoneService": encoders["PhoneService"].transform([phone])[0],
        "MultipleLines": encoders["MultipleLines"].transform([multiple])[0],
        "InternetService": encoders["InternetService"].transform([internet])[0],
        "OnlineSecurity": encoders["OnlineSecurity"].transform([security])[0],
        "OnlineBackup": encoders["OnlineBackup"].transform([backup])[0],
        "DeviceProtection": encoders["DeviceProtection"].transform([protection])[0],
        "TechSupport": encoders["TechSupport"].transform([support])[0],
        "StreamingTV": encoders["StreamingTV"].transform([tv])[0],
        "StreamingMovies": encoders["StreamingMovies"].transform([movies])[0],
        "Contract": encoders["Contract"].transform([contract])[0],
        "PaperlessBilling": encoders["PaperlessBilling"].transform([paperless])[0],
        "PaymentMethod": encoders["PaymentMethod"].transform([payment])[0],
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    st.write(f"**Churn Probability:** {probability:.2%}")

    st.info("Model Used: Random Forest Classifier")