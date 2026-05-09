import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("fraud_model.pkl")

# Page settings
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

# Title
st.title("💳 Credit Card Fraud Detection")

st.write(
    "This system predicts whether a transaction is Genuine or Fraudulent."
)

st.write(
    "Enter the transaction behavior scores below."
)

# Friendly feature names
feature_labels = {
    "V17": "Transaction Risk Score",
    "V14": "Spending Pattern Score",
    "V12": "Account Activity Score",
    "V10": "Transaction Behavior Score",
    "V16": "Fraud Risk Indicator"
}

# Feature descriptions
feature_help = {
    "V17": "Higher unusual values may indicate suspicious transactions.",
    "V14": "Represents abnormal spending patterns.",
    "V12": "Represents account activity behavior.",
    "V10": "Represents transaction behavior changes.",
    "V16": "Indicates overall fraud-related risk."
}

user_input = []

# Input fields
for feature in feature_labels:

    value = st.number_input(
        label=feature_labels[feature],
        help=feature_help[feature],
        value=0.0
    )

    user_input.append(value)

# Predict button
if st.button("🔍 Predict Transaction"):

    input_array = np.array(user_input).reshape(1, -1)

    prediction = model.predict(input_array)[0]

    probability = model.predict_proba(input_array)[0][1]

    if prediction == 1:

        st.error(
            f"🚨 Fraudulent Transaction Detected\n\nFraud Probability: {probability:.2f}"
        )

    else:

        st.success(
            f"✅ Genuine Transaction\n\nFraud Probability: {probability:.2f}"
        )

# Footer
st.write("---")
st.write("Developed using CTGAN and XGBoost")        
