import streamlit as st
import numpy as np
import joblib

model = joblib.load("fraud_model.pkl")
features = joblib.load("features.pkl")

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details")

user_input = []

for feature in features:

    value = st.number_input(
        f"Enter {feature}",
        value=0.0
    )

    user_input.append(value)

if st.button("Predict"):

    input_array = np.array(user_input).reshape(1, -1)

    prediction = model.predict(input_array)[0]

    probability = model.predict_proba(input_array)[0][1]

    if prediction == 1:

        st.error(
            f"🚨 Fraud Transaction\nProbability: {probability:.2f}"
        )

    else:

        st.success(
            f"✅ Genuine Transaction\nProbability: {probability:.2f}"
        )
