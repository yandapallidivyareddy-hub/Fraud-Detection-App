import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("fraud_model.pkl")

# Page configuration
st.set_page_config(page_title="Fraud Detection", layout="wide")

# Title
st.title("💳 Credit Card Fraud Detection System")
st.markdown("AI-powered system to detect fraudulent transactions")

st.write("### Enter Transaction Details")

# Create input fields dynamically (30 features)
inputs = []

col1, col2, col3 = st.columns(3)

for i in range(30):
    with [col1, col2, col3][i % 3]:
        val = st.number_input(f"Feature {i+1}", value=0.0)
        inputs.append(val)

# Predict button
if st.button("🔍 Predict Transaction"):

    try:
        data = np.array([inputs])

        # Prediction
        prediction = model.predict(data)[0]

        # Probability (if available)
        try:
            prob = model.predict_proba(data)[0][1]
        except:
            prob = None

        st.write("### Result")

        if prediction == 1:
            st.error("🚨 Fraud Transaction Detected")
        else:
            st.success("✅ Genuine Transaction")

        if prob is not None:
            st.info(f"Fraud Probability: {prob:.2f}")

    except Exception as e:
        st.warning("⚠️ Error in input. Please check values.")

# Footer
st.markdown("---")
st.markdown("Developed for Credit Card Fraud Detection using Machine Learning")
