import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("fraud_model.pkl")

st.title("Credit Card Fraud Detection")

input_data = st.text_area("Enter transaction values separated by commas")

if st.button("Predict"):
    try:
        data = np.array([float(x) for x in input_data.split(',')]).reshape(1, -1)

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("Fraud Transaction Detected")
        else:
            st.success("Genuine Transaction")

    except:
        st.warning("Invalid input")
