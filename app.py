import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('fraud_model.keras', compile=False)

st.title("Credit Card Fraud Detection")

st.write("Enter transaction values separated by commas")

input_data = st.text_area("Input Data")

if st.button("Predict"):
    try:
        data = np.array([float(x) for x in input_data.split(',')]).reshape(1, -1)
        prediction = model.predict(data)
        prob = prediction[0][0]

        st.write(f"Fraud Probability: {prob:.4f}")

        if prob > 0.5:
            st.error("Fraud Transaction Detected")
        else:
            st.success("Genuine Transaction")

    except:
        st.warning("Invalid input")