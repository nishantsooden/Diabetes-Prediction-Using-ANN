import streamlit as st
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model 

# Get absolute path of Model folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Model", "diabetes_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "Model", "scaler.pkl")

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# UI Title
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes.")

# Create input fields
preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Predict button
if st.button("Predict"):

    # Create input array
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    probability = prediction[0][0]

    # Show result
    if probability > 0.5:
        st.error(f"⚠️ Diabetes Detected (Probability: {probability:.2f})")
    else:
        st.success(f"✅ No Diabetes (Probability: {probability:.2f})")