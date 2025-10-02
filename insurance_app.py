import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("insurance_app.pkl")

st.title("🏥 Insurance Cost Prediction App")
st.write("Enter the details below to predict the insurance cost.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sex = st.selectbox("Sex", options=["female", "male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", options=["no", "yes"])
region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

# Encoding categorical values (adjust based on how dataset was preprocessed)
sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0
region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
region_encoded = region_map[region]

# Collect features
features = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])

# Predict button
if st.button("Predict Insurance Cost"):
    prediction = model.predict(features)
    st.success(f"💰 Predicted Insurance Cost: ${prediction[0]:.2f}")
