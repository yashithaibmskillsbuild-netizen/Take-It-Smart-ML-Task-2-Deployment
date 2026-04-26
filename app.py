import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏡 House Price Prediction App")

st.write("Enter house details below:")

# Input fields
sqft = st.number_input("Square Footage", min_value=0)
bedrooms = st.number_input("Number of Bedrooms", min_value=0)
bathrooms = st.number_input("Number of Bathrooms", min_value=0)
year = st.number_input("Year Built", min_value=1900, max_value=2025)
lot = st.number_input("Lot Size", min_value=0.0)
garage = st.number_input("Garage Size", min_value=0)
neigh = st.number_input("Neighborhood Quality (1-10)", min_value=1, max_value=10)

# Predict button
if st.button("Predict Price"):
    input_data = np.array([[sqft, bedrooms, bathrooms, year, lot, garage, neigh]])
    
    # Apply scaling (if used)
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)

    st.success(f"🏠 Estimated House Price: ₹ {prediction[0]:,.2f}")