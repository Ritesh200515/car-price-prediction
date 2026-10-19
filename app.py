import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Set page title
st.title("🚘 Car Price Prediction App")
st.write("Enter the vehicle specifications below to estimate its market price.")

# 2. Load the trained model artifacts
@st.cache_resource
def load_model_artifacts():
    return joblib.load('car_price_model.pkl')

artifacts = load_model_artifacts()
model = artifacts['model']
encoders = artifacts['encoders']

# 3. Layout Inputs using columns
col1, col2 = st.columns(2)

with col1:
    manufacturer = st.selectbox("Manufacturer", encoders['Manufacturer'].classes_)
    model_car = st.selectbox("Model", encoders['Model'].classes_)
    prod_year = st.number_input("Production Year", min_value=1950, max_value=2026, value=2015)
    category = st.selectbox("Category", encoders['Category'].classes_)
    leather = st.selectbox("Leather Interior", encoders['Leather interior'].classes_)
    fuel = st.selectbox("Fuel Type", encoders['Fuel type'].classes_)
    levy = st.number_input("Levy (Tax)", min_value=0.0, value=0.0)

with col2:
    engine_vol = st.number_input("Engine Volume (L)", min_value=0.0, max_value=10.0, value=2.0)
    mileage = st.number_input("Mileage (km)", min_value=0.0, value=50000.0)
    cylinders = st.number_input("Cylinders", min_value=1, max_value=16, value=4)
    gearbox = st.selectbox("Gear Box Type", encoders['Gear box type'].classes_)
    drive_wheels = st.selectbox("Drive Wheels", encoders['Drive wheels'].classes_)
    doors = st.selectbox("Doors", [2, 4, 5])
    wheel = st.selectbox("Wheel (Left/Right)", encoders['Wheel'].classes_)
    color = st.selectbox("Color", encoders['Color'].classes_)
    airbags = st.number_input("Airbags", min_value=0, max_value=20, value=4)

# 4. Predict Button
if st.button("Predict Price", type="primary"):
    try:
        input_data = pd.DataFrame([{
            'Levy': levy,
            'Manufacturer': encoders['Manufacturer'].transform([manufacturer])[0],
            'Model': encoders['Model'].transform([model_car])[0],
            'Prod. year': prod_year,
            'Category': encoders['Category'].transform([category])[0],
            'Leather interior': encoders['Leather interior'].transform([leather])[0],
            'Fuel type': encoders['Fuel type'].transform([fuel])[0],
            'Engine volume': engine_vol,
            'Mileage': mileage,
            'Cylinders': cylinders,
            'Gear box type': encoders['Gear box type'].transform([gearbox])[0],
            'Drive wheels': encoders['Drive wheels'].transform([drive_wheels])[0],
            'Doors': float(doors),
            'Wheel': encoders['Wheel'].transform([wheel])[0],
            'Color': encoders['Color'].transform([color])[0],
            'Airbags': airbags
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"### Estimated Price: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"Error processing input parameters: {e}")