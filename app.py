

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Lifestyle Weight Prediction")

stress = st.number_input("Stress Level", 1, 10, 5)

calories = st.number_input("Calories Consumed", 500, 6000, 2000)

protein = st.number_input("Protein (g)", 0, 300, 80)

carbs = st.number_input("Carbs (g)", 0, 500, 250)

fat = st.number_input("Fat (g)", 0, 200, 60)

if st.button("Predict"):

    data = pd.DataFrame({
        "Stress_Level":[stress],
        "Calories_Consumed":[calories],
        "Protein_g":[protein],
        "Carbs_g":[carbs],
        "Fat_g":[fat]
    })

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(f"Predicted Weight = {prediction[0]:.2f} kg")