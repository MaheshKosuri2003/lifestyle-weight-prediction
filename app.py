import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration (MUST BE FIRST)
# -----------------------------
st.set_page_config(
    page_title="Lifestyle Weight Change Predictor",
    page_icon="🏋️",
    layout="wide"
)

# -----------------------------
# Load CSS
# -----------------------------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("css/style.css")

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = joblib.load("best_model1.pkl")
scaler = joblib.load("scaler1.pkl")

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<h1 class="main-title">
🏋️ Lifestyle Weight Change Predictor
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p class="sub-title">
Predict your weight change using your daily nutrition and stress level.
</p>
""", unsafe_allow_html=True)

st.write("---")

# -----------------------------
# Input Form
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    stress = st.slider(
        "Stress Level",
        min_value=1,
        max_value=10,
        value=5
    )

    calories = st.number_input(
        "Calories Consumed",
        min_value=500,
        max_value=6000,
        value=2200
    )

    protein = st.number_input(
        "Protein (g)",
        min_value=0,
        max_value=300,
        value=80
    )

with col2:
    carbs = st.number_input(
        "Carbohydrates (g)",
        min_value=0,
        max_value=500,
        value=250
    )

    fat = st.number_input(
        "Fat (g)",
        min_value=0,
        max_value=200,
        value=60
    )

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Weight Change"):

    # Note: Double-check that this column order matches your training dataset precisely
    input_df = pd.DataFrame({
        "Stress_Level": [stress],
        "Calories_Consumed": [calories],
        "Protein_g": [protein],
        "Carbs_g": [carbs],
        "Fat_g": [fat]
    })

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    weight_change = prediction[0]

    if weight_change >= 0:
        color = "green"
        result = f"⬆️ Expected Weight Gain: +{weight_change:.2f} kg"
    else:
        color = "red"
        result = f"⬇️ Expected Weight Loss: {weight_change:.2f} kg"

    html = f"""
    <div class="card">
        <h2 style="text-align:center;color:#1565C0;margin-top:0;">
            Prediction Result
        </h2>
        <h1 style="text-align:center;color:{color};margin:20px 0;">
            {result}
        </h1>
        <p style="text-align:center;color:#555;">
            Prediction based on stress level, calories, protein, carbohydrates, and fat intake.
        </p>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)
    st.balloons()
