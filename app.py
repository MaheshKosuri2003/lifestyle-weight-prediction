import streamlit as st
import pandas as pd
import pickle
import os

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Lifestyle Weight Tracker Predictor",
    layout="wide",
    page_icon="🏋️‍♂️"
)

# 2. Inject Custom CSS[cite: 4]
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)[cite: 4]
    else:
        # Fallback basic styling if CSS file is missing
        st.markdown(
            """
            <style>
            .reportview-container { background: #f0f2f6; }
            .main-title { color: #2e4053; font-weight: bold; text-align: center; margin-bottom: 20px; }
            </style>
            """, 
            unsafe_allow_html=True
        )

local_css("css/style.css")[cite: 4]

# Title
st.markdown("<h1 class='main-title'>🏋️‍♂️ Lifestyle Weight Tracker</h1>", unsafe_allow_html=True)
st.write("Predict your projected weight change based on your daily lifestyle metrics.")

# 3. Load Serialized Core ML Assets
@st.cache_resource
def load_ml_assets():
    try:
        with open("scaler1.pkl", "rb") as f:
            scaler = pickle.load(f)[cite: 5]
        with open("best_model1.pkl", "rb") as f:
            model = pickle.load(f)[cite: 6]
        return scaler, model
    except FileNotFoundError as e:
        st.error(f"Missing essential model file: {e.filename}. Please ensure both 'scaler1.pkl' and 'best_model1.pkl' are in the root directory.")
        return None, None

scaler, model = load_ml_assets()

# 4. Input User Interface Form
if scaler and model:
    st.subheader("Enter Your Daily Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prevents negative values inside the interface to fix dataset anomalies[cite: 4, 7]
        calories = st.number_input("Calories Consumed (kcal)", min_value=0, max_value=10000, value=2095, step=50) 
        protein = st.number_input("Protein Intake (g)", min_value=0, max_value=500, value=90, step=5)
        carbs = st.number_input("Carbohydrates Intake (g)", min_value=0, max_value=1000, value=250, step=5)
        
    with col2:
        fat = st.number_input("Fat Intake (g)", min_value=0, max_value=300, value=70, step=5)
        stress_level = st.slider("Stress Level (1 = Low, 10 = High)", min_value=1, max_value=10, value=5, step=1)[cite: 5, 7]

    # Predict Button Execution Trigger
    if st.button("Predict Weight Change", type="primary"):
        # Explicit features vector generation matching scaler training parameters[cite: 4, 5]
        input_data = pd.DataFrame([{
            'Stress_Level': stress_level,[cite: 4, 5]
            'Calories_Consumed': calories,[cite: 4, 5]
            'Protein_g': protein,[cite: 4, 5]
            'Carbs_g': carbs,[cite: 4, 5]
            'Fat_g': fat[cite: 4, 5]
        }])
        
        # Ensure sequential feature column alignment matching feature_names_in_
        input_data = input_data[['Stress_Level', 'Calories_Consumed', 'Protein_g', 'Carbs_g', 'Fat_g']][cite: 5]
        
        try:
            # Scale inputs and pass to the Linear Regression model[cite: 5, 6]
            scaled_inputs = scaler.transform(input_data)[cite: 5]
            prediction = model.predict(scaled_inputs)[cite: 6]
            
            # Display Prediction
            weight_change = prediction[0]
            st.markdown("---")
            st.subheader("Prediction Result")
            if weight_change >= 0:
                st.success(f"Predicted Weight Gain: **+{weight_change:.3f} kg** over this tracking frame.")
            else:
                st.info(f"Predicted Weight Loss: **{weight_change:.3f} kg** over this tracking frame.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction processing: {e}")
