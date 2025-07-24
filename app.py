import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ================================
# Load Preprocessor and Model
# ================================
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("heart_disease_model.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ================================
# Helper Mappings (for user-friendly inputs)
# ================================
cp_map = {
    "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-anginal Pain": 3,
    "Asymptomatic": 4
}

restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

slope_map = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3
}

sex_map = {"Male": 1, "Female": 0}
fbs_map = {"Yes (>120 mg/dl)": 1, "No": 0}
exang_map = {"Yes": 1, "No": 0}

thal_options = ["normal", "fixed", "reversible"]

# ================================
# App Header
# ================================
st.title("❤️ Heart Disease Risk Prediction")
st.markdown(
    """
    ### Check your heart health risk instantly  
    Fill in the details below, and this AI-powered tool will predict whether you may be at risk of heart disease.
    **Disclaimer:** This is for educational purposes only and not a medical diagnosis.
    """
)

# ================================
# Input Form
# ================================
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 90, 40)
    sex = st.selectbox("Sex", list(sex_map.keys()))
    cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl?", list(fbs_map.keys()))
    
with col2:
    restecg = st.selectbox("Resting ECG Results", list(restecg_map.keys()))
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina?", list(exang_map.keys()))
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
    slope = st.selectbox("Slope of ST Segment", list(slope_map.keys()))
    ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.selectbox("Thalassemia Type", thal_options)

# ================================
# Prediction
# ================================
if st.button("Predict Risk"):
    # Prepare input (map user choices to model format)
    input_dict = {
        "age": age,
        "sex": sex_map[sex],
        "cp": cp_map[cp],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_map[fbs],
        "restecg": restecg_map[restecg],
        "thalach": thalach,
        "exang": exang_map[exang],
        "oldpeak": oldpeak,
        "slope": slope_map[slope],
        "ca": ca,
        "thal": thal  # stays as string (OneHotEncoded)
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[preprocessor.feature_names_in_]  # match column order

    # Transform & Predict
    processed = preprocessor.transform(input_df)
    prediction = model.predict(processed)[0]

    # Display result
    st.subheader("Result:")
    if prediction == 1:
        st.error("⚠️ **High Risk of Heart Disease!** Please consult a doctor.")
    else:
        st.success("✅ **No Significant Risk Detected.** Keep up a healthy lifestyle!")

    # Show input summary
    st.write("### Your Input Data:")
    st.dataframe(input_df)
