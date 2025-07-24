import streamlit as st
import joblib
import pandas as pd

# ================================
# Load Preprocessor and Model
# ================================
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("heart_disease_model.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# ================================
# Helper Mappings
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
# Sidebar for Info
# ================================
st.sidebar.title("ℹ️ About")
st.sidebar.markdown(
    """
    **Heart Disease Risk Predictor**  
    - Based on clinical heart disease dataset (303 patients)  
    - Uses machine learning to assess risk  
    - For educational use only (not medical advice)  
    """
)

st.sidebar.markdown("### Tips for Use:")
st.sidebar.markdown(
    """
    - Enter your vitals and symptoms  
    - Click **Predict Risk** to see the result  
    - Consult a doctor for any concerns
    """
)

# ================================
# Main Header
# ================================
st.title("❤️ Heart Disease Risk Prediction")
st.markdown(
    """
    ### Fill in your health details  
    Our AI tool will predict your risk of heart disease.  
    **Note:** This tool is not a substitute for medical advice.
    """
)

# ================================
# Input Form
# ================================
st.header("Patient Information")

with st.form("patient_form"):
    st.subheader("🧍 Personal Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=20, max_value=90, value=40)
        sex = st.selectbox("Sex", list(sex_map.keys()))
    with col2:
        cp = st.selectbox("Chest Pain Type", list(cp_map.keys()))
        thal = st.selectbox("Thalassemia Type", thal_options)

    st.subheader("🩺 Vitals & Measurements")
    col3, col4 = st.columns(2)
    with col3:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=700, value=200)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
    with col4:
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)

    st.subheader("⚠️ Risk Indicators")
    col5, col6 = st.columns(2)
    with col5:
        fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl?", list(fbs_map.keys()))
        exang = st.selectbox("Exercise Induced Angina?", list(exang_map.keys()))
    with col6:
        restecg = st.selectbox("Resting ECG Results", list(restecg_map.keys()))
        slope = st.selectbox("Slope of ST Segment", list(slope_map.keys()))

    submitted = st.form_submit_button("🔍 Predict Risk")

# ================================
# Prediction
# ================================
if submitted:
    # Map inputs
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
        "thal": thal
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[preprocessor.feature_names_in_]

    # Predict
    processed = preprocessor.transform(input_df)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1] * 100  # Risk percentage

    # Result
    st.header("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease!**\nEstimated risk: **{probability:.1f}%**")
    else:
        st.success(f"✅ **No Significant Risk Detected.**\nEstimated risk: **{probability:.1f}%**")

    # Show input summary
    st.write("### Your Input Data:")
    st.dataframe(input_df)
