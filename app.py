import streamlit as st
import pandas as pd
import joblib
model = joblib.load("logistic_regression_model.pkl")
scalar = joblib.load("scaler.pkl")
expected_columns = joblib.load("model_columns.pkl")
st.title("Heart Disease Prediction App")
st.markdown("Enter the following details to predict the likelihood of heart disease:")
age = st.slider("Age", min_value=18, max_value=120, value=40)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"])
rest_ecg = st.selectbox("Resting Electrocardiographic Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])
if st.button("Predict"):
    input_data = {
        "age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol, 
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "FastingBS": fasting_bs,
        "Sex_"+sex: 1,
        "ChestPainType_"+cp: 1,
        "RestingECG_"+rest_ecg: 1,
        "ExerciseAngina_"+exercise_angina: 1,
        "ST_Slope_"+st_slope: 1
    }
    input_df = pd.DataFrame([input_data])   
    for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
    input_df = input_df[expected_columns]
    input_scaled = scalar.transform(input_df)
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("The model predicts that you are likely to have heart disease. Please consult a healthcare professional for further evaluation.")
    else:
            st.success("The model predicts that you are unlikely to have heart disease. However, it's always a good idea to maintain a healthy lifestyle and consult a healthcare professional for regular check-ups.")


