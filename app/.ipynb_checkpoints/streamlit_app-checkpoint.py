import streamlit as st
import pandas as pd
import numpy as np
import joblib

# load the trained model
model = joblib.load('app/best_model.pkl')

st.title("StressSignals: Mental Health Risk Predictor")

st.write("""
Enter your details below to predict whether you might need mental health treatment.
""")

#  Input form

age = st.number_input("Age", min_value=18, max_value=100, value=30)

gender = st.selectbox("Gender", options=["Male", "Female", "Other"])

family_history = st.selectbox("Family history of mental illness?", options=["Yes", "No"])

remote_work = st.selectbox("Do you work remotely?", options=["Yes", "No"])

company_size = st.selectbox("Company Size", options=[
    "1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"
])

benefits = st.selectbox("Does your employer provide mental health benefits?", options=["Yes", "No"])

#data preprocessing

def preprocess_input(age, gender, family_history, remote_work, company_size, benefits):
    # Map inputs to numerical / encoded form
    
    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    family_map = {"Yes": 1, "No": 0}
    remote_map = {"Yes": 1, "No": 0}
    benefits_map = {"Yes": 1, "No": 0}
    company_map = {
        "1-5": 0,
        "6-25": 4,
        "26-100": 2,
        "100-500": 1,
        "500-1000": 3,
        "More than 1000": 5
    }
    
    # Create DataFrame with columns in the order your model expects
    data = pd.DataFrame({
        'age': [age],
        'gender': [gender_map[gender]],
        'family_history': [family_map[family_history]],
        'remote_work': [remote_map[remote_work]],
        'company_size': [company_map[company_size]],
        'benefits': [benefits_map[benefits]]
    })
    
    return data

input_df = preprocess_input(age, gender, family_history, remote_work, company_size, benefits)

if st.button("Predict Mental Health Treatment Need"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Prediction: You may need mental health treatment.\nProbability: {proba:.2f}")
    else:
        st.info(f"Prediction: You may not need mental health treatment.\nProbability: {proba:.2f}")
