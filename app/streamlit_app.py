import streamlit as st
import pandas as pd
import numpy as np
import joblib

# load trained model
model = joblib.load('app/best_model.pkl')

st.title("StressSignals: Mental Health Risk Predictor")

st.write("""
Enter your details below to predict whether you may need mental health treatment.
""")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)

gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
family_history = st.selectbox("Family history of mental illness?", options=["Yes", "No"])
work_interfere = st.selectbox("Level of Work Interference due to Mental Health", options=["Never", "Rarely", "Sometimes", "Often"])

company_size = st.selectbox("Company Size", options=[
    "1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"
])

remote_work = st.selectbox("Do you work remotely?", options=["Yes", "No"])
benefits = st.selectbox("Employer provides mental health benefits?", options=["Yes", "No"])
care_options = st.selectbox("Availability of mental health care options?", options=["Not sure", "No", "Yes"])
wellness_program = st.selectbox("Wellness program available?", options=["No", "Don't know", "Yes"])
seek_help = st.selectbox("Anonymity encouraged for seeking help?", options=["No", "Don't know", "Yes"])
anonymity = st.selectbox("Anonymity in seeking mental health treatment?", options=["No", "Don't know", "Yes"])
leave = st.selectbox("Ease of taking leave for mental health?", options=["Very difficult", "Somewhat difficult", "Don't know", "Somewhat easy", "Very easy"])
mental_health_consequence = st.selectbox("Fear of negative mental health consequence at work?", options=["No", "Maybe", "Yes"])
phys_health_consequence = st.selectbox("Fear of physical health consequence at work?", options=["No", "Maybe", "Yes"])
coworkers = st.selectbox("Comfortable discussing mental health with coworkers?", options=["No", "Some of them", "Yes"])
supervisor = st.selectbox("Comfortable discussing with supervisor?", options=["No", "Some of them", "Yes"])
mental_health_interview = st.selectbox("Would mental health issues affect interview decision?", options=["No", "Maybe", "Yes"])
phys_health_interview = st.selectbox("Would physical health issues affect interview decision?", options=["No", "Maybe", "Yes"])
mental_vs_physical = st.selectbox("Does employer value mental health like physical health?", options=["Don't know", "No", "Yes"])
obs_consequence = st.selectbox("Observed negative consequences for colleagues with mental health issues?", options=["No", "Yes"])

# Data Preprocessing
def preprocess(age, gender, family_history, work_interfere, company_size, remote_work, benefits, care_options,
               wellness_program, seek_help, anonymity, leave, mental_health_consequence, phys_health_consequence,
               coworkers, supervisor, mental_health_interview, phys_health_interview, mental_vs_physical, obs_consequence):

    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    yes_no_map = {"Yes": 1, "No": 0}
    work_interfere_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    company_map = {"1-5": 0, "6-25": 4, "26-100": 2, "100-500": 1, "500-1000": 3, "More than 1000": 5}
    care_map = {"Not sure": 0, "No": 1, "Yes": 2}
    wellness_map = {"No": 0, "Don't know": 1, "Yes": 2}
    seek_map = {"No": 0, "Don't know": 1, "Yes": 2}
    anonymity_map = {"No": 0, "Don't know": 1, "Yes": 2}
    leave_map = {"Very difficult": 0, "Somewhat difficult": 1, "Don't know": 2, "Somewhat easy": 3, "Very easy": 4}
    consequence_map = {"No": 0, "Maybe": 1, "Yes": 2}
    coworkers_map = {"No": 0, "Some of them": 1, "Yes": 2}
    supervisor_map = {"No": 0, "Some of them": 1, "Yes": 2}
    interview_map = {"No": 0, "Maybe": 1, "Yes": 2}
    mental_vs_physical_map = {"Don't know": 0, "No": 1, "Yes": 2}
    obs_map = {"No": 0, "Yes": 1}

    data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_map[gender]],
        'family_history': [yes_no_map[family_history]],
        'work_interfere': [work_interfere_map[work_interfere]],
        'no_employees': [company_map[company_size]],
        'remote_work': [yes_no_map[remote_work]],
        'benefits': [yes_no_map[benefits]],
        'care_options': [care_map[care_options]],
        'wellness_program': [wellness_map[wellness_program]],
        'seek_help': [seek_map[seek_help]],
        'anonymity': [anonymity_map[anonymity]],
        'leave': [leave_map[leave]],
        'mental_health_consequence': [consequence_map[mental_health_consequence]],
        'phys_health_consequence': [consequence_map[phys_health_consequence]],
        'coworkers': [coworkers_map[coworkers]],
        'supervisor': [supervisor_map[supervisor]],
        'mental_health_interview': [interview_map[mental_health_interview]],
        'phys_health_interview': [interview_map[phys_health_interview]],
        'mental_vs_physical': [mental_vs_physical_map[mental_vs_physical]],
        'obs_consequence': [obs_map[obs_consequence]]
    })

    return data

# Prediction
if st.button("Predict Mental Health Treatment Need"):
    input_df = preprocess(
        age, gender, family_history, work_interfere, company_size, remote_work, benefits,
        care_options, wellness_program, seek_help, anonymity, leave,
        mental_health_consequence, phys_health_consequence, coworkers, supervisor,
        mental_health_interview, phys_health_interview, mental_vs_physical, obs_consequence
    )

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Prediction: You may need mental health treatment.\nProbability: {proba:.2f}")
    else:
        st.info(f"Prediction: You may not need mental health treatment.\nProbability: {proba:.2f}")
