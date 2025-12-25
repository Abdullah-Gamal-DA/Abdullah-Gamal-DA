# =============================
# Car Insurance Claim Predictor - Streamlit Only Using Saved Model
# =============================

import pandas as pd
import joblib
import streamlit as st

# =============================
# 1) Load saved model
# =============================
model_path = "insurance_model.pkl"
model = joblib.load(model_path)

# =============================
# 2) Define input mappings
# =============================
age_map = {'16-25':0, '26-39':1, '40-64':2, '65+':3}
driving_exp_map = {'0-9y':0, '10-19y':1, '20-29y':2, '30y+':3}
education_map = {'none':0, 'high school':1, 'university':2}
income_map = {'poverty':0, 'working class':1, 'middle class':2, 'upper class':3}
vehicle_year_map = {'before 2015':0, 'after 2015':1}
yes_no_map = {'No':0, 'Yes':1, 'Does not own':0, 'Owns':1}

# =============================
# 3) Streamlit UI
# =============================
st.title("Car Insurance Claim Predictor")
st.write("Enter client details to predict the likelihood of an insurance claim.")

# User inputs
age_input = st.selectbox("Age group", ["16-25","26-39","40-64","65+"])
driving_exp_input = st.selectbox("Driving Experience", ["0-9y","10-19y","20-29y","30y+"])
education_input = st.selectbox("Education", ["none","high school","university"])
income_input = st.selectbox("Income", ["poverty","working class","middle class","upper class"])
credit_score_input = st.slider("Credit Score (0-1)", 0.0, 1.0, 0.5)
vehicle_ownership_input = st.radio("Vehicle Ownership", ["Does not own","Owns"])
vehicle_year_input = st.radio("Vehicle Year", ["before 2015","after 2015"])
annual_mileage_input = st.number_input("Annual Mileage", 0, 50000, 10000)
speeding_violations_input = st.number_input("Speeding Violations", 0, 20, 0)
past_accidents_input = st.number_input("Past Accidents", 0, 10, 0)

# Prepare input DataFrame
input_df = pd.DataFrame([[
    age_map[age_input],
    driving_exp_map[driving_exp_input],
    income_map[income_input],
    credit_score_input,
    yes_no_map[vehicle_ownership_input],
    vehicle_year_map[vehicle_year_input],
    annual_mileage_input,
    speeding_violations_input,
    past_accidents_input
]], columns=[
    'age','driving_experience','income','credit_score',
    'vehicle_ownership','vehicle_year','annual_mileage',
    'speeding_violations','past_accidents'
])

# Predict
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.error("This client is likely to make a claim!")
    else:
        st.success("This client is unlikely to make a claim.")
