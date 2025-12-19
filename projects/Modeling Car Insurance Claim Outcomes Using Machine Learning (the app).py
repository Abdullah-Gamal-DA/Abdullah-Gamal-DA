# =============================
# Car Insurance Claim Prediction - Full Version with Streamlit
# =============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import joblib
import streamlit as st

# =============================
# 1) Load dataset and preprocess
# =============================
car = pd.read_csv('car_insurance.csv')

# Map categorical values to numbers
age_map = {'16-25':0, '26-39':1, '40-64':2, '65+':3}
driving_exp_map = {'0-9y':0, '10-19y':1, '20-29y':2, '30y+':3}
education_map = {'none':0, 'high school':1, 'university':2}
income_map = {'poverty':0, 'working class':1, 'middle class':2, 'upper class':3}
vehicle_year_map = {'before 2015':0, 'after 2015':1}
vehicle_type_map = {'sedan':0, 'sports car':1}
yes_no_map = {'No':0, 'Yes':1, 'Does not own':0, 'Owns':1}

car['age'] = car['age'].map(age_map)
car['driving_experience'] = car['driving_experience'].map(driving_exp_map)
car['education'] = car['education'].map(education_map)
car['income'] = car['income'].map(income_map)
car['vehicle_year'] = car['vehicle_year'].map(vehicle_year_map)
car['vehicle_type'] = car['vehicle_type'].map(vehicle_type_map)
car['vehicle_ownership'] = car['vehicle_ownership'].map(yes_no_map)
car['married'] = car['married'].map(yes_no_map)

# Fill missing values
car['credit_score'] = car.groupby('income')['credit_score'].transform(lambda x: x.fillna(x.mean()))
car['annual_mileage'] = car.groupby('vehicle_type')['annual_mileage'].transform(lambda x: x.fillna(x.mean()))

# Features and target
features = [
    'age','driving_experience','income','credit_score',
    'vehicle_ownership','vehicle_year','annual_mileage',
    'speeding_violations','past_accidents'
]
X = car[features].values
y = car['outcome'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# =============================
# 2) Train model
# =============================
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=5, random_state=42)
tree.fit(X_train, y_train)

# Save model
joblib.dump(tree, "insurance_model.pkl")

# =============================
# 3) Streamlit UI
# =============================
st.title("Car Insurance Claim Predictor")
st.write("Enter client details to predict the likelihood of an insurance claim.")

# User inputs
age_input = st.selectbox("Age group", ["16-25","26-39","40-64","65+"])
gender_input = st.selectbox("Gender", ["Female", "Male"])  # not used in model
driving_exp_input = st.selectbox("Driving Experience", ["0-9y","10-19y","20-29y","30y+"])
education_input = st.selectbox("Education", ["none","high school","university"])
income_input = st.selectbox("Income", ["poverty","working class","middle class","upper class"])
credit_score_input = st.slider("Credit Score (0-1)", 0.0, 1.0, 0.5)
vehicle_ownership_input = st.radio("Vehicle Ownership", ["Does not own","Owns"])
vehicle_year_input = st.radio("Vehicle Year", ["before 2015","after 2015"])
married_input = st.radio("Married?", ["No","Yes"])  # not used
children_input = st.number_input("Number of Children", 0, 10, 0)  # not used
annual_mileage_input = st.number_input("Annual Mileage", 0, 50000, 10000)
vehicle_type_input = st.radio("Vehicle Type", ["sedan","sports car"])  # not used
speeding_violations_input = st.number_input("Speeding Violations", 0, 20, 0)
duis_input = st.number_input("DUIs", 0, 10, 0)  # not used
past_accidents_input = st.number_input("Past Accidents", 0, 10, 0)

# Map inputs to numeric values
age = age_map[age_input]
driving_experience = driving_exp_map[driving_exp_input]
education = education_map[education_input]
income = income_map[income_input]
vehicle_year = vehicle_year_map[vehicle_year_input]
vehicle_ownership = yes_no_map[vehicle_ownership_input]

# Prepare input DataFrame
input_df = pd.DataFrame([[
    age, driving_experience, income, credit_score_input,
    vehicle_ownership, vehicle_year, annual_mileage_input,
    speeding_violations_input, past_accidents_input
]], columns=features)

# Load model
model = joblib.load("insurance_model.pkl")

# Predict
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.error("This client is likely to make a claim!")
    else:
        st.success("This client is unlikely to make a claim.")
