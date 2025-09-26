import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Car Insurance Claim Predictor", layout="centered")

st.title("Car Insurance Claim Predictor")
st.write("Predict whether a client is likely to make a car insurance claim.\n\nThis app will try to load `insurance_model.pkl` from the app folder. If it is missing but `car_insurance.csv` is present, the app will train a quick Decision Tree model automatically.")

MODEL_PATH = "insurance_model.pkl"
CSV_PATH = "car_insurance.csv"

# Mappings (same as your notebook)
mapping = {
    "16-25":0, "26-39":1, "40-64":2, "65+":3,
    "Female":0, "Male":1,
    "0-9":0, "10-19":1, "20-29":2, "30+":3,
    "No education":0, "High school":1, "University":2,
    "Poverty":0, "Working class":1, "Middle class":2, "Upper class":3,
    "Does not own":0, "Owns":1,
    "Before 2015":0, "2015 or later":1,
    "No":0, "Yes":1,
    "Sedan":0, "Sports car":1
}

# Helper: try to load model, otherwise train if CSV exists
@st.cache_data
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model, "loaded"
        except Exception as e:
            return None, f"error_loading:{e}"
    # try to train from CSV
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            # basic preprocessing similar to notebook
            df = df.copy()
            # map where necessary (assume columns already numeric where appropriate)
            # fill na for credit_score and annual_mileage like notebook
            if 'credit_score' in df.columns and 'income' in df.columns:
                df['credit_score'] = df.groupby('income')['credit_score'].transform(lambda x: x.fillna(x.mean()))
            if 'annual_mileage' in df.columns and 'vehicle_type' in df.columns:
                df['annual_mileage'] = df.groupby('vehicle_type')['annual_mileage'].transform(lambda x: x.fillna(x.mean()))

            X = df.drop(columns=[c for c in ['id','outcome','postal_code'] if c in df.columns], errors='ignore')
            if 'outcome' in df.columns:
                y = df['outcome']
            else:
                return None, 'no_outcome_column'

            # If there are non-numeric columns, try to drop some columns to match sample app
            drop_cols = [c for c in ['education','vehicle_type','married','duis','children','gender'] if c in X.columns]
            X = X.drop(columns=drop_cols, errors='ignore')

            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))
            ])
            pipeline.fit(X, y)
            # save model
            try:
                joblib.dump(pipeline, MODEL_PATH)
            except Exception:
                pass
            return pipeline, 'trained'
        except Exception as e:
            return None, f'train_error:{e}'

    return None, 'no_model_no_data'

model, status = load_or_train_model()

if status == 'loaded':
    st.success("Model loaded from insurance_model.pkl")
elif status == 'trained':
    st.success("Trained a fresh model from car_insurance.csv (saved to insurance_model.pkl)")
elif status.startswith('error') or status.startswith('train_error'):
    st.error(f"Model error: {status}")
else:
    st.info("No model found and no dataset available. Upload `insurance_model.pkl` or `car_insurance.csv` to the app folder.")

st.sidebar.header("Client inputs")
age = st.sidebar.selectbox("Age group", ["16-25", "26-39", "40-64", "65+"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"]) 
driving_exp = st.sidebar.selectbox("Driving Experience", ["0-9", "10-19", "20-29", "30+"])
education = st.sidebar.selectbox("Education", ["No education", "High school", "University"]) 
income = st.sidebar.selectbox("Income", ["Poverty", "Working class", "Middle class", "Upper class"]) 
credit_score = st.sidebar.slider("Credit Score", 0.0, 1.0, 0.5)
vehicle_ownership = st.sidebar.radio("Vehicle Ownership", ["Does not own", "Owns"]) 
vehicle_year = st.sidebar.radio("Vehicle Year", ["Before 2015", "2015 or later"]) 
married = st.sidebar.radio("Married?", ["No", "Yes"]) 
children = st.sidebar.number_input("Number of Children", 0, 10, 0)
annual_mileage = st.sidebar.number_input("Annual Mileage", 0, 50000, 10000)
vehicle_type = st.sidebar.radio("Vehicle Type", ["Sedan", "Sports car"]) 
speeding = st.sidebar.number_input("Speeding Violations", 0, 20, 0)
duis = st.sidebar.number_input("DUIs", 0, 10, 0)
accidents = st.sidebar.number_input("Past Accidents", 0, 10, 0)

input_vector = np.array([[
    mapping[age], mapping[gender], mapping[driving_exp], mapping[education], mapping[income], credit_score,
    mapping[vehicle_ownership], mapping[vehicle_year], mapping[married], children, 0, annual_mileage,
    mapping[vehicle_type], speeding, duis, accidents
]])

st.write("### Preview inputs")
st.write({
    'age': age, 'gender': gender, 'driving_exp': driving_exp, 'education': education,
    'income': income, 'credit_score': credit_score, 'vehicle_ownership': vehicle_ownership,
    'vehicle_year': vehicle_year, 'married': married, 'children': children,
    'annual_mileage': annual_mileage, 'vehicle_type': vehicle_type,
    'speeding': speeding, 'duis': duis, 'accidents': accidents
})

if st.button("Predict"):
    if model is None:
        st.error("No model available. Upload `insurance_model.pkl` or `car_insurance.csv` into the app folder and reload.")
    else:
        try:
            pred = model.predict(input_vector)[0]
            proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_vector)[0].tolist()
            if pred == 1:
                st.error("This client is likely to make a claim!")
            else:
                st.success("This client is unlikely to make a claim.")
            if proba is not None:
                st.write(f"Prediction probabilities: {proba}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.write("If you want me to deploy this to Streamlit Cloud and provide an iframe embed for your site, either upload the `insurance_model.pkl` into this folder and push to a GitHub repo, or send me the file and I can prepare the repo for you.")
