Car Insurance Claim Predictor - Streamlit app

How to run locally

1. Install requirements:

   python -m pip install -r requirements.txt

2. Place either `insurance_model.pkl` (trained pipeline saved with joblib) or the dataset `car_insurance.csv` in this folder.

3. Run:

   streamlit run app.py

Deploying to Streamlit Cloud

1. Push the `streamlit_apps/car_insurance` folder to a GitHub repository.
2. On Streamlit Cloud, create a new app and point it to the repository and folder.
3. Add `insurance_model.pkl` to the repo root or upload it after deployment.

If you want I can prepare a GitHub repo for you and deploy to Streamlit Cloud, but I need either the `insurance_model.pkl` file or the dataset to be present in the repo.
