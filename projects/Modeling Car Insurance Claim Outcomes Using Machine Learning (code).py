# =============================
# Car Insurance Claim Prediction
# =============================

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# =============================
# 1) Load the dataset
# =============================
# - Dataset: car_insurance.csv
# - Goal: Predict whether a customer will file an insurance claim (binary outcome)
car = pd.read_csv('car_insurance.csv')
print("Dataset loaded. Shape:", car.shape)

# =============================
# 2) Encode categorical variables
# =============================
# - Map text categories into numerical values for ML models
# - Ensures models like Logistic Regression and Random Forest can process the data
car['driving_experience'] = car['driving_experience'].map(
    {'0-9y':0, '10-19y':1, '20-29y':2, '30y+':3})
car['education'] = car['education'].map(
    {'none':0, 'high school':1, 'university':2})
car['income'] = car['income'].map(
    {'poverty':0, 'working class':1, 'middle class':2, 'upper class':3})
car['vehicle_year'] = car['vehicle_year'].map(
    {'before 2015':0, 'after 2015':1})
car['vehicle_type'] = car['vehicle_type'].map(
    {'sedan':0, 'sports car':1})

# =============================
# 3) Handle missing values
# =============================
# - Fill missing values with group-wise means
car['credit_score'] = car.groupby('income')['credit_score'] \
                          .transform(lambda x: x.fillna(x.mean()))
car['annual_mileage'] = car.groupby('vehicle_type')['annual_mileage'] \
                            .transform(lambda x: x.fillna(x.mean()))

# =============================
# 4) Define feature set and target
# =============================
features = [
    'age','driving_experience','income','credit_score',
    'vehicle_ownership','vehicle_year',
    'annual_mileage','speeding_violations','past_accidents'
]
X = car[features].values      # Feature matrix
y = car['outcome'].values     # Target variable (claim or no claim)

# =============================
# 5) Split data into training and testing sets
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# =============================
# 6) Baseline model using Logistic Regression
# =============================
baseline = LogisticRegression(solver='liblinear')
baseline.fit(X_train, y_train)
print('Baseline Accuracy (Logistic Regression on all features):', 
      accuracy_score(y_test, baseline.predict(X_test)))

# =============================
# 7) Feature importance analysis using Random Forest
# =============================
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
best_feature = features[np.argmax(importances)]
print('Most Important Feature:', best_feature)

# =============================
# 8) Single-feature model using Decision Tree
# =============================
# - Train a simpler model using only the most important feature
# - Decision Tree provides interpretable predictions
X_single = car[[best_feature]].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X_single, y, test_size=0.3, stratify=y, random_state=42
)

single_model = DecisionTreeClassifier(random_state=42)
cv_score = cross_val_score(single_model, X_tr, y_tr, cv=5, scoring='accuracy')

single_model.fit(X_tr, y_tr)
y_pred = single_model.predict(X_te)

print('\nSingle Feature Decision Tree Model Results')
print('Cross-Validation Accuracy:', cv_score.mean())
print('Test Accuracy:', accuracy_score(y_te, y_pred))
print(classification_report(y_te, y_pred))

# =============================
# 9) Plot feature importance
# =============================
plt.figure(figsize=(7,4))
plt.bar(features, importances)
plt.xticks(rotation=45)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()

# =============================
# 10) Save the trained model
# =============================
joblib.dump(single_model, 'insurance_model.pkl')
print("Model saved as 'insurance_model.pkl'")
