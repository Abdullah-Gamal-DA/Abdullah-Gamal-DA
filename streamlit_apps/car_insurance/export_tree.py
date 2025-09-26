import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import json

# Load CSV
df = pd.read_csv('car_insurance.csv')

# Basic preprocessing similar to notebook
# Map string categories if present; try to detect
mappings = {
    'age': {'16-25':0, '26-39':1, '40-64':2, '65+':3},
    'gender': {'Female':0, 'Male':1},
    'driving_experience': {'0-9':0, '10-19':1, '20-29':2, '30+':3},
    'education': {'No education':0, 'High school':1, 'University':2},
    'income': {'Poverty':0, 'Working class':1, 'Middle class':2, 'Upper class':3},
    'vehicle_ownership': {'Does not own':0, 'Owns':1},
    'vehicle_year': {'Before 2015':0, '2015 or later':1},
    'married': {'No':0, 'Yes':1},
    'vehicle_type': {'Sedan':0, 'Sports car':1}
}

for col, mp in mappings.items():
    if col in df.columns and df[col].dtype == object:
        df[col] = df[col].map(mp).fillna(df[col])

# Fill missing credit_score and annual_mileage like earlier
if 'credit_score' in df.columns and 'income' in df.columns:
    df['credit_score'] = df.groupby('income')['credit_score'].transform(lambda x: x.fillna(x.mean()))
if 'annual_mileage' in df.columns and 'vehicle_type' in df.columns:
    df['annual_mileage'] = df.groupby('vehicle_type')['annual_mileage'].transform(lambda x: x.fillna(x.mean()))

# Drop unused columns if exist
drop_cols = [c for c in ['id','postal_code','education','vehicle_type','married','duis','children','gender'] if c in df.columns]
X = df.drop(columns=[c for c in drop_cols + ['outcome'] if c in df.columns], errors='ignore')
y = df['outcome']

# Fit pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('clf', DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42))
])

pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, 'insurance_model.pkl')

# Export tree structure to JSON (walk sklearn tree)
clf = pipeline.named_steps['clf']
feature_names = X.columns.tolist()

def node_to_dict(tree, node_id=0):
    if tree.feature[node_id] == -2:
        # leaf
        return {'type': 'leaf', 'value': tree.value[node_id].tolist()}
    feature = feature_names[tree.feature[node_id]]
    threshold = tree.threshold[node_id]
    left = node_to_dict(tree, tree.children_left[node_id])
    right = node_to_dict(tree, tree.children_right[node_id])
    return {'type': 'node', 'feature': feature, 'threshold': float(threshold), 'left': left, 'right': right}

tr = clf.tree_
root = node_to_dict(tr, 0)
with open('insurance_tree.json', 'w', encoding='utf-8') as f:
    json.dump({'feature_names': feature_names, 'tree': root}, f, ensure_ascii=False, indent=2)

print('Exported insurance_model.pkl and insurance_tree.json')
