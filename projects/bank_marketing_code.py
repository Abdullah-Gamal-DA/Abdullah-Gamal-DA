# ============================================================
# Data Analysis Project: Bank Marketing Campaign
# ============================================================
# Objective:
# Analyze bank marketing campaign data to understand client behavior,
# clean the dataset, perform exploratory data analysis, and model campaign outcomes.
# The cleaned data will be split into three CSV files for future PostgreSQL import.
# ============================================================

# ---------------------------
# Step 1: Import necessary libraries
# ---------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.cluster import KMeans

# ---------------------------
# Step 2: Load raw dataset
# ---------------------------
df = pd.read_csv("bank_marketing.csv")

# Inspect the dataset to understand structure and data types
print(df.info())
print(df.describe())

# Check unique values in boolean-like columns
for col in ["credit_default", "mortgage", "previous_outcome", "campaign_outcome"]:
    print(col)
    print("--------------")
    print(df[col].value_counts())

# ---------------------------
# Step 3: Prepare `client.csv`
# ---------------------------
# Select relevant columns
client = df[['client_id','age','job','marital','education','credit_default','mortgage']]

# Data cleaning
client['job'] = client['job'].replace('.', '_')  # Replace '.' with '_'
client['education'] = client['education'].replace('.', '_')
client['education'] = client['education'].replace('unknown', np.nan)
client['credit_default'] = client['credit_default'].replace(['yes','unknown','no'], [1,0,0])
client['mortgage'] = client['mortgage'].replace(['yes','unknown','no'], [1,0,0])

# Convert to proper data types
client_dtypes = {
    'client_id':'int',
    'age':'int',
    'job':'O',
    'marital':'O',
    'education':'O',
    'credit_default':'bool',
    'mortgage':'bool'
}
client = client.astype(client_dtypes)

# Save cleaned client data
client.to_csv('client.csv', index=False)

# ---------------------------
# Step 4: Prepare `campaign.csv`
# ---------------------------
# Create 'year' column and last contact date
df['year'] = 2022
df['last_contact_date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str))

# Select relevant columns
campaign = df[['client_id','number_contacts','contact_duration','previous_campaign_contacts','previous_outcome','campaign_outcome','last_contact_date']]

# Clean boolean columns
campaign['previous_outcome'] = campaign['previous_outcome'].replace(['success','failure','nonexistent'], [1,0,0])
campaign['campaign_outcome'] = campaign['campaign_outcome'].replace(['yes','no'], [1,0])

# Convert to proper data types
campaign_dtypes = {
    'client_id':'int',
    'number_contacts':'int',
    'contact_duration':'int',
    'previous_campaign_contacts':'int',
    'previous_outcome':'bool',
    'campaign_outcome':'bool'
}
campaign = campaign.astype(campaign_dtypes)

# Save cleaned campaign data
campaign.to_csv('campaign.csv', index=False)

# ---------------------------
# Step 5: Prepare `economics.csv`
# ---------------------------
economics = df[['client_id','cons_price_idx','euribor_three_months']]
economics.to_csv('economics.csv', index=False)

# ---------------------------
# Step 6: Merge datasets for analysis
# ---------------------------
df = client.merge(campaign, on='client_id', how='outer').merge(economics, on='client_id', how='outer')
df.info()

# Convert categorical variables
df[['job','marital','education']] = df[['job','marital','education']].astype('category')
df['marital'] = df['marital'].cat.reorder_categories(['unknown', 'single', 'married', 'divorced'], ordered=True)
df['education'] = df['education'].astype(str).fillna('nan').astype('category')
df['education'] = df['education'].cat.reorder_categories([
    'nan','illiterate','basic.4y', 'basic.6y', 'basic.9y', 'high.school',
    'professional.course','university.degree'], ordered=True)

# Convert contact duration to minutes
df['contact_duration'] = df['contact_duration'] / 60

# ---------------------------
# Step 7: Exploratory Data Analysis
# ---------------------------

# Age distribution
sns.boxenplot(x=df['age'])
plt.show()

# Age categories
df['age_category'] = pd.cut(df['age'], bins=[0,30,45,60,120], labels=['18-29','30-44','45-59','60+'])
sns.countplot(x='age_category', data=df, order=df['age_category'].value_counts().index)
plt.show()

# Education distribution
sns.countplot(x='education', data=df, order=['university.degree', 'high.school', 'basic.9y',
                  'professional.course', 'basic.4y', 'basic.6y'])
plt.show()

# Job distribution
sns.countplot(x='job', data=df, order=df['job'].value_counts().index)
plt.show()

# Outcome trends vs Euribor
fig, ax1 = plt.subplots(figsize=(18,6))
sns.lineplot(x='month', y='previous_outcome', data=df, ax=ax1, label='Previous Outcome')
sns.lineplot(x='month', y='campaign_outcome', data=df, ax=ax1, label='Campaign Outcome', color='r')
ax2 = ax1.twinx()
sns.lineplot(x='month', y='euribor_three_months', data=df, ax=ax2, label='Euribor 3M', color='g', legend=False)
ax1.set_xlabel("Month")
ax1.set_ylabel("Outcomes")
ax2.set_ylabel("Euribor 3M Rate")
plt.tight_layout()
plt.show()

# ---------------------------
# Step 8: Correlation Analysis
# ---------------------------
sns.heatmap(round(df[['campaign_outcome','contact_duration','number_contacts']].corr(),2), annot=True)
statistic, p_value = pearsonr(df['campaign_outcome'], df['contact_duration'])
print(f'Pearson correlation: {statistic}, p-value: {p_value}')


# ============================================================
# End of documentation-style notebook
# ============================================================
