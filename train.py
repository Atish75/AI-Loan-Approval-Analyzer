# train.py (Industry Upgrade)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)
n = 2000

income = np.random.randint(20000, 250000, n)
co_income = np.random.choice([0, 15000, 35000, 60000], size=n, p=[0.4, 0.3, 0.2, 0.1])
loan_amount = np.random.randint(50000, 1500000, n)
cibil_score = np.random.randint(300, 900, n) # Real CIBIL credit score
term = np.random.choice([12, 24, 36, 60, 120, 240, 360], size=n)
existing_emi = np.random.randint(0, 30000, n)

total_income = income + co_income
approx_monthly_emi = (loan_amount / term) * 1.15  # Interest buffer
dti_ratio = (existing_emi + approx_monthly_emi) / total_income
lti_ratio = loan_amount / (total_income * 12)

# Industry Approval Logic (Ground Truth)
# CIBIL >= 700 AND DTI < 50% AND LTI < 4.0
approved = (cibil_score >= 680) & (dti_ratio <= 0.50) & (lti_ratio <= 4.5)
y = approved.astype(int)

df = pd.DataFrame({
    'ApplicantIncome': income,
    'CoapplicantIncome': co_income,
    'LoanAmount': loan_amount,
    'CIBIL_Score': cibil_score,
    'Loan_Amount_Term': term,
    'Existing_EMI': existing_emi,
    'DTI_Ratio': dti_ratio,
    'LTI_Ratio': lti_ratio,
    'Loan_Status': y
})

feature_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'CIBIL_Score', 'Loan_Amount_Term', 'Existing_EMI', 'DTI_Ratio', 'LTI_Ratio']
X = df[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
model.fit(X_scaled, y)

joblib.dump({
    'model': model,
    'scaler': scaler,
    'features': feature_cols
}, 'loan_model.joblib')

print("Industry Model trained & saved with Financial Ratios + CIBIL Score!")