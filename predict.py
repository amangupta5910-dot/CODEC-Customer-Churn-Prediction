import joblib
import pandas as pd

# Load the trained model
model = joblib.load("churn_model.pkl")

print("===== Customer Churn Prediction =====")

# Take input from user
gender = int(input("Gender (Female=0, Male=1): "))
SeniorCitizen = int(input("Senior Citizen (0/1): "))
Partner = int(input("Partner (No=0, Yes=1): "))
Dependents = int(input("Dependents (No=0, Yes=1): "))
tenure = int(input("Tenure (months): "))
PhoneService = int(input("Phone Service (No=0, Yes=1): "))
MultipleLines = int(input("Multiple Lines (No=0, Yes=1): "))
InternetService = int(input("Internet Service (0=DSL, 1=Fiber, 2=No): "))
OnlineSecurity = int(input("Online Security (No=0, Yes=1): "))
OnlineBackup = int(input("Online Backup (No=0, Yes=1): "))
DeviceProtection = int(input("Device Protection (No=0, Yes=1): "))
TechSupport = int(input("Tech Support (No=0, Yes=1): "))
StreamingTV = int(input("Streaming TV (No=0, Yes=1): "))
StreamingMovies = int(input("Streaming Movies (No=0, Yes=1): "))
Contract = int(input("Contract (0=Month-to-month, 1=One year, 2=Two year): "))
PaperlessBilling = int(input("Paperless Billing (No=0, Yes=1): "))
PaymentMethod = int(input("Payment Method (0-3): "))
MonthlyCharges = float(input("Monthly Charges: "))
TotalCharges = float(input("Total Charges: "))

# Create DataFrame
new_customer = pd.DataFrame([[ 
    gender, SeniorCitizen, Partner, Dependents,
    tenure, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup,
    DeviceProtection, TechSupport, StreamingTV,
    StreamingMovies, Contract, PaperlessBilling,
    PaymentMethod, MonthlyCharges, TotalCharges
]], columns=[
    'gender','SeniorCitizen','Partner','Dependents',
    'tenure','PhoneService','MultipleLines',
    'InternetService','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV',
    'StreamingMovies','Contract','PaperlessBilling',
    'PaymentMethod','MonthlyCharges','TotalCharges'
])

# Predict
prediction = model.predict(new_customer)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("Customer is likely to CHURN.")
else:
    print("Customer is NOT likely to CHURN.")