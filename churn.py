import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# Load Dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

print("First 5 Rows:")
print(df.head())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")


# Remove Missing Values
df.dropna(inplace=True)

# Remove customerID column
df.drop("customerID", axis=1, inplace=True)

# Convert Categorical Columns
categorical_columns = df.select_dtypes(include=["object"]).columns

for col in categorical_columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100)

print("\nData Types:\n")
print(X.dtypes)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Save the trained model
joblib.dump(model, "churn_model.pkl")
print("\nModel saved successfully as churn_model.pkl")

# Churn Distribution Graph
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")



# 2nd Task: Data Visualization (EDA)

# ===========================
# DATA VISUALIZATION (EDA)
# ===========================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Customer Churn Analysis", fontsize=16, fontweight="bold")

# 1. Churn Distribution
sns.countplot(x="Churn", data=df, ax=axes[0,0])
axes[0,0].set_title("1. Customer Churn Distribution")
axes[0,0].set_xticklabels(["No","Yes"])

# 2. Gender vs Churn
sns.countplot(x="gender", hue="Churn", data=df, ax=axes[0,1])
axes[0,1].set_title("2. Gender vs Churn")
axes[0,1].set_xticklabels(["Female","Male"])

# 3. Contract Type vs Churn
sns.countplot(x="Contract", hue="Churn", data=df, ax=axes[1,0])
axes[1,0].set_title("3. Contract Type vs Churn")

# 4. Tenure Distribution
sns.histplot(df["tenure"], bins=30, kde=True, ax=axes[1,1], color="green")
axes[1,1].set_title("4. Customer Tenure Distribution")

plt.tight_layout()
plt.show()