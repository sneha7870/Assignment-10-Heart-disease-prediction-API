"""
train_model.py — Train and serialize a heart disease classification model.

Dataset: Heart Disease Prediction Dataset
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Usage:
    python train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------
# Task 1: Data Understanding and Preprocessing
# ---------------------------------------------------------------
df = pd.read_csv("heart.csv")
print("First five records:")
print(df.head())

print("\nDataset shape:", df.shape)

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "target"

print("\nNumerical features:", FEATURE_COLUMNS)
print("Target variable:", TARGET_COLUMN)

print("\nMissing values per column:")
print(df.isnull().sum())

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# ---------------------------------------------------------------
# Task 2: Model Development
# ---------------------------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))

# Save the trained model
joblib.dump(model, "model.pkl")
print("\nModel saved to model.pkl")
