#!/usr/bin/env python
# coding: utf-8
"""Exploratory Data Analysis and model evaluation plots for financial fraud detection."""

from pathlib import Path

import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(r"C:\Users\dell\OneDrive\Desktop\PBeL\Synthetic_Financial_datasets_log.csv")

print(df.head(100000))
print(df.info)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nisFraud value counts:")
print(df["isFraud"].value_counts())

print("\nisFlaggedFraud value counts:")
print(df["isFlaggedFraud"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

fraud_pct = round(df["isFraud"].value_counts()[1] / df.shape[0] * 100, 4)
print(f"\nFraud percentage: {fraud_pct}%")

print("\nTransaction type counts:")
print(df["type"].value_counts())

# ---------------------------------------------------------------------------
# Initial visualizations
# ---------------------------------------------------------------------------
df["type"].value_counts().plot(kind="bar", color="green")
plt.title("Transaction Type Distribution")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.show()

fraud_transactions = df.groupby("type")["isFraud"].sum().sort_values(ascending=False)
fraud_transactions.plot(kind="bar", color="orange")
plt.title("Fraudulent Transactions by Type")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.show()

print(df["amount"].describe().astype(int))

sns.histplot(df["amount"], bins=200, kde=True, color="red")
plt.title("Distribution of Transaction Amounts (log scale)")
plt.xlabel("log(Transaction Amount + 1)")
plt.ylabel("Frequency")
plt.show()

sns.boxplot(data=df[df["amount"] < 150000], x="amount", y="type", palette="Set2")
plt.title("Boxplot of Transaction Amounts by Type (Amount < 150000)")
plt.show()

# ---------------------------------------------------------------------------
# Feature engineering & cleaning
# ---------------------------------------------------------------------------
df["BalanceDifference"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["BalanceDifference"] = df["newbalanceOrig"] - df["oldbalanceOrg"]

print("Negative BalanceDifference count:", (df["BalanceDifference"] < 0).sum())

fraud = df["isFraud"].value_counts(normalize=True) * 100
print("\nFraud distribution (%):")
print(fraud)

df = df.drop(["nameOrig", "nameDest"], axis=1)
print(df.head())

encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])
print(df.head())

mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Type encoding map:", mapping)

# ---------------------------------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="isFraud", data=df)
plt.title("Fraud Distribution")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="type", data=df)
plt.title("Transaction Types")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df["amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="type", hue="isFraud", data=df)
plt.title("Fraud by Transaction Type")
plt.show()

# ---------------------------------------------------------------------------
# Persist preprocessed data and encoder for ML
# ---------------------------------------------------------------------------
df.to_csv("processed_fraud_data.csv", index=False)
print("Preprocessed dataset saved successfully.")

joblib.dump(encoder, "label_encoder.pkl")
print("Encoder saved successfully!")

# ---------------------------------------------------------------------------
# Model evaluation plots (requires eval_artifacts.pkl from ml_model.py)
# ---------------------------------------------------------------------------
artifacts_path = Path("eval_artifacts.pkl")
if not artifacts_path.exists():
    print("\nSkipping model plots — run ml_model.py first to create eval_artifacts.pkl")
else:
    artifacts = joblib.load(artifacts_path)
    log_model = artifacts["log_model"]
    dt_model = artifacts["dt_model"]
    rf_model = artifacts["rf_model"]
    scaler = artifacts["scaler"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    feature_names = artifacts["feature_names"]
    comparison = artifacts["comparison"]

    X_test_scaled = scaler.transform(X_test)

    log_predictions = log_model.predict(X_test_scaled)
    log_probability = log_model.predict_proba(X_test_scaled)
    dt_pred = dt_model.predict(X_test)
    dt_prob = dt_model.predict_proba(X_test)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)

    # Logistic Regression
    cm = confusion_matrix(y_test, log_predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": log_model.coef_[0],
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Coefficient", y="Feature", data=importance)
    plt.title("Logistic Regression Feature Importance")
    plt.show()

    # Decision Tree
    cm = confusion_matrix(y_test, dt_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
    plt.title("Decision Tree Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, dt_prob[:, 1])
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label="Decision Tree")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Decision Tree ROC Curve")
    plt.legend()
    plt.show()

    importance_dt = pd.DataFrame({
        "Feature": feature_names,
        "Importance": dt_model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_dt)
    plt.title("Decision Tree Feature Importance")
    plt.show()

    # Random Forest
    cm = confusion_matrix(y_test, rf_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, rf_prob[:, 1])
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label="Random Forest")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest ROC Curve")
    plt.legend()
    plt.show()

    importance_rf = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_rf)
    plt.title("Random Forest Feature Importance")
    plt.show()

    # Model comparison
    comparison.set_index("Model").plot(kind="bar", figsize=(12, 6))
    plt.title("Performance Comparison of All Models")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis="y")
    plt.show()

    print(comparison.sort_values(by="Accuracy", ascending=False))
