#!/usr/bin/env python
# coding: utf-8
"""ML model training for financial fraud detection.

Preprocessing + model creation only. Visualizations live in eda.py.
UI: streamlit run app.py
"""

import pandas as pd
import warnings
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load & preprocess
# ---------------------------------------------------------------------------
df = pd.read_csv(r"Synthetic_Financial_datasets_log.csv")

df["BalanceDifference"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
df = df.drop(["nameOrig", "nameDest"], axis=1)

encoder = LabelEncoder()
df["type"] = encoder.fit_transform(df["type"])

df.to_csv("processed_fraud_data.csv", index=False)
joblib.dump(encoder, "label_encoder.pkl")
print("Preprocessing complete. Encoder saved.")

X = df.drop("isFraud", axis=1)
y = df["isFraud"]

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling Completed")

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)
print("Logistic Regression Trained")

log_predictions = log_model.predict(X_test_scaled)
log_probability = log_model.predict_proba(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, log_predictions))
print("Precision:", precision_score(y_test, log_predictions))
print("Recall:", recall_score(y_test, log_predictions))
print(classification_report(y_test, log_predictions))

# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
print("Decision Tree Model Trained Successfully")

dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Precision:", precision_score(y_test, dt_pred))
print("Recall:", recall_score(y_test, dt_pred))
print("F1 Score:", f1_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)
print("Random Forest Trained Successfully")

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
print("ROC AUC:", roc_auc_score(y_test, rf_prob[:, 1]))
print(classification_report(y_test, rf_pred))

# ---------------------------------------------------------------------------
# Comparison table (printed; plotted in eda.py)
# ---------------------------------------------------------------------------
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, log_predictions),
        accuracy_score(y_test, dt_pred),
        accuracy_score(y_test, rf_pred),
    ],
    "Precision": [
        precision_score(y_test, log_predictions),
        precision_score(y_test, dt_pred),
        precision_score(y_test, rf_pred),
    ],
    "Recall": [
        recall_score(y_test, log_predictions),
        recall_score(y_test, dt_pred),
        recall_score(y_test, rf_pred),
    ],
    "F1 Score": [
        f1_score(y_test, log_predictions),
        f1_score(y_test, dt_pred),
        f1_score(y_test, rf_pred),
    ],
    "ROC AUC": [
        roc_auc_score(y_test, log_probability[:, 1]),
        roc_auc_score(y_test, dt_prob[:, 1]),
        roc_auc_score(y_test, rf_prob[:, 1]),
    ],
})

print(comparison)
print(comparison.sort_values(by="Accuracy", ascending=False))

# ---------------------------------------------------------------------------
# Save models & evaluation artifacts for eda.py / app.py
# ---------------------------------------------------------------------------
joblib.dump(rf_model, "fraud_model.pkl")
joblib.dump(
    {
        "log_model": log_model,
        "dt_model": dt_model,
        "rf_model": rf_model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "comparison": comparison,
    },
    "eval_artifacts.pkl",
)

print("Model saved successfully!")
print("Evaluation artifacts saved for eda.py plots.")
print("Model expects features:", list(rf_model.feature_names_in_))
print("Run the UI with: streamlit run app.py")
