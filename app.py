#!/usr/bin/env python
# coding: utf-8
"""Streamlit UI for financial fraud detection.

Requires fraud_model.pkl and label_encoder.pkl from ml_model.py.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Financial Fraud Detection System")
st.write("Predict whether a transaction is Fraud or Genuine")

st.sidebar.header("Transaction Details")

transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    encoder.classes_,
)

amount = st.sidebar.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance Sender", value=0.0)
newbalanceOrig = st.sidebar.number_input("New Balance Sender", value=0.0)
oldbalanceDest = st.sidebar.number_input("Old Balance Receiver", value=0.0)
newbalanceDest = st.sidebar.number_input("New Balance Receiver", value=0.0)
step = st.sidebar.slider("Step", 1, 750, 100)
isFlaggedFraud = st.sidebar.selectbox("Flagged Fraud", [0, 1])


def build_features(
    step,
    type_encoded,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    isFlaggedFraud,
):
    """Build a feature row matching the trained model's expected columns."""
    features = pd.DataFrame({
        "step": [step],
        "type": [type_encoded],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest],
        "isFlaggedFraud": [isFlaggedFraud],
        "BalanceDifference": [newbalanceOrig - oldbalanceOrg],
    })
    return features.reindex(columns=model.feature_names_in_, fill_value=0)


type_encoded = encoder.transform([transaction_type])[0]
features = build_features(
    step,
    type_encoded,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest,
    isFlaggedFraud,
)

if st.button("Predict"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("Prediction")
    if prediction == 1:
        st.error("🚨 Fraud Transaction Detected")
    else:
        st.success("✅ Genuine Transaction")
    st.write(f"Fraud Probability: {probability * 100:.2f}%")

st.markdown("---")
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if "type" in data.columns and data["type"].dtype == object:
        data["type"] = encoder.transform(data["type"])

    if "BalanceDifference" not in data.columns and {
        "oldbalanceOrg",
        "newbalanceOrig",
    }.issubset(data.columns):
        data["BalanceDifference"] = data["newbalanceOrig"] - data["oldbalanceOrg"]

    predict_cols = list(model.feature_names_in_)
    missing = [c for c in predict_cols if c not in data.columns]
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
    else:
        batch_features = data[predict_cols]
        data["Prediction"] = model.predict(batch_features)
        data["FraudProbability"] = model.predict_proba(batch_features)[:, 1]
        st.write(data.head())
        st.download_button(
            "Download Results",
            data.to_csv(index=False),
            "prediction.csv",
            "text/csv",
        )

st.markdown("---")
st.write("Developed using Python + Scikit-Learn + Streamlit")
st.caption(f"Model expects: {list(model.feature_names_in_)}")
