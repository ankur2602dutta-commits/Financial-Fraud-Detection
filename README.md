# Financial-Fraud-Detection

# Financial Fraud Detection Using Machine Learning

## Overview
This project is a Machine Learning-based Financial Fraud Detection System that predicts whether a financial transaction is **Fraudulent** or **Legitimate**. It analyzes transaction details such as transaction type, amount, sender balance, and receiver balance using a trained Random Forest model. The application provides a simple Streamlit web interface for real-time predictions.

## Features
- Detects fraudulent transactions
- User-friendly Streamlit interface
- Real-time prediction
- Machine Learning model using Random Forest
- Label Encoding for categorical data
- Model saved using Joblib

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- LabelEncoder
- Joblib
- Streamlit
- Matplotlib
- Seaborn

## Project Structure

```
Financial-Fraud-Detection/
│
├── app.py
├── train.py
├── fraud_model.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
└── dataset/
    └── transactions.csv
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ankur2602dutta-commits/Financial-Fraud-Detection.git
```

Move into the project folder:

```bash
cd Financial-Fraud-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Input Features
- Transaction Type
- Amount
- Old Balance Sender
- New Balance Sender
- Old Balance Receiver
- New Balance Receiver

## Machine Learning Workflow
1. Load dataset
2. Data preprocessing
3. Label Encoding
4. Feature Engineering
5. Train Random Forest model
6. Save model using Joblib
7. Predict fraud using Streamlit

## Author
**Ankur**
