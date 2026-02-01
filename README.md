💳 Credit Card Fraud Detection
📌 Overview

A machine learning–based Credit Card Fraud Detection System built using the Kaggle European credit card transactions dataset.
The project predicts whether a transaction is fraudulent or legitimate and displays results through an interactive Streamlit web app.

🚀 Features

Real-time fraud prediction

Fraud probability visualization

Adjustable detection threshold

Dark / Light theme support

Clean sidebar-based UI

Sample transaction for demo

🗂 Dataset

Kaggle European Credit Card Transactions

Features: Time, Amount, V1 – V28 (PCA-based)

Only Time and Amount were scaled

🧠 Model

Algorithm: XGBoost Classifier

Output: Fraud probability

🖥 Tech Stack

Python

Scikit-learn

XGBoost

Streamlit
🖥 Web Application (Streamlit)

The trained model is deployed using Streamlit, allowing users to:

Enter transaction details

Adjust fraud detection sensitivity using a threshold slider

View fraud probability with visual indicators

Instantly classify transactions as fraud or legitimate
🚀 Deployment

The application is deployed using Streamlit Cloud, making it accessible as a web-based fraud detection system without requiring local setup.

📈 Use Cases

Banking and financial fraud monitoring

Real-time transaction risk analysis

Academic demonstration of ML deployment

Interview and project evaluation showcase
