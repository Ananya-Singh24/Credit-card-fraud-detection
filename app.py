import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("This app uses a trained XGBoost model to detect fraudulent transactions.")

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("xgboost_fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Inputs
time = st.number_input("Transaction Time (seconds)", min_value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0)

threshold = st.slider(
    "Fraud Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.01
)

if st.button("Predict Transaction"):
    # Create dummy V1–V28 features (as in dataset)
    v_features = np.zeros(28)

    # Combine input
    input_data = np.array([[time] + list(v_features) + [amount]])

    # Scale Time & Amount together
    time_amount = input_data[:, [0, -1]]
    time_amount_scaled = scaler.transform(time_amount)
    input_data[:, 0] = time_amount_scaled[:, 0]
    input_data[:, -1] = time_amount_scaled[:, 1]

    # Prediction
    prob = model.predict_proba(input_data)[0][1]
    pred = 1 if prob > threshold else 0

    if pred == 1:
        st.error("⚠️ Fraudulent Transaction")
    else:
        st.success("✅ Normal Transaction")

    st.write(f"**Fraud Probability:** {prob:.4f}")

    if prob < 0.1:
        st.info("🟢 Low Risk")
    elif prob < 0.5:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")
