import streamlit as st
import joblib
import numpy as np
import xgboost as xgb

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("This demo predicts whether a transaction is fraudulent using an XGBoost model.")

# -------------------- LOAD MODEL & SCALER --------------------
@st.cache_resource
def load_resources():
    model = joblib.load("xgboost_fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_resources()

# -------------------- USER INPUT --------------------
st.subheader("Enter Transaction Details")

time = st.number_input("Transaction Time (seconds)", min_value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0)

threshold = st.slider(
    "Fraud Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.01
)

st.caption("⚠️ PCA-based features (V1–V28) are auto-generated for demo purposes.")

# -------------------- PREDICTION --------------------
if st.button("Predict Transaction"):

    # Generate realistic PCA features
    v_features = np.random.normal(0, 1, 28)

    # Scale amount (same as training)
    amount_scaled = scaler.transform([[amount]])[0][0]

    # Final input: [Time, V1–V28, Amount]
    input_data = np.array([[time] + list(v_features) + [amount_scaled]])

    # Predict fraud probability
    try:
        prob = model.predict_proba(input_data)[0][1]
    except:
        dmatrix = xgb.DMatrix(input_data)
        prob = float(model.predict(dmatrix, output_margin=False)[0])

    # Classification
    prediction = "Fraudulent" if prob > threshold else "Normal"

    # -------------------- OUTPUT --------------------
    st.subheader("Prediction Result")

    if prediction == "Fraudulent":
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Normal Transaction")

    st.write(f"**Fraud Probability:** {prob:.4f}")

    if prob < 0.1:
        st.info("🟢 Low Risk")
    elif prob < 0.5:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")
