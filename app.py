import streamlit as st
import numpy as np
import joblib

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------- Theme Toggle ----------------
theme = st.sidebar.radio("🌗 Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #0E1117; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------------- Load Model & Scaler ----------------
model = joblib.load("xgboost_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")   # scaler ONLY for Time & Amount

# ---------------- Title ----------------
st.title("💳 Credit Card Fraud Detection System")
st.subheader("European Card Transactions | XGBoost Model")

st.write(
    "This application predicts whether a credit card transaction is **fraudulent or legitimate** "
    "using a trained **XGBoost model** on real European card transaction data."
)

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🧾 Transaction Details")

time = st.sidebar.number_input("Transaction Time", min_value=0.0)
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)

st.sidebar.markdown("### PCA Features (V1 – V28)")

v_features = []
for i in range(1, 29):
    v = st.sidebar.number_input(f"V{i}", value=0.0)
    v_features.append(v)

# ---------------- Sample Transaction ----------------
if st.sidebar.button("⚡ Use Sample Fraud Transaction"):
    time = 45000
    amount = 2500
    v_features = [
        -2.3, 1.9, -0.8, 0.6, -1.1, 0.4, -0.5,
        -1.8, 0.2, -0.3, 1.2, -0.9, -0.4, 0.7,
        -1.5, 0.1, -0.2, -0.6, 0.9, -0.7,
        0.3, -1.0, -0.8, 0.5, -0.4, 0.6, -0.2, -1.3
    ]

# ---------------- Threshold ----------------
st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "🎚 Fraud Threshold",
    0.0, 1.0, 0.5, 0.01
)

# ---------------- Prediction ----------------
if st.button("🔍 Check Transaction"):

    # Scale ONLY Time & Amount
    scaled_time_amount = scaler.transform([[time, amount]])

    # Combine scaled + PCA features
    final_input = np.concatenate(
        [scaled_time_amount[0], np.array(v_features)]
    ).reshape(1, -1)

    # Predict probability
    prob = model.predict_proba(final_input)[0][1]

    # UI Output
    st.metric("Fraud Probability", f"{prob:.2%}")
    st.progress(prob)

    if prob >= threshold:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

# ---------------- Explainability ----------------
with st.expander("ℹ️ How does this model work?"):
    st.write(
        """
        - **Time & Amount** are scaled using the same scaler used during training.
        - **V1–V28** are PCA-transformed features and are used directly.
        - The XGBoost model outputs a **fraud probability**.
        - You can adjust the **threshold** to control sensitivity.
        
        Lower threshold → catches more fraud  
        Higher threshold → fewer false alarms
        """
    )

# ---------------- Footer ----------------
st.markdown("---")
st.caption("🔐 Credit Card Fraud Detection | Kaggle European Dataset | XGBoost + Streamlit")
