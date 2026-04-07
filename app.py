import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Fraud Detector App", page_icon="🛡️", layout="centered")

st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("""
This application uses a Machine Learning model (XGBoost) trained with SMOTE to detect fraudulent credit card transactions.
""")

# Load the model
@st.cache_resource # This makes the app run faster by caching the model
def load_model():
    return joblib.load('fraud_model.pkl')

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error("❌ Model file not found. Please ensure 'fraud_model.pkl' is in the directory.")
    st.stop()

st.divider()

st.subheader("Enter Transaction Details")
st.write("Input the key transaction features below to test the model.")

# Create input fields for the user
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
    v1 = st.number_input("Feature V1", value=0.0)
    v2 = st.number_input("Feature V2", value=0.0)

with col2:
    time = st.number_input("Time (Seconds from first transaction)", min_value=0, value=3600)
    v3 = st.number_input("Feature V3", value=0.0)
    v4 = st.number_input("Feature V4", value=0.0)

# Prediction Logic
if st.button("🔍 Analyze Transaction", use_container_width=True):
    
    # The model expects exactly 30 features (scaled_amount, scaled_time, V1-V28).
    # We take the user's inputs and fill the remaining 24 PCA features with 0.0 (the mean).
    input_features = [amount, time, v1, v2, v3, v4] + [0.0] * 24
    
    # Reshape for the model
    input_array = np.array(input_features).reshape(1, -1)
    
    # Predict
    prediction = model.predict(input_array)
    
    st.divider()
    if prediction[0] == 1:
        st.error("### ⚠️ WARNING: FRAUDULENT TRANSACTION DETECTED")
        st.write("This transaction matches the patterns of known credit card fraud.")
    else:
        st.success("### ✅ TRANSACTION APPROVED")
        st.write("This transaction appears legitimate based on our data.")
