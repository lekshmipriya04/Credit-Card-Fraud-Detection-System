import streamlit as st
import pandas as pd
import joblib

# Set up the page
st.set_page_config(page_title="Fraud Detection App", page_icon="🛡️")
st.title("🛡️ Credit Card Fraud Detection System")
st.write("Upload a CSV of transactions to detect potential fraud.")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('xgboost_fraud_model.pkl')

model = load_model()

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read data
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Uploaded Data Preview:")
    st.dataframe(df.head())
    
    # Remove 'Class' column if it exists in the uploaded file
    if 'Class' in df.columns:
        features = df.drop('Class', axis=1)
    else:
        features = df

    # Make Predictions
    if st.button("Predict Fraud"):
        with st.spinner('Analyzing...'):
            predictions = model.predict(features)
            df['Fraud_Prediction'] = predictions
            
            fraud_count = (df['Fraud_Prediction'] == 1).sum()
            legit_count = len(df) - fraud_count
            
            st.error(f"⚠️ Found {fraud_count} fraudulent transactions!")
            st.success(f"✅ Found {legit_count} legitimate transactions.")