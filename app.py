import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "Decision Tree": BASE_DIR / "models" / "decision_tree_model.pkl",
    "XGBoost": BASE_DIR / "models" / "xgboost_model.pkl",
    "Neural Network": BASE_DIR / "models" / "neural_network_model.pkl",
}

# -----------------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    st.title("💳 Credit Card Fraud Detection System")

    # -----------------------------------------------------------------------------
    # MODEL SELECTION
    # -----------------------------------------------------------------------------
    model_name = st.sidebar.selectbox(
        "Select Model",
        list(MODEL_PATHS.keys())
    )

    model_path = MODEL_PATHS[model_name]

    if not model_path.exists():
        st.error(f"{model_name} model not found!")
        st.stop()

    model = load_model(model_path)

    # -----------------------------------------------------------------------------
    # FEATURE ORDER (AUTO)
    # -----------------------------------------------------------------------------
    try:
        FEATURE_ORDER = list(model.feature_names_in_)
    except:
        st.error("Model missing feature names. Retrain properly.")
        st.stop()

    # -----------------------------------------------------------------------------
    # THRESHOLD SETTINGS
    # -----------------------------------------------------------------------------
    st.sidebar.subheader("Threshold Settings")

    auto_threshold = st.sidebar.checkbox("Auto threshold tuning", value=False)

    if auto_threshold:
        threshold = 0.5  # default, will adjust dynamically
    else:
        threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5, 0.01)

    # -----------------------------------------------------------------------------
    # SINGLE INPUT
    # -----------------------------------------------------------------------------
    st.header("🔢 Manual Prediction")

    inputs = {}
    cols = st.columns(4)

    for i, feature in enumerate(FEATURE_ORDER):
        with cols[i % 4]:
            inputs[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict", type="primary"):
        x_input = pd.DataFrame(
            [[inputs[f] for f in FEATURE_ORDER]],
            columns=FEATURE_ORDER
        )

        prob = float(model.predict_proba(x_input)[0][1])

        # Auto threshold logic
        if auto_threshold:
            threshold = 0.3 if prob > 0.7 else 0.5

        pred = int(prob >= threshold)

        st.subheader("Result")

        if pred == 1:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Legit Transaction")

        st.metric("Fraud Probability", f"{prob:.4f}")
        st.metric("Threshold Used", f"{threshold:.2f}")

    # -----------------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------------------------------
    st.header("📊 Feature Importance")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]  # top 15

        fig, ax = plt.subplots()
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([FEATURE_ORDER[i] for i in indices])

        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model")

    # -----------------------------------------------------------------------------
    # CSV UPLOAD
    # -----------------------------------------------------------------------------
    st.header("📁 Batch Prediction (CSV Upload)")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("Preview:", df.head())

        # Align features
        df = df.reindex(columns=FEATURE_ORDER, fill_value=0)

        probs = model.predict_proba(df)[:, 1]

        if auto_threshold:
            preds = (probs >= 0.3).astype(int)
        else:
            preds = (probs >= threshold).astype(int)

        df["fraud_probability"] = probs
        df["prediction"] = preds

        st.success("Predictions completed!")

        st.dataframe(df.head())

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )

# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()