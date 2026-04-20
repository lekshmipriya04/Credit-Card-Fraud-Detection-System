import pickle
from pathlib import Path

import numpy as np
import streamlit as st


st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "neural_network_model.pkl"

# Model was trained on:
# ['scaled_amount', 'scaled_time', 'V1', ..., 'V28']
FEATURE_ORDER = ["scaled_amount", "scaled_time"] + [f"V{i}" for i in range(1, 29)]


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def main():
    st.title("Credit Card Fraud Detection")
    st.caption("Using best model: Neural Network (MLP) from pickle file")

    if not MODEL_PATH.exists():
        st.error("neural_network_model.pkl not found in the same folder as this app.")
        st.stop()

    model = load_model()

    st.info(
        "Enter feature values and click Predict. "
        "Important: Use scaled_amount and scaled_time values (same preprocessing as training)."
    )

    threshold = st.slider(
        "Fraud decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01
    )

    st.subheader("Input Features")
    inputs = {}
    cols = st.columns(3)

    for i, feature in enumerate(FEATURE_ORDER):
        with cols[i % 3]:
            inputs[feature] = st.number_input(feature, value=0.0, format="%.6f")

    if st.button("Predict", type="primary"):
        x_input = np.array([[inputs[f] for f in FEATURE_ORDER]], dtype=float)

        fraud_prob = float(model.predict_proba(x_input)[0][1])
        pred = int(fraud_prob >= threshold)

        st.subheader("Prediction Result")
        if pred == 1:
            st.error("Fraud Transaction Detected")
        else:
            st.success("Legit Transaction")

        st.metric("Fraud Probability", f"{fraud_prob:.4f}")
        st.metric("Predicted Class", str(pred))


if __name__ == "__main__":
    main()
