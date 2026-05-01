#!/usr/bin/env python3
"""
PROJECT_P Fraud Detection App
=============================
Streamlit frontend for scoring uploaded CSV files with the trained models.
Users can choose Decision Tree, XGBoost, HGNN, or any combination.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Streamlit's file watcher can be unstable on some macOS setups.
# Disable it so the app starts reliably.
MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".mplconfig"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure local package imports (src/*) resolve consistently in cloud/runtime environments.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAX_TABLE_RENDER_ROWS = 2000
MAX_CHART_RENDER_ROWS = 5000

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config import (
    SCALER_PATH,
    LABEL_ENCODERS_PATH,
    FEATURE_NAMES_PATH,
    DT_MODEL_PATH,
    XGB_MODEL_PATH,
    XGB_BOOSTER_PATH,
    HGNN_ATT_TD_PATH,
    MISSING_THRESHOLD,
    AMT_CAP_PERCENTILE,
    TARGET_COL,
    ID_COL,
)
try:
    from src.input_resolution import expand_model_input
except ModuleNotFoundError:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _ResolvedInput:
        frame: pd.DataFrame
        mode: str = "passthrough"
        split: str | None = None
        note: str | None = (
            "input_resolution module not available in this deployment. "
            "Using uploaded CSV as-is; id-only auto-expansion is disabled."
        )

    def expand_model_input(df: pd.DataFrame, data_dir: str | None = None) -> _ResolvedInput:
        _ = data_dir
        return _ResolvedInput(frame=df)


def app_cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap transaction amount outliers for inference consistency."""
    out = df.copy()
    if "TransactionAmt" in out.columns:
        cap_val = out["TransactionAmt"].quantile(AMT_CAP_PERCENTILE / 100)
        out["TransactionAmt"] = out["TransactionAmt"].clip(upper=cap_val)
    return out


def app_handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight missing-value handling compatible with training pipeline."""
    out = df.copy()

    # Drop extremely sparse columns like training preprocessing.
    missing_pct = out.isnull().mean()
    high_missing_cols = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    high_missing_cols = [c for c in high_missing_cols if c != TARGET_COL]
    if high_missing_cols:
        out = out.drop(columns=high_missing_cols)

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if out[col].isnull().any():
            out[col] = out[col].fillna(out[col].median())

    cat_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        if out[col].isnull().any():
            out[col] = out[col].fillna("Unknown")

    return out


st.set_page_config(
    page_title="PROJECT_P Fraud Detector",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: linear-gradient(180deg, rgba(30,35,50,0.9), rgba(16,20,30,0.95));
        box-shadow: 0 12px 24px rgba(0,0,0,0.18);
    }
    .small-note { color: #9aa4b2; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_preprocess_artifacts() -> Dict[str, object]:
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return {
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_names": feature_names,
    }


@st.cache_resource(show_spinner=False)
def load_decision_tree_model():
    return joblib.load(DT_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_xgboost_model():
    import xgboost as xgb

    if XGB_BOOSTER_PATH.exists():
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(XGB_BOOSTER_PATH))
        xgb_model.set_param({"nthread": 1})
        return ("booster", xgb_model)

    if XGB_MODEL_PATH.exists():
        return ("sklearn", joblib.load(XGB_MODEL_PATH))

    raise FileNotFoundError(
        f"Missing XGBoost artifacts. Expected one of: {XGB_BOOSTER_PATH} or {XGB_MODEL_PATH}"
    )


def read_uploaded_csv(uploaded_file, nrows: int | None = None) -> pd.DataFrame:
    """Read uploaded CSV without materializing a full bytes copy in memory."""
    uploaded_file.seek(0)
    read_kwargs = {"nrows": nrows}
    name = str(getattr(uploaded_file, "name", "")).lower()
    if name.endswith(".gz"):
        read_kwargs["compression"] = "gzip"
    return pd.read_csv(uploaded_file, **read_kwargs)


def safe_label_encode(series: pd.Series, encoder) -> pd.Series:
    known = {str(cls): idx for idx, cls in enumerate(encoder.classes_)}
    fallback_key = "Unknown" if "Unknown" in known else encoder.classes_[0]
    fallback_value = known.get(fallback_key, 0)

    values = series.astype(str).fillna("Unknown")
    encoded = values.map(lambda value: known.get(value, fallback_value)).astype(np.int64)
    return encoded


def preprocess_for_model(
    raw_df: pd.DataFrame,
    scaler,
    feature_names,
    label_encoders: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = raw_df.copy()

    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    df = app_cap_outliers(df)
    df = app_handle_missing_values(df)

    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = safe_label_encode(df[col], encoder)

    # Keep only the trained feature columns and add any missing ones as zeros.
    missing_features = [col for col in feature_names if col not in df.columns]
    if missing_features:
        df = pd.concat(
            [df, pd.DataFrame(0, index=df.index, columns=missing_features)],
            axis=1,
        )

    numeric_df = df.reindex(columns=feature_names).copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.fillna(0)

    scaled = scaler.transform(numeric_df)
    scaled_df = pd.DataFrame(scaled, columns=feature_names, index=numeric_df.index)
    return scaled_df, df


def build_hgnn_graph(raw_df: pd.DataFrame, scaled_df: pd.DataFrame):
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    n = len(raw_df)
    data["transaction"].x = torch.tensor(scaled_df.values, dtype=torch.float32)

    card_source = raw_df["card1"] if "card1" in raw_df.columns else pd.Series(np.arange(n))
    card_codes, card_uniques = pd.factorize(card_source.fillna("missing").astype(str), sort=True)
    data["card"].x = torch.ones((len(card_uniques), 1), dtype=torch.float32)
    card_idx = np.vstack([np.arange(n, dtype=np.int64), card_codes.astype(np.int64)])
    card_edges = torch.from_numpy(card_idx)
    data[("transaction", "used", "card")].edge_index = card_edges
    data[("card", "used_by", "transaction")].edge_index = card_edges.flip(0)

    if "DeviceInfo" in raw_df.columns:
        device_source = raw_df["DeviceInfo"]
    elif "DeviceType" in raw_df.columns:
        device_source = raw_df["DeviceType"]
    else:
        device_source = pd.Series(["unknown"] * n)

    dev_codes, dev_uniques = pd.factorize(device_source.fillna("missing").astype(str), sort=True)
    data["device"].x = torch.ones((len(dev_uniques), 1), dtype=torch.float32)
    dev_idx = np.vstack([np.arange(n, dtype=np.int64), dev_codes.astype(np.int64)])
    dev_edges = torch.from_numpy(dev_idx)
    data[("transaction", "on", "device")].edge_index = dev_edges
    data[("device", "hosts", "transaction")].edge_index = dev_edges.flip(0)

    if "TransactionDT" in raw_df.columns:
        tx_time = pd.to_numeric(raw_df["TransactionDT"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        age = tx_time.max() - tx_time
        age_scale = max(np.percentile(age, 95), 1.0)
        decay = np.exp(-age / age_scale).astype(np.float32)
    else:
        decay = np.ones(n, dtype=np.float32)
    data["transaction"].time_decay = torch.from_numpy(decay).unsqueeze(-1)
    return data


def predict_hgnn(raw_df: pd.DataFrame, X_scaled: pd.DataFrame) -> np.ndarray:
    import torch
    from src.models import FraudHGNN

    checkpoint = torch.load(HGNN_ATT_TD_PATH, map_location="cpu", weights_only=False)
    expected_dim = int(checkpoint.get("input_dim", X_scaled.shape[1]))

    if X_scaled.shape[1] > expected_dim:
        x_for_hgnn = X_scaled.iloc[:, :expected_dim].copy()
    elif X_scaled.shape[1] < expected_dim:
        pad = expected_dim - X_scaled.shape[1]
        x_for_hgnn = X_scaled.copy()
        for i in range(pad):
            x_for_hgnn[f"__hgnn_pad_{i}"] = 0.0
    else:
        x_for_hgnn = X_scaled

    model = FraudHGNN(
        metadata=(
            ["transaction", "card", "device"],
            [
                ("transaction", "used", "card"),
                ("card", "used_by", "transaction"),
                ("transaction", "on", "device"),
                ("device", "hosts", "transaction"),
            ],
        ),
        input_dim=expected_dim,
        hidden_dims=checkpoint.get("hidden_dims"),
        dropout_rates=checkpoint.get("dropout_rates"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    graph = build_hgnn_graph(raw_df, x_for_hgnn)
    with torch.no_grad():
        logits = model(
            graph.x_dict,
            graph.edge_index_dict,
            tx_time_decay=graph["transaction"].time_decay,
        )
        proba = torch.sigmoid(logits).squeeze().cpu().numpy()
    return proba


def predict_models(raw_df: pd.DataFrame, artifacts: Dict[str, object], selected_models: List[str]) -> pd.DataFrame:
    import xgboost as xgb

    scaled_df, processed_df = preprocess_for_model(
        raw_df,
        artifacts["scaler"],
        artifacts["feature_names"],
        artifacts["label_encoders"],
    )

    result = pd.DataFrame({
        ID_COL: raw_df[ID_COL].values if ID_COL in raw_df.columns else np.arange(len(raw_df)),
    })

    for model_name in selected_models:
        if model_name == "decision_tree":
            dt_model = load_decision_tree_model()
            proba = dt_model.predict_proba(scaled_df)[:, 1]
            result["decision_tree_probability"] = proba
            result["decision_tree_pred"] = (proba >= 0.5).astype(int)
        elif model_name == "xgboost":
            model_kind, xgb_model = load_xgboost_model()
            xgb_array = np.ascontiguousarray(scaled_df.values.astype(np.float32))
            if model_kind == "booster":
                xgb_dmatrix = xgb.DMatrix(xgb_array)
                proba = xgb_model.predict(xgb_dmatrix)
            else:
                proba = xgb_model.predict_proba(xgb_array)[:, 1]
            result["xgboost_probability"] = proba
            result["xgboost_pred"] = (proba >= 0.5).astype(int)
        elif model_name == "hgnn":
            proba = predict_hgnn(processed_df, scaled_df)
            result["hgnn_probability"] = proba
            result["hgnn_pred"] = (proba >= 0.5).astype(int)

    if TARGET_COL in raw_df.columns:
        y_true = raw_df[TARGET_COL].astype(int).to_numpy()
        result["true_label"] = y_true

    return result


def predict_models_subprocess(raw_df: pd.DataFrame, selected_models: List[str]) -> pd.DataFrame:
    """Run each selected model in an isolated process to keep Streamlit alive on native crashes."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    merged_result: pd.DataFrame | None = None
    failures: List[str] = []

    with tempfile.TemporaryDirectory(prefix="fraud_app_") as tmpdir:
        input_path = os.path.join(tmpdir, "input.csv")
        raw_df.to_csv(input_path, index=False)

        for model_name in selected_models:
            output_path = os.path.join(tmpdir, f"pred_{model_name}.csv")
            cmd = [
                sys.executable,
                "predict_models_options.py",
                "--input",
                input_path,
                "--model",
                model_name,
                "--output",
                output_path,
            ]
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

            proc = subprocess.run(
                cmd,
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
            )

            if proc.returncode != 0:
                failures.append(f"{model_name}: exit {proc.returncode}")
                continue
            if not os.path.exists(output_path):
                failures.append(f"{model_name}: no output generated")
                continue

            part = pd.read_csv(output_path)
            if merged_result is None:
                merged_result = part
            else:
                extra_cols = [c for c in part.columns if c != ID_COL]
                merged_result = merged_result.merge(part[[ID_COL] + extra_cols], on=ID_COL, how="left")

    if merged_result is None:
        detail = "; ".join(failures) if failures else "unknown failure"
        raise RuntimeError(f"All selected model processes failed ({detail}).")

    if TARGET_COL in raw_df.columns and "true_label" not in merged_result.columns:
        if ID_COL in raw_df.columns and ID_COL in merged_result.columns:
            labels = raw_df[[ID_COL, TARGET_COL]].copy()
            labels = labels.rename(columns={TARGET_COL: "true_label"})
            merged_result = merged_result.merge(labels, on=ID_COL, how="left")
        else:
            merged_result["true_label"] = raw_df[TARGET_COL].to_numpy()[: len(merged_result)]

    return merged_result


st.title("Fraud Detection Checker")
st.caption("Upload transaction data and run the three trained models: Decision Tree, XGBoost, and HGNN.")

with st.sidebar:
    st.header("Upload CSV")
    uploaded_file = st.file_uploader("CSV file", type=["csv"])
    selected_models = st.multiselect(
        "Select model(s)",
        options=["decision_tree", "xgboost", "hgnn"],
        default=["decision_tree", "xgboost", "hgnn"],
    )
    threshold = st.slider("Fraud threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01)
    max_rows = st.number_input(
        "Max rows to score (0 = all)",
        min_value=0,
        max_value=1_000_000,
        value=2000,
        step=1000,
    )
    max_cols = st.number_input(
        "Max columns to score (0 = all)",
        min_value=0,
        max_value=10000,
        value=80,
        step=10,
    )
    st.markdown(
        """
        <div class="small-note">
        Supported input: full merged feature CSV,
        or id-only CSV (for example sample_submission with TransactionID).
        Id-only files are auto-expanded to model features by TransactionID.
        Very large files (for example 0.7GB) can upload slowly on Streamlit Cloud;
        id-only CSV is usually much faster.
        Use row/column limits above for faster test runs on large files.
        If <code>isFraud</code> is present, the app also shows accuracy metrics.
        </div>
        """,
        unsafe_allow_html=True,
    )

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='metric-card'><h4>Decision Tree</h4><p>Fast baseline</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-card'><h4>XGBoost</h4><p>Best tabular model</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-card'><h4>HGNN</h4><p>Graph-aware fraud model</p></div>", unsafe_allow_html=True)

if not uploaded_file:
    st.info("Upload a CSV file to begin.")
    st.stop()

if not selected_models:
    st.warning("Select at least one model.")
    st.stop()

try:
    read_nrows = int(max_rows) if int(max_rows) > 0 else None
    raw_df = read_uploaded_csv(uploaded_file, nrows=read_nrows)
    pre_expand_row_cap_note: str | None = None
    if int(max_rows) > 0 and len(raw_df) > int(max_rows):
        raw_df = raw_df.head(int(max_rows)).copy()
        pre_expand_row_cap_note = f"Applied pre-expansion row cap: {len(raw_df):,}"

    resolved_input = expand_model_input(raw_df)
    raw_df = resolved_input.frame
    if int(max_rows) > 0 and len(raw_df) > int(max_rows):
        raw_df = raw_df.head(int(max_rows)).copy()
    if int(max_cols) > 0 and len(raw_df.columns) > int(max_cols):
        required_cols = [c for c in [ID_COL, TARGET_COL] if c in raw_df.columns]
        other_cols = [c for c in raw_df.columns if c not in required_cols]
        keep_limit = int(max_cols)
        if keep_limit <= len(required_cols):
            keep_cols = required_cols[:keep_limit]
        else:
            keep_cols = required_cols + other_cols[: keep_limit - len(required_cols)]
        raw_df = raw_df[keep_cols].copy()
except Exception as exc:
    st.error(f"Could not read the uploaded files: {exc}")
    st.stop()

st.subheader("Uploaded Data Preview")
st.write(raw_df.head(10))
st.caption(f"Rows: {len(raw_df):,} | Columns: {len(raw_df.columns):,}")
if pre_expand_row_cap_note:
    st.info(pre_expand_row_cap_note)
if resolved_input.note:
    st.info(resolved_input.note)

if st.button("Run Fraud Check", type="primary"):
    try:
        with st.spinner("Loading models and scoring rows..."):
            results = predict_models_subprocess(raw_df, selected_models)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    flag_specs = [
        ("decision_tree_probability", "DT Fraud Flags"),
        ("xgboost_probability", "XGB Fraud Flags"),
        ("hgnn_probability", "HGNN Fraud Flags"),
    ]
    for prob_col, _ in flag_specs:
        if prob_col in results.columns:
            results[prob_col.replace("_probability", "_flag")] = (results[prob_col] >= threshold).astype(int)

    st.subheader("Prediction Summary")
    shown = [(p, label) for p, label in flag_specs if p in results.columns]
    summary_cols = st.columns(max(1, len(shown)))
    for i, (prob_col, label) in enumerate(shown):
        with summary_cols[i]:
            st.metric(label, int(results[prob_col.replace("_probability", "_flag")].sum()))

    if "true_label" in results.columns:
        st.subheader("Metrics vs Uploaded Labels")
        y_true = results["true_label"].to_numpy()
        metric_rows = []
        for model_name, prob_col in [
            ("Decision Tree", "decision_tree_probability"),
            ("XGBoost", "xgboost_probability"),
            ("HGNN", "hgnn_probability"),
        ]:
            if prob_col not in results.columns:
                continue
            pred_col = prob_col.replace("_probability", "_prediction")
            if pred_col not in results.columns:
                # Backward compatibility with any older in-memory outputs.
                legacy_pred_col = prob_col.replace("_probability", "_pred")
                if legacy_pred_col in results.columns:
                    pred_col = legacy_pred_col
                else:
                    continue
            probs = results[prob_col].to_numpy()
            preds = results[pred_col].to_numpy()
            metric_rows.append({
                "Model": model_name,
                "ROC-AUC": roc_auc_score(y_true, probs),
                "Accuracy": accuracy_score(y_true, preds),
                "Precision": precision_score(y_true, preds, zero_division=0),
                "Recall": recall_score(y_true, preds, zero_division=0),
                "F1": f1_score(y_true, preds, zero_division=0),
            })
        if metric_rows:
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)

    st.subheader("Per-Row Predictions")
    render_rows = min(len(results), MAX_TABLE_RENDER_ROWS)
    if len(results) > MAX_TABLE_RENDER_ROWS:
        st.caption(
            f"Showing first {MAX_TABLE_RENDER_ROWS:,} rows for responsiveness. "
            f"Download CSV for full {len(results):,} rows."
        )
    st.dataframe(results.head(render_rows), use_container_width=True, height=420)

    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions CSV",
        data=csv_bytes,
        file_name="fraud_predictions.csv",
        mime="text/csv",
    )

    st.subheader("Model Comparison")
    chart_map = {
        "decision_tree_probability": "Decision Tree",
        "xgboost_probability": "XGBoost",
        "hgnn_probability": "HGNN",
    }
    chart_cols = [c for c in chart_map if c in results.columns]
    if chart_cols:
        chart_df = results[chart_cols].head(MAX_CHART_RENDER_ROWS).copy()
        chart_df.columns = [chart_map[c] for c in chart_cols]
        if len(results) > MAX_CHART_RENDER_ROWS:
            st.caption(f"Chart uses first {MAX_CHART_RENDER_ROWS:,} rows for responsiveness.")
        st.line_chart(chart_df)

    st.success("Fraud scoring complete.")
