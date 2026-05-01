#!/usr/bin/env python3
"""
Option-wise fraud prediction using 3 trained models:
- decision_tree
- xgboost
- hgnn

Supports input CSV as either:
1) full feature table, or
2) id-only file like sample_submission.csv (TransactionID + placeholder isFraud)

When id-only input is provided, this script automatically joins the Kaggle
TEST_TRANSACTION + TEST_IDENTITY files by TransactionID.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Any

# Keep native runtimes conservative on macOS to reduce OpenMP/libomp crashes.
MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".mplconfig"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure local package imports (src/*) resolve consistently in cloud/runtime environments.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    TARGET_COL,
    ID_COL,
    AMT_CAP_PERCENTILE,
    MISSING_THRESHOLD,
    SCALER_PATH,
    LABEL_ENCODERS_PATH,
    FEATURE_NAMES_PATH,
    DT_MODEL_PATH,
    XGB_MODEL_PATH,
    XGB_BOOSTER_PATH,
    HGNN_ATT_TD_PATH,
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
            "Using input CSV as-is; id-only auto-expansion is disabled."
        )

    def expand_model_input(df: pd.DataFrame, data_dir: str | None = None) -> _ResolvedInput:
        _ = data_dir
        return _ResolvedInput(frame=df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run option-wise fraud prediction with 3 models.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV (can be sample_submission.csv or full feature CSV).",
    )
    parser.add_argument(
        "--model",
        choices=["decision_tree", "xgboost", "hgnn", "all"],
        default="all",
        help="Model option to run.",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions_option_wise.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional IEEE-CIS dataset folder containing test_transaction.csv and test_identity.csv.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for quick testing.",
    )
    return parser.parse_args()


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "TransactionAmt" in out.columns:
        cap_val = out["TransactionAmt"].quantile(AMT_CAP_PERCENTILE / 100)
        out["TransactionAmt"] = out["TransactionAmt"].clip(upper=cap_val)
    return out


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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


def safe_label_encode(series: pd.Series, encoder) -> pd.Series:
    known = {str(cls): idx for idx, cls in enumerate(encoder.classes_)}
    fallback_key = "Unknown" if "Unknown" in known else encoder.classes_[0]
    fallback_value = known.get(fallback_key, 0)
    values = series.astype(str).fillna("Unknown")
    return values.map(lambda value: known.get(value, fallback_value)).astype(np.int64)


def preprocess_for_model(raw_df: pd.DataFrame):
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(LABEL_ENCODERS_PATH)
    feature_names: List[str] = joblib.load(FEATURE_NAMES_PATH)

    df = raw_df.copy()
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    df = cap_outliers(df)
    df = handle_missing_values(df)

    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = safe_label_encode(df[col], encoder)

    missing_features = [col for col in feature_names if col not in df.columns]
    if missing_features:
        df = pd.concat(
            [df, pd.DataFrame(0, index=df.index, columns=missing_features)],
            axis=1,
        )

    X = df.reindex(columns=feature_names).copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names, index=X.index)

    return X_scaled, df


def build_hgnn_graph(raw_df: pd.DataFrame, scaled_df: pd.DataFrame) -> Any:
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    n = len(raw_df)

    data["transaction"].x = torch.tensor(scaled_df.values, dtype=torch.float32)

    # transaction -> card
    card_source = raw_df["card1"] if "card1" in raw_df.columns else pd.Series(np.arange(n))
    card_codes, card_uniques = pd.factorize(card_source.fillna("missing").astype(str), sort=True)
    data["card"].x = torch.ones((len(card_uniques), 1), dtype=torch.float32)
    card_idx = np.vstack([np.arange(n, dtype=np.int64), card_codes.astype(np.int64)])
    card_edges = torch.from_numpy(card_idx)
    data[("transaction", "used", "card")].edge_index = card_edges
    data[("card", "used_by", "transaction")].edge_index = card_edges.flip(0)

    # transaction -> device
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

    # temporal decay
    if "TransactionDT" in raw_df.columns:
        tx_time = pd.to_numeric(raw_df["TransactionDT"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        age = tx_time.max() - tx_time
        age_scale = max(np.percentile(age, 95), 1.0)
        decay = np.exp(-age / age_scale).astype(np.float32)
    else:
        decay = np.ones(n, dtype=np.float32)
    data["transaction"].time_decay = torch.from_numpy(decay).unsqueeze(-1)

    return data


def predict_decision_tree(X_scaled: pd.DataFrame) -> np.ndarray:
    model = joblib.load(DT_MODEL_PATH)
    return model.predict_proba(X_scaled)[:, 1]


def predict_xgboost(X_scaled: pd.DataFrame) -> np.ndarray:
    X_array = np.ascontiguousarray(X_scaled.values.astype(np.float32))

    if XGB_BOOSTER_PATH.exists():
        booster = xgb.Booster()
        booster.load_model(str(XGB_BOOSTER_PATH))
        booster.set_param({"nthread": 1})
        dmat = xgb.DMatrix(X_array)
        return booster.predict(dmat)

    if XGB_MODEL_PATH.exists():
        model = joblib.load(XGB_MODEL_PATH)
        return model.predict_proba(X_array)[:, 1]

    raise FileNotFoundError(
        f"Missing XGBoost artifacts. Expected one of: {XGB_BOOSTER_PATH} or {XGB_MODEL_PATH}"
    )


def predict_hgnn(raw_df: pd.DataFrame, X_scaled: pd.DataFrame) -> np.ndarray:
    import torch
    from src.models import FraudHGNN

    checkpoint = torch.load(HGNN_ATT_TD_PATH, map_location="cpu", weights_only=False)
    expected_dim = int(checkpoint.get("input_dim", X_scaled.shape[1]))

    if X_scaled.shape[1] > expected_dim:
        # Checkpoint expects fewer features (legacy HGNN training schema).
        x_for_hgnn = X_scaled.iloc[:, :expected_dim].copy()
    elif X_scaled.shape[1] < expected_dim:
        pad = expected_dim - X_scaled.shape[1]
        pad_cols = [f"__hgnn_pad_{i}" for i in range(pad)]
        x_for_hgnn = X_scaled.copy()
        for col in pad_cols:
            x_for_hgnn[col] = 0.0
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


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading input: {in_path}")
    read_kwargs = {"nrows": args.max_rows} if args.max_rows is not None else {}
    df_in = pd.read_csv(in_path, **read_kwargs)

    if args.max_rows is not None:
        print(f"Applied pre-expansion row cap: {len(df_in):,}")

    resolved = expand_model_input(df_in, data_dir=args.data_dir)
    if resolved.note:
        print(resolved.note)
        print(f"Prepared rows: {len(resolved.frame):,}, cols: {resolved.frame.shape[1]:,}")
    df_in = resolved.frame

    if args.max_rows is not None and len(df_in) > args.max_rows:
        df_in = df_in.head(args.max_rows).copy()
        print(f"Applied post-expansion row cap: {len(df_in):,}")

    X_scaled, df_pre = preprocess_for_model(df_in)

    result = pd.DataFrame({
        ID_COL: df_in[ID_COL].values if ID_COL in df_in.columns else np.arange(len(df_in))
    })

    selected = [args.model] if args.model != "all" else ["decision_tree", "xgboost", "hgnn"]

    for model_name in selected:
        print(f"Running: {model_name}")
        try:
            if model_name == "decision_tree":
                proba = predict_decision_tree(X_scaled)
            elif model_name == "xgboost":
                proba = predict_xgboost(X_scaled)
            elif model_name == "hgnn":
                proba = predict_hgnn(df_pre, X_scaled)
            else:
                raise ValueError(model_name)
        except Exception as exc:
            print(f"Warning: {model_name} failed: {exc}")
            proba = np.full(len(result), np.nan)

        result[f"{model_name}_probability"] = proba
        pred = np.where(np.isnan(proba), -1, (proba >= 0.5).astype(int))
        result[f"{model_name}_prediction"] = pred

    result.to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")
    print(result.head(5))


if __name__ == "__main__":
    main()
