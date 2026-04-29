"""
preprocessing.py — Stage 3: Data Preprocessing & Cleaning
============================================================
Handles missing values, label encoding, outlier capping, scaling,
stratified train/val/test splitting, and heterogeneous graph construction
for the HGNN-ATT-TD model.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.config import (
    TARGET_COL, ID_COL, DROP_COLS, CATEGORICAL_COLS,
    MISSING_THRESHOLD, AMT_CAP_PERCENTILE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE,
    SCALER_PATH, LABEL_ENCODERS_PATH, FEATURE_NAMES_PATH, MODEL_DIR,
)
from src.utils import get_logger, Timer, format_number
import torch
from torch_geometric.data import HeteroData

logger = get_logger("Preprocessing")


def _should_build_hetero_graph() -> bool:
    """Return whether heterogeneous graph construction is enabled."""
    return os.environ.get("BUILD_HETERO_GRAPH", "1") == "1"


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values with a multi-strategy approach:
      1. Drop columns with >50% missing (too sparse to be useful)
      2. Median imputation for numeric columns
      3. 'Unknown' fill for categorical columns
    """
    n_before = df.shape[1]

    # Step 1: Drop high-missing columns
    missing_pct = df.isnull().mean()
    high_missing_cols = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    high_missing_cols = [c for c in high_missing_cols if c != TARGET_COL]
    df = df.drop(columns=high_missing_cols)
    logger.info(f"  Dropped {len(high_missing_cols)} columns with >{MISSING_THRESHOLD*100:.0f}% missing")

    # Step 2: Numeric imputation (median)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Step 3: Categorical imputation ("Unknown")
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    n_after = df.shape[1]
    logger.info(f"  Features: {n_before} → {n_after} (removed {n_before - n_after})")
    logger.info(f"  Remaining nulls: {df.isnull().sum().sum()}")
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme outliers in TransactionAmt at the 99th percentile.
    Prevents extremely high fraud amounts from destabilizing training.
    """
    if "TransactionAmt" in df.columns:
        cap_val = df["TransactionAmt"].quantile(AMT_CAP_PERCENTILE / 100)
        n_capped = (df["TransactionAmt"] > cap_val).sum()
        df["TransactionAmt"] = df["TransactionAmt"].clip(upper=cap_val)
        logger.info(f"  Capped {format_number(n_capped)} outliers in TransactionAmt at {cap_val:.2f}")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple:
    """
    Label-encode all categorical columns.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        Encoded DataFrame and dictionary of fitted LabelEncoders.
    """
    label_encoders = {}
    actual_cats = [c for c in CATEGORICAL_COLS if c in df.columns]

    for col in actual_cats:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    logger.info(f"  Encoded {len(actual_cats)} categorical columns")
    return df, label_encoders


def split_data(df: pd.DataFrame):
    """
    Stratified train/val/test split (70/15/15).

    Stratification preserves the ~3.5% fraud rate in every split,
    which is critical for reliable evaluation under class imbalance.

    Returns
    -------
    tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=[TARGET_COL] + drop_existing)
    y = df[TARGET_COL].astype(int)

    test_size = TEST_RATIO
    val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted,
        stratify=y_trainval, random_state=RANDOM_STATE
    )

    logger.info(f"  Train: {format_number(len(X_train))} (fraud: {y_train.sum()})")
    logger.info(f"  Val:   {format_number(len(X_val))} (fraud: {y_val.sum()})")
    logger.info(f"  Test:  {format_number(len(X_test))} (fraud: {y_test.sum()})")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """
    Standardize features using StandardScaler fitted ONLY on training data.
    This prevents data leakage from val/test into training statistics.
    """
    scaler = StandardScaler()
    feature_names = list(X_train.columns)

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=feature_names, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    logger.info(f"  Scaler saved → {SCALER_PATH}")
    logger.info(f"  {len(feature_names)} feature names saved")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def build_hetero_graph(
    df: pd.DataFrame,
    X_scaled: pd.DataFrame,
    y: pd.Series,
    train_idx,
    val_idx,
    test_idx,
) -> HeteroData:
    """
    Build a Heterogeneous Graph for the HGNN-ATT-TD model.

    Node types:
      - transaction: Features = scaled tabular data
      - card:        Features = dummy (1-dim placeholder)
      - device:      Features = dummy (1-dim placeholder)

    Edge types:
      - (transaction, used, card) + reverse
      - (transaction, on, device) + reverse

    Also attaches temporal decay weights from TransactionDT recency.
    """
    logger.info("  Constructing Heterogeneous Graph (HGT Structure)...")
    data = HeteroData()

    # --- 1. TRANSACTION NODES ---
    num_transactions = len(X_scaled)
    data['transaction'].x = torch.FloatTensor(X_scaled.values)
    data['transaction'].y = torch.LongTensor(y.values)

    # Temporal decay: newer transactions → higher weight
    if 'TransactionDT' in df.columns:
        tx_time = pd.to_numeric(df['TransactionDT'], errors='coerce').fillna(0).values.astype(np.float32)
    else:
        tx_time = np.arange(len(df), dtype=np.float32)

    age = tx_time.max() - tx_time
    age_scale = np.maximum(np.percentile(age, 95), 1.0)
    decay_np = np.exp(-age / age_scale).astype(np.float32)
    data['transaction'].time_decay = torch.from_numpy(decay_np).unsqueeze(-1)

    # Train/Val/Test Masks
    train_mask = torch.zeros(num_transactions, dtype=torch.bool)
    val_mask   = torch.zeros(num_transactions, dtype=torch.bool)
    test_mask  = torch.zeros(num_transactions, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True
    data['transaction'].train_mask = train_mask
    data['transaction'].val_mask   = val_mask
    data['transaction'].test_mask  = test_mask

    tx_indices = np.arange(num_transactions)

    # --- 2. CARD NODES & EDGES ---
    unique_cards, card_idx = np.unique(df['card1'].values, return_inverse=True)
    num_cards = len(unique_cards)
    data['card'].x = torch.ones((num_cards, 1))

    card_edge_index = torch.tensor([tx_indices, card_idx], dtype=torch.long)
    data['transaction', 'used', 'card'].edge_index = card_edge_index
    data['card', 'used_by', 'transaction'].edge_index = card_edge_index.flip([0])

    # --- 3. DEVICE NODES & EDGES ---
    if 'DeviceInfo' in df.columns:
        device_vals = df['DeviceInfo'].values
    else:
        device_vals = (
            df['DeviceType'].values if 'DeviceType' in df.columns
            else np.zeros(num_transactions)
        )

    unique_devices, device_idx = np.unique(device_vals, return_inverse=True)
    num_devices = len(unique_devices)
    data['device'].x = torch.ones((num_devices, 1))

    device_edge_index = torch.tensor([tx_indices, device_idx], dtype=torch.long)
    data['transaction', 'on', 'device'].edge_index = device_edge_index
    data['device', 'hosts', 'transaction'].edge_index = device_edge_index.flip([0])

    logger.info(
        f"    Graph: {num_transactions} Transactions | "
        f"{num_cards} Cards | {num_devices} Devices"
    )
    return data


def run_preprocessing_pipeline(df: pd.DataFrame):
    """
    Execute the full preprocessing pipeline end-to-end.

    Steps:
      1. Handle missing values (drop high-missing, impute rest)
      2. Cap outliers in TransactionAmt
      3. Encode categorical features
      4. Stratified train/val/test split (70/15/15)
      5. Scale features (fit on train only)
      6. Build heterogeneous graph for HGNN-ATT-TD

    Returns
    -------
    dict with keys:
        X_train, X_val, X_test, y_train, y_val, y_test,
        scaler, label_encoders, feature_names, hetero_data
    """
    logger.info("=" * 60)
    logger.info("     PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    with Timer("Handling missing values", logger):
        df = handle_missing_values(df)

    with Timer("Capping outliers", logger):
        df = cap_outliers(df)

    with Timer("Encoding categoricals", logger):
        df, label_encoders = encode_categoricals(df)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(label_encoders, LABEL_ENCODERS_PATH)

    with Timer("Splitting data (70/15/15 stratified)", logger):
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    with Timer("Scaling features", logger):
        X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    # Reconstruct full aligned dataframe for graph construction
    X_full = pd.concat([X_train, X_val, X_test])
    y_full = pd.concat([y_train, y_val, y_test])
    train_indices = np.arange(len(X_train))
    val_indices   = np.arange(len(X_train), len(X_train) + len(X_val))
    test_indices  = np.arange(len(X_train) + len(X_val), len(X_full))

    df_aligned = df.loc[X_full.index]

    hetero_data = None
    if _should_build_hetero_graph():
        with Timer("Building Heterogeneous Graph", logger):
            hetero_data = build_hetero_graph(
                df_aligned, X_full, y_full,
                train_indices, val_indices, test_indices,
            )
    else:
        logger.info("  Skipping heterogeneous graph construction (BUILD_HETERO_GRAPH=0)")

    feature_names = list(X_train.columns)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler, "label_encoders": label_encoders,
        "feature_names": feature_names,
        "hetero_data": hetero_data,
    }


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    df = load_raw_data(sample_frac=0.1)
    result = run_preprocessing_pipeline(df)
    print(f"\nFinal feature count: {len(result['feature_names'])}")
    print(f"Training set shape: {result['X_train'].shape}")
