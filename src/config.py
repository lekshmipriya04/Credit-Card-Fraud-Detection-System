"""config.py - Central Configuration
====================================
Single source of truth for all paths, hyperparameters, feature
definitions, and constants used across every module in the project.

Environment Variables:
  IEEE_FRAUD_DATA_DIR: Path to IEEE-CIS fraud dataset folder (train_transaction.csv, etc.)
                       Defaults to ./data if not set.
"""

import os
from pathlib import Path

# ─────────────────────────────── Paths ───────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

# IEEE-CIS dataset location (read-only source files)
# Set DATA_DIR environment variable to point to your IEEE-CIS fraud dataset folder.
# If not set, defaults to ./data (you can place dataset there).
DATA_DIR = Path(os.getenv("IEEE_FRAUD_DATA_DIR", PROJECT_ROOT / "data"))

# Project output directories (auto-created at runtime)
MODEL_DIR = PROJECT_ROOT / "models"
EVAL_DIR  = str(PROJECT_ROOT / "outputs")

# Create directories on import
for _d in [MODEL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ───────────────────── Raw Data Files ─────────────────────
TRAIN_TRANSACTION = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY    = DATA_DIR / "train_identity.csv"
TEST_TRANSACTION  = DATA_DIR / "test_transaction.csv"
TEST_IDENTITY     = DATA_DIR / "test_identity.csv"

# ───────────────────── Model Artifact Paths ─────────────────────
DT_MODEL_PATH       = MODEL_DIR / "decision_tree.pkl"
XGB_MODEL_PATH      = MODEL_DIR / "xgboost_model.pkl"
XGB_BOOSTER_PATH    = MODEL_DIR / "xgboost_booster.json"
HGNN_ATT_TD_PATH    = MODEL_DIR / "hgnn_att_td.pt"
SCALER_PATH         = MODEL_DIR / "scaler.pkl"
FEATURE_NAMES_PATH  = MODEL_DIR / "feature_names.pkl"
LABEL_ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"

# ───────────────────── Dataset Constants ─────────────────────
TARGET_COL   = "isFraud"
ID_COL       = "TransactionID"
RANDOM_STATE = 42

# 70 / 15 / 15 stratified split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ───────────────────── Feature Groups ─────────────────────
CATEGORICAL_COLS = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
    "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18",
    "id_19", "id_20", "id_21", "id_22", "id_23", "id_24", "id_25",
    "id_26", "id_27", "id_28", "id_29", "id_30", "id_31", "id_32",
    "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
]

# Columns to drop before modelling (ID leaks features or non-predictive)
DROP_COLS         = [ID_COL, "TransactionDT"]
MISSING_THRESHOLD = 0.50   # Drop columns with > 50% missing
AMT_CAP_PERCENTILE = 99    # Cap TransactionAmt outliers at 99th percentile

# ───────────────────── Model Hyperparameters ─────────────────────

# --- Decision Tree ---
DT_PARAMS = {
    "max_depth":        12,
    "min_samples_split": 50,
    "min_samples_leaf":  20,
    "class_weight":     "balanced",
    "criterion":        "gini",
    "random_state":     RANDOM_STATE,
}

# --- XGBoost ---
XGB_PARAMS = {
    "n_estimators":    500,
    "max_depth":        8,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "eval_metric":    "aucpr",
    "tree_method":    "auto",
    "random_state":    RANDOM_STATE,
    "n_jobs":          1,  # Force 1 thread on macOS to prevent libomp EXC_BAD_ACCESS crashes
    "verbosity":       0,
}

# --- HGNN (HGTConv-based FraudHGNN) ---
# NOTE: hidden_dims kept small (64) for Mac-friendly full-batch CPU training.
# On a GPU server, increase to [256, 128, 64] for better performance.
NN_PARAMS = {
    "hidden_dims":    [64, 32],
    "dropout_rates":  [0.3, 0.2],
    "learning_rate":  1e-3,
    "weight_decay":   1e-4,
    "batch_size":     2048,
    "max_epochs":     30,
    "patience":       8,
    "focal_gamma":    2.0,
    "focal_alpha":    0.75,
}

# --- DenseHGNN_ATT_TD (reference variant) ---
HGNN_PARAMS = {
    "hidden_dims":    [128, 64],
    "num_layers":     2,
    "dropout":        0.2,
    "learning_rate":  1e-3,
    "weight_decay":   1e-4,
    "batch_size":     512,
    "max_epochs":     15,
    "patience":       5,
    "focal_gamma":    2.0,
    "focal_alpha":    0.75,
    "early_stopping": True,
}

# ───────────────────── SMOTE ─────────────────────
SMOTE_PARAMS = {
    "sampling_strategy": 0.5,
    "k_neighbors":       5,
    "random_state":      RANDOM_STATE,
}

# ───────────────────── Device & Memory ─────────────────────
import torch as _torch

def _detect_device() -> str:
    if _torch.cuda.is_available():
        return "cuda"
    if hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _detect_device()
DENSE_HGNN_MAX_NODES_CUDA = 12_000
DENSE_HGNN_MAX_NODES_CPU  = 3_000

# ───────────────────── Feature Engineering ─────────────────────
PCA_COMPONENTS = 50
N_FOLDS        = 5

# ───────────────────── Cost-Benefit Model ─────────────────────
COST_FN = 500   # False Negative: missed fraud (high)
COST_FP = 10    # False Positive: false alarm (low)
COST_TP = -500  # True Positive: caught fraud (saves loss)
COST_TN = 0     # True Negative: correctly cleared

# ───────────────────── Visualization ─────────────────────
COLORS = {
    "fraud":     "#FF4B4B",
    "legit":     "#00D26A",
    "primary":   "#6C63FF",
    "secondary": "#FF6B6B",
    "accent":    "#4ECDC4",
    "bg_dark":   "#0E1117",
    "bg_card":   "#1A1F2E",
    "text":      "#FAFAFA",
    "grid":      "#2D3748",
}

FIGSIZE_LARGE  = (16, 10)
FIGSIZE_MEDIUM = (12, 7)
FIGSIZE_SMALL  = (8, 5)
DPI = 150
