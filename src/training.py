"""
training.py — Stage 5: Model Training Pipelines
==================================================
Implements training loops for all three model architectures:
  1. Decision Tree   — sklearn + SMOTE
  2. XGBoost         — early stopping + cost-sensitive
  3. HGNN-ATT-TD     — full-batch HGTConv with FocalLoss ⭐

Handles SMOTE oversampling, device selection (MPS/CUDA/CPU),
early stopping, and model checkpointing.
"""

import os
import copy
import sys
import gc
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import joblib
import torch
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from xgboost.core import XGBoostError

import src.config as config

SMOTE_PARAMS     = config.SMOTE_PARAMS
NN_PARAMS        = config.NN_PARAMS
N_FOLDS          = config.N_FOLDS
RANDOM_STATE     = config.RANDOM_STATE
DT_MODEL_PATH    = config.DT_MODEL_PATH
XGB_MODEL_PATH   = config.XGB_MODEL_PATH
XGB_BOOSTER_PATH = config.XGB_BOOSTER_PATH
HGNN_ATT_TD_PATH = config.HGNN_ATT_TD_PATH
MODEL_DIR        = config.MODEL_DIR

from src.models import build_decision_tree, build_xgboost, build_neural_network
from src.utils import get_logger, Timer, format_number

logger = get_logger("Training")


# ─────────────────────── Device Selection ───────────────────────

def _select_safe_device(prefer_mps: bool = True) -> torch.device:
    """Auto-select best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        try:
            probe = torch.empty((1,), device="cuda")
            del probe
            return torch.device("cuda")
        except RuntimeError:
            logger.warning("  CUDA reported available but busy. Falling back.")
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _clear_cuda_state() -> None:
    """Release CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# ─────────────────────── SMOTE Oversampling ───────────────────────

def apply_smote(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to balance the training set.

    SMOTE generates synthetic fraud samples by interpolating between
    existing minority samples and their k-nearest neighbors.
    Applied ONLY to training set — never to validation/test.
    """
    n_before_pos = int(y_train.sum() if hasattr(y_train, 'sum') else np.sum(y_train))
    n_before_neg = len(y_train) - n_before_pos

    smote = SMOTE(**SMOTE_PARAMS)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    n_after_pos = int(y_resampled.sum())
    n_after_neg = len(y_resampled) - n_after_pos

    logger.info(f"  SMOTE Resampling:")
    logger.info(
        f"    Before: {format_number(n_before_neg)} legit, "
        f"{format_number(n_before_pos)} fraud (1:{n_before_neg / max(n_before_pos, 1):.1f})"
    )
    logger.info(
        f"    After:  {format_number(n_after_neg)} legit, "
        f"{format_number(n_after_pos)} fraud (1:{n_after_neg / max(n_after_pos, 1):.1f})"
    )
    logger.info(f"    Synthetic samples: {format_number(n_after_pos - n_before_pos)}")

    return X_resampled, y_resampled


# ══════════════════════════════════════════════════════════════
# 1. DECISION TREE TRAINING
# ══════════════════════════════════════════════════════════════

def train_decision_tree(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    use_smote: bool = True,
) -> Any:
    """
    Train Decision Tree with cost-sensitive learning + optional SMOTE.
    """
    logger.info("=" * 50)
    logger.info("  🌳 Training Decision Tree")
    logger.info("=" * 50)

    if use_smote:
        X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train, y_train

    with Timer("Decision Tree training", logger):
        model = build_decision_tree()
        model.fit(X_train_sm, y_train_sm)

    val_score = model.score(X_val, y_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    val_preds = model.predict(X_val)
    val_f1  = f1_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_proba)

    logger.info(f"  Val Accuracy: {val_score:.4f}")
    logger.info(f"  Val F1-Score: {val_f1:.4f}")
    logger.info(f"  Val ROC-AUC:  {val_auc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, DT_MODEL_PATH)
    logger.info(f"  Model saved → {DT_MODEL_PATH}")

    return model


# ══════════════════════════════════════════════════════════════
# 2. XGBOOST TRAINING
# ══════════════════════════════════════════════════════════════

def _force_classic_dmatrix(model: Any) -> None:
    """Force sklearn XGBoost wrapper to use classic DMatrix."""
    model._create_dmatrix = _create_classic_dmatrix


def _create_classic_dmatrix(ref=None, **kwargs):
    """Module-level DMatrix factory (picklable)."""
    return xgb.DMatrix(**kwargs)


def train_xgboost(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    use_smote: bool = True,
) -> Any:
    """
    Train XGBoost with early stopping and cost-sensitive learning.
    """
    logger.info("=" * 50)
    logger.info("  🚀 Training XGBoost")
    logger.info("=" * 50)

    # macOS thread safety
    if sys.platform == "darwin":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    def _to_numpy_2d(x):
        arr = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return np.ascontiguousarray(arr)

    def _to_numpy_1d(y):
        arr = y.values if isinstance(y, pd.Series) else np.asarray(y)
        return np.ascontiguousarray(arr.astype(np.float32))

    y_train_arr = _to_numpy_1d(y_train)
    y_val_arr   = _to_numpy_1d(y_val)
    X_val_arr   = _to_numpy_2d(X_val)

    n_pos = int(y_train_arr.sum())
    n_neg = len(y_train_arr) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    if use_smote:
        X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    else:
        X_train_sm, y_train_sm = X_train, y_train

    X_train_arr = _to_numpy_2d(X_train_sm)
    y_train_arr_sm = _to_numpy_1d(y_train_sm)

    with Timer("XGBoost training", logger):
        model = build_xgboost(scale_pos_weight=scale_pos_weight)
        xgb_n_jobs = int(os.environ.get("XGB_N_JOBS", "1" if sys.platform == "darwin" else "-1"))
        model.set_params(tree_method="hist", n_jobs=xgb_n_jobs)
        model.gpu_id = None
        _force_classic_dmatrix(model)

        try:
            model.fit(
                X_train_arr, y_train_arr_sm,
                eval_set=[(X_val_arr, y_val_arr)],
                verbose=False,
            )
        except XGBoostError as exc:
            raise RuntimeError(f"XGBoost training failed: {exc}") from exc

    val_proba = model.predict_proba(X_val_arr)[:, 1]
    val_preds = model.predict(X_val_arr)
    val_f1    = f1_score(y_val_arr, val_preds)
    val_auc   = roc_auc_score(y_val_arr, val_proba)
    val_auprc = average_precision_score(y_val_arr, val_proba)

    logger.info(f"  Val F1-Score: {val_f1:.4f}")
    logger.info(f"  Val ROC-AUC:  {val_auc:.4f}")
    logger.info(f"  Val AUPRC:    {val_auprc:.4f}")

    # Extract training history
    training_history = {'train_losses': [], 'val_aucs': []}
    try:
        # XGBClassifier stores evaluation results in evals_result_ after fitting
        if hasattr(model, 'evals_result_'):
            evals_result = model.evals_result_
            logger.info(f"  Evals result keys: {evals_result.keys()}")
            if 'validation_0' in evals_result:
                # Get the metric names available
                metrics_available = list(evals_result['validation_0'].keys())
                logger.info(f"  Available metrics: {metrics_available}")
                
                training_history['train_losses'] = evals_result['validation_0'].get('logloss', [])
                training_history['val_aucs'] = evals_result['validation_0'].get('auc', [])
                
                logger.info(f"  Extracted {len(training_history['train_losses'])} loss values")
                logger.info(f"  Extracted {len(training_history['val_aucs'])} AUC values")
    except Exception as e:
        logger.warning(f"  Could not extract training history: {e}")
        import traceback
        logger.warning(traceback.format_exc())

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, XGB_MODEL_PATH)
    logger.info(f"  Model saved → {XGB_MODEL_PATH}")
    model.get_booster().save_model(XGB_BOOSTER_PATH)
    logger.info(f"  Booster saved → {XGB_BOOSTER_PATH}")

    return model, training_history


# ══════════════════════════════════════════════════════════════
# 3. HGNN-ATT-TD TRAINING ⭐
# ══════════════════════════════════════════════════════════════

def train_neural_network(
    X_train_df: Union[pd.DataFrame, np.ndarray],
    y_train_df: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    hetero_data: Optional[Any] = None,
    use_smote: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, List[float]]]:
    """
    Train HGNN-ATT-TD (FraudHGNN with HGTConv) using full-batch training.

    Training features:
      - FocalLoss for extreme class imbalance
      - OneCycleLR learning rate scheduler
      - Early stopping with patience=10
      - Best model checkpointing based on validation AUC
      - Temporal decay applied to transaction embeddings
    """
    logger.info("=" * 50)
    logger.info("  🧠 Training Neural Network (HGNN-ATT-TD)")
    logger.info("=" * 50)

    if hetero_data is None:
        raise RuntimeError(
            "hetero_data is required for HGNN-ATT-TD training. "
            "Ensure BUILD_HETERO_GRAPH=1 during preprocessing."
        )

    # PyG HGTConv uses pyg::segment_matmul which is NOT implemented on MPS.
    # Force CPU for HGNN training. CUDA works if available.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # safety net
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("  Device: cuda")
    else:
        device = torch.device("cpu")
        logger.info("  Device: cpu (MPS not supported by PyG HGTConv)")

    if use_smote:
        logger.warning(
            "  SMOTE is incompatible with GNN topology. "
            "Using FocalLoss for imbalance handling instead."
        )

    hetero_data = hetero_data.to(device)
    input_dim: int = hetero_data['transaction'].x.shape[1]

    model, criterion, optimizer = build_neural_network(input_dim, metadata=hetero_data.metadata())
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=NN_PARAMS["learning_rate"] * 10,
        epochs=NN_PARAMS["max_epochs"],
        steps_per_epoch=1,
    )

    best_val_auc: float = 0.0
    patience_counter: int = 0
    train_losses: List[float] = []
    val_aucs: List[float] = []

    y_train_t = hetero_data['transaction'].y[hetero_data['transaction'].train_mask]
    y_val_t   = hetero_data['transaction'].y[hetero_data['transaction'].val_mask]
    tx_decay  = getattr(hetero_data['transaction'], 'time_decay', None)

    with Timer("HGNN-ATT-TD training (full-batch)", logger):
        for epoch in range(NN_PARAMS["max_epochs"]):
            # ── Train ──
            model.train()
            optimizer.zero_grad()

            logits = model(
                hetero_data.x_dict,
                hetero_data.edge_index_dict,
                tx_time_decay=tx_decay,
            )
            train_logits = logits[hetero_data['transaction'].train_mask]
            loss = criterion(train_logits, y_train_t)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            avg_loss = float(loss.item())
            train_losses.append(avg_loss)

            # ── Validate ──
            model.eval()
            with torch.no_grad():
                val_logits_full = model(
                    hetero_data.x_dict,
                    hetero_data.edge_index_dict,
                    tx_time_decay=tx_decay,
                )
                val_logits = val_logits_full[hetero_data['transaction'].val_mask]
                val_proba = torch.sigmoid(val_logits).squeeze().cpu().numpy()

            val_auc = float(roc_auc_score(y_val_t.cpu().numpy(), val_proba))
            val_aucs.append(val_auc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1:3d}/{NN_PARAMS['max_epochs']} | "
                    f"Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}"
                )

            # ── Checkpoint best model ──
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                os.makedirs(MODEL_DIR, exist_ok=True)
                torch.save({
                    "model_type":       "hgnn_att_td",
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "input_dim":        input_dim,
                    "hidden_dims":      NN_PARAMS["hidden_dims"],
                    "dropout_rates":    NN_PARAMS["dropout_rates"],
                    "best_val_auc":     best_val_auc,
                    "epoch":            epoch + 1,
                }, HGNN_ATT_TD_PATH)
            else:
                patience_counter += 1
                if patience_counter >= NN_PARAMS["patience"]:
                    logger.info(
                        f"  ⏹ Early stopping at epoch {epoch+1} "
                        f"(patience={NN_PARAMS['patience']})"
                    )
                    break

    # ── Load best checkpoint ──
    checkpoint = torch.load(HGNN_ATT_TD_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ── Final validation metrics ──
    model.eval()
    with torch.no_grad():
        final_logits = model(
            hetero_data.x_dict,
            hetero_data.edge_index_dict,
            tx_time_decay=tx_decay,
        )
        val_logits = final_logits[hetero_data['transaction'].val_mask]
        val_proba = torch.sigmoid(val_logits).squeeze().cpu().numpy()

    val_preds = (val_proba >= 0.5).astype(int)
    val_f1    = float(f1_score(y_val_t.cpu().numpy(), val_preds))
    val_auprc = float(average_precision_score(y_val_t.cpu().numpy(), val_proba))

    logger.info(f"  Best Val AUC:  {best_val_auc:.4f}")
    logger.info(f"  Val F1-Score:  {val_f1:.4f}")
    logger.info(f"  Val AUPRC:     {val_auprc:.4f}")
    logger.info(f"  Model saved → {HGNN_ATT_TD_PATH}")

    return model, {"train_losses": train_losses, "val_aucs": val_aucs}
