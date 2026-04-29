"""
models.py — Stage 4: Model Definitions
========================================
Defines the three model architectures used in PROJECT_P:
    1. Decision Tree  (sklearn — interpretable baseline)
    2. XGBoost        (gradient boosting — strong tabular baseline)
    3. FraudHGNN      (HGTConv — heterogeneous graph neural network) ⭐

Plus FocalLoss for handling extreme class imbalance in neural models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from src.config import DT_PARAMS, XGB_PARAMS, NN_PARAMS
from src.utils import get_logger

logger = get_logger("Models")


# ══════════════════════════════════════════════════════════════
# 1. DECISION TREE — Interpretable Baseline
# ══════════════════════════════════════════════════════════════

def build_decision_tree():
    """
    Build a Decision Tree classifier with cost-sensitive weighting.

    class_weight='balanced' automatically adjusts weights inversely
    proportional to class frequencies, penalizing misclassification
    of the minority (fraud) class more heavily.

    Returns
    -------
    DecisionTreeClassifier
        Configured but unfitted decision tree.
    """
    model = DecisionTreeClassifier(**DT_PARAMS)
    logger.info(
        f"  🌳 Decision Tree created "
        f"(max_depth={DT_PARAMS['max_depth']}, class_weight=balanced)"
    )
    return model


# ══════════════════════════════════════════════════════════════
# 2. XGBOOST — Gradient Boosted Trees
# ══════════════════════════════════════════════════════════════

def build_xgboost(scale_pos_weight: float = None):
    """
    Build an XGBoost classifier with cost-sensitive learning.

    scale_pos_weight = n_negative / n_positive handles class imbalance
    by increasing the cost of misclassifying positive (fraud) samples.

    Parameters
    ----------
    scale_pos_weight : float
        Ratio of negative to positive samples. If None, computed from data.

    Returns
    -------
    XGBClassifier
        Configured but unfitted XGBoost model.
    """
    params = XGB_PARAMS.copy()
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    model = XGBClassifier(**params)
    logger.info(
        f"  🚀 XGBoost created (n_estimators={params['n_estimators']}, "
        f"max_depth={params['max_depth']}, "
        f"scale_pos_weight={scale_pos_weight:.1f})"
    )
    return model


# ══════════════════════════════════════════════════════════════
# 3. HGNN-ATT-TD — Heterogeneous Graph Transformer ⭐
# ══════════════════════════════════════════════════════════════

class FraudHGNN(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) for fraud detection.

    Architecture:
        1. Node-specific Linear projections to unify dimensions
        2. HGTConv layers for attention-based message passing
           across distinct edge types (transaction→card, transaction→device)
        3. Temporal decay applied to transaction embeddings
        4. Linear classifier over 'transaction' nodes → Sigmoid

    This is the primary HGNN-ATT-TD model that leverages:
    - ATT: Multi-head attention in HGTConv across heterogeneous edge types
    - TD:  Temporal decay weighting from TransactionDT recency
    """

    def __init__(
        self,
        metadata,
        input_dim: int,
        hidden_dims: list = None,
        dropout_rates: list = None,
    ):
        super().__init__()
        hidden_dim = hidden_dims[0] if hidden_dims else NN_PARAMS["hidden_dims"][0]
        self.dropout_rate = dropout_rates[0] if dropout_rates else NN_PARAMS["dropout_rates"][0]

        # 1. Linear projection for each node type
        self.node_lin = nn.ModuleDict()
        for node_type in metadata[0]:
            in_d = input_dim if node_type == 'transaction' else 1
            self.node_lin[node_type] = nn.Linear(in_d, hidden_dim)

        # 2. HGTConv layers (2 layers, 2 attention heads)
        self.convs = nn.ModuleList()
        for _ in range(2):
            self.convs.append(
                pyg_nn.HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=2,
                )
            )

        # 3. Classification head
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x_dict, edge_index_dict, tx_time_decay=None):
        # Initial projection phase
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = F.gelu(self.node_lin[node_type](x))

        # Apply temporal decay to transaction embeddings
        if tx_time_decay is not None:
            decay = tx_time_decay
            if decay.dim() == 1:
                decay = decay.unsqueeze(-1)
            h_dict['transaction'] = h_dict['transaction'] * decay

        # HGT message passing with residual
        for conv in self.convs:
            out_dict = conv(h_dict, edge_index_dict)
            for node_type, h in out_dict.items():
                h_dict[node_type] = (
                    F.dropout(F.gelu(h), p=self.dropout_rate, training=self.training)
                    + h_dict[node_type]
                )

        logits = self.classifier(h_dict['transaction'])
        return logits

    def predict_proba(self, x_dict, edge_index_dict, tx_time_decay=None):
        """Get probability predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_dict, edge_index_dict, tx_time_decay=tx_time_decay)
            probs = torch.sigmoid(logits).squeeze()
        return probs


# ══════════════════════════════════════════════════════════════
# 4. FOCAL LOSS — For Extreme Class Imbalance
# ══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in neural networks.

    Down-weights well-classified examples and focuses training
    on hard, misclassified examples. Superior to simple class
    weighting for extreme imbalance ratios.

    Loss = -α * (1 - p_t)^γ * log(p_t)

    Parameters
    ----------
    gamma : float — Focusing parameter (higher = more focus on hard examples)
    alpha : float — Balancing factor for positive class
    """

    def __init__(self, gamma: float = None, alpha: float = None):
        super().__init__()
        self.gamma = gamma or NN_PARAMS["focal_gamma"]
        self.alpha = alpha or NN_PARAMS["focal_alpha"]

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).squeeze()
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits.squeeze(), targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_weight * focal_weight * bce
        return loss.mean()


# ══════════════════════════════════════════════════════════════
# 5. MODEL BUILDERS
# ══════════════════════════════════════════════════════════════

def build_neural_network(input_dim: int, metadata=None):
    """
    Build the FraudHGNN neural network with HGT.

    Returns
    -------
    tuple of (FraudHGNN, FocalLoss, torch.optim.AdamW)
    """
    if metadata is None:
        raise ValueError("metadata (node types, edge types) is required for FraudHGNN.")

    model = FraudHGNN(metadata, input_dim)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=NN_PARAMS["learning_rate"],
        weight_decay=NN_PARAMS["weight_decay"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  🧠 FraudHGNN created ({n_params:,} parameters)")
    logger.info(f"     Architecture: HGTConv × 2 layers, 2 attention heads")
    logger.info(f"     Loss: FocalLoss(γ={NN_PARAMS['focal_gamma']}, α={NN_PARAMS['focal_alpha']})")
    return model, criterion, optimizer
