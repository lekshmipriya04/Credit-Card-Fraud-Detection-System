"""
hgnn_utils.py — Heterogeneous Graph Construction & Training Utilities
======================================================================
Functions for building multi-view dense heterogeneous graphs with
temporal decay, memory-efficient batch processing, and staged
training helpers for large datasets.
"""

import gc
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Optional
from src.utils import get_logger, Timer

logger = get_logger("HGNN_Utils")


# ══════════════════════════════════════════════════════════════
# 1. DENSE GRAPH CONSTRUCTION (Memory-Optimized)
# ══════════════════════════════════════════════════════════════

def build_dense_graphs(
    df: pd.DataFrame,
    relations: List[str],
    max_nodes: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Build dense heterogeneous adjacency matrices with temporal decay.

    For each relation, creates:
      - Adjacency matrix: connects nodes sharing the relation value
      - Time distance matrix: absolute temporal gaps between transactions

    Parameters
    ----------
    df : pd.DataFrame
        Transaction DataFrame with 'TransactionDT' and relation columns
    relations : List[str]
        Relation columns (e.g., ['card1', 'addr1', 'P_emaildomain'])
    max_nodes : int, optional
        Max nodes (random sample if exceeded for memory safety)
    device : str
        Target device ('cpu', 'cuda', 'mps')

    Returns
    -------
    tuple of (adj_list, dts_list) — lists of torch.Tensor
    """
    num_original = len(df)

    if max_nodes is not None and num_original > max_nodes:
        logger.info(f"  📉 Sampling {max_nodes} from {num_original} nodes (memory limit)")
        df = df.sample(max_nodes, random_state=42).reset_index(drop=True)

    num_nodes = len(df)
    adj_list = []
    dts_list = []

    for rel in relations:
        logger.info(f"  🔗 Building {rel} adjacency ({num_nodes} nodes)...")

        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        dt_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

        groups = df.groupby(rel).indices

        for group_val, g_idx in groups.items():
            if len(g_idx) > 1:
                g_idx_tensor = torch.tensor(g_idx, dtype=torch.long)

                rows = g_idx_tensor.repeat_interleave(len(g_idx))
                cols = g_idx_tensor.repeat(len(g_idx))

                adj[rows, cols] = 1.0

                times = torch.tensor(
                    df.iloc[g_idx]['TransactionDT'].values, dtype=torch.float32
                )
                temporal_gaps = torch.abs(
                    times.repeat_interleave(len(g_idx)) - times.repeat(len(g_idx))
                )
                dt_matrix[rows, cols] = temporal_gaps

        # Row-normalize adjacency
        row_sums = adj.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-6)
        adj = adj / row_sums

        if device != "cpu":
            adj = adj.to(device)
            dt_matrix = dt_matrix.to(device)

        adj_list.append(adj)
        dts_list.append(dt_matrix)

        del adj, dt_matrix
        gc.collect()

    logger.info(f"  ✅ Built {len(relations)} relational views ({num_nodes} nodes)")
    return adj_list, dts_list


# ══════════════════════════════════════════════════════════════
# 2. BATCH PROCESSING
# ══════════════════════════════════════════════════════════════

def create_node_batches(num_nodes: int, batch_size: int = 2000) -> List[Tuple[int, int]]:
    """Create (start, end) batches of node indices for staged training."""
    batches = []
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        batches.append((start, end))
    return batches


def get_batch_indices_for_gradient_accumulation(
    num_nodes: int, accumulation_steps: int = 4
) -> List[List[int]]:
    """Create node-index batches for gradient accumulation."""
    step_size = max(1, num_nodes // accumulation_steps)
    batches = []
    for i in range(accumulation_steps):
        start = i * step_size
        end = (
            min((i + 1) * step_size, num_nodes)
            if i < accumulation_steps - 1
            else num_nodes
        )
        batches.append(list(range(start, end)))
    return batches


# ══════════════════════════════════════════════════════════════
# 3. MEMORY PROFILING
# ══════════════════════════════════════════════════════════════

def estimate_graph_memory(num_nodes: int, num_relations: int) -> float:
    """
    Estimate GPU memory needed for dense graphs (in GB).

    Each adjacency/time matrix: N² × 4 bytes (float32).
    """
    bytes_per_matrix = (num_nodes ** 2) * 4
    total_bytes = bytes_per_matrix * num_relations * 2  # adj + time
    return total_bytes / (1024 ** 3)


def get_safe_max_nodes(gpu_memory_gb: int = 80, safety_margin: float = 0.7) -> int:
    """
    Calculate safe max nodes for dense HGNN given GPU memory.

    N² × 3 relations × 2 matrices × 4 bytes <= safety_margin × GPU_GB
    """
    num_relations = 3
    available_bytes = gpu_memory_gb * (1024 ** 3) * safety_margin
    max_n_squared = available_bytes / (num_relations * 2 * 4)
    return int(np.sqrt(max_n_squared))


def log_graph_info(x_shape: tuple, adj_list: List[torch.Tensor], dts_list: List[torch.Tensor]):
    """Log graph construction statistics."""
    num_nodes, num_features = x_shape
    num_relations = len(adj_list)

    logger.info(f"\n  📊 Graph Construction Summary:")
    logger.info(f"     Nodes: {num_nodes:,}")
    logger.info(f"     Features: {num_features}")
    logger.info(f"     Relations: {num_relations}")

    total_edges = 0
    for i, adj in enumerate(adj_list):
        num_edges = int((adj > 0).sum().item())
        total_edges += num_edges
        logger.info(f"     Relation {i}: {num_edges:,} edges")

    logger.info(f"     Est. Memory: {estimate_graph_memory(num_nodes, num_relations):.2f} GB")


# ══════════════════════════════════════════════════════════════
# 4. TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════

def prepare_training_data(
    y: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare labels and masks for dense HGNN training.

    Returns
    -------
    tuple of (y_tensor, train_mask, val_mask)
    """
    n = len(y)
    y_tensor = torch.tensor(y, dtype=torch.long)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True

    logger.info(f"  ✅ Prepared: {len(train_indices)} train, {len(val_indices)} val")
    return y_tensor, train_mask, val_mask


def focal_loss_weight(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Apply focal loss weighting to imbalanced classification."""
    probs = torch.sigmoid(logits).squeeze()
    targets_float = targets.float()

    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(), targets_float, reduction="none"
    )

    p_t = probs * targets_float + (1 - probs) * (1 - targets_float)
    focal_weight = (1 - p_t) ** gamma
    alpha_weight = alpha * targets_float + (1 - alpha) * (1 - targets_float)

    return (alpha_weight * focal_weight * bce).mean()
