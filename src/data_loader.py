"""
data_loader.py — Stage 1: Data Loading
========================================
Loads and merges the IEEE-CIS Fraud Detection dataset.

The dataset has two tables:
  - train_transaction.csv  (590,540 rows × 394 cols)
  - train_identity.csv     (144,233 rows × 41 cols)

They are joined on TransactionID via LEFT JOIN.
Not all transactions have identity records (~24% do).
"""

import pandas as pd
import numpy as np
from src.config import (
    TRAIN_TRANSACTION, TRAIN_IDENTITY,
    TEST_TRANSACTION, TEST_IDENTITY,
    TARGET_COL, ID_COL,
)
from src.utils import get_logger, reduce_memory_usage, Timer, format_number, format_pct

logger = get_logger("DataLoader")


def load_raw_data(sample_frac: float = None) -> pd.DataFrame:
    """
    Load and merge the IEEE-CIS Fraud Detection dataset.

    Parameters
    ----------
    sample_frac : float, optional
        If provided (0 < frac < 1), randomly sample that fraction
        of data after merging. Useful for fast experimentation.
        Default: None (use full dataset).

    Returns
    -------
    pd.DataFrame
        Merged dataset with all transaction + identity features.
    """
    with Timer("Loading transaction data", logger):
        df_trans = pd.read_csv(TRAIN_TRANSACTION)
        logger.info(f"  Transactions: {format_number(len(df_trans))} rows × {df_trans.shape[1]} cols")

    with Timer("Loading identity data", logger):
        df_id = pd.read_csv(TRAIN_IDENTITY)
        logger.info(f"  Identity:     {format_number(len(df_id))} rows × {df_id.shape[1]} cols")

    with Timer("Merging datasets on TransactionID (LEFT JOIN)", logger):
        df = pd.merge(df_trans, df_id, on=ID_COL, how="left")
        logger.info(f"  Merged:       {format_number(len(df))} rows × {df.shape[1]} cols")
        matched = df_id[ID_COL].isin(df_trans[ID_COL]).sum()
        logger.info(f"  Identity coverage: {format_pct(matched / len(df_trans))} of transactions")

    with Timer("Optimizing memory usage", logger):
        df = reduce_memory_usage(df)

    if sample_frac is not None and 0 < sample_frac < 1:
        n_before = len(df)
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        logger.info(
            f"  Sampled: {format_number(n_before)} → {format_number(len(df))} rows "
            f"({format_pct(sample_frac)})"
        )

    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive dataset statistics.

    Returns
    -------
    dict
        Keys: n_rows, n_cols, n_fraud, n_legit, fraud_rate, imbalance_ratio,
              cols_no_missing, cols_some_missing, cols_high_missing,
              n_numeric, n_categorical, memory_mb.
    """
    n_rows, n_cols = df.shape
    n_fraud = int(df[TARGET_COL].sum())
    n_legit = n_rows - n_fraud
    fraud_rate = n_fraud / n_rows
    imbalance_ratio = n_legit / max(n_fraud, 1)

    missing_pct = df.isnull().mean()
    cols_no_missing   = int((missing_pct == 0).sum())
    cols_some_missing = int(((missing_pct > 0) & (missing_pct < 0.5)).sum())
    cols_high_missing = int((missing_pct >= 0.5).sum())

    n_numeric     = len(df.select_dtypes(include=[np.number]).columns)
    n_categorical = len(df.select_dtypes(include=["object"]).columns)

    summary = {
        "n_rows":             n_rows,
        "n_cols":             n_cols,
        "n_fraud":            n_fraud,
        "n_legit":            n_legit,
        "fraud_rate":         fraud_rate,
        "imbalance_ratio":    imbalance_ratio,
        "cols_no_missing":    cols_no_missing,
        "cols_some_missing":  cols_some_missing,
        "cols_high_missing":  cols_high_missing,
        "n_numeric":          n_numeric,
        "n_categorical":      n_categorical,
        "memory_mb":          df.memory_usage(deep=True).sum() / 1024 ** 2,
    }

    logger.info("=" * 60)
    logger.info("         DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total samples:     {format_number(n_rows)}")
    logger.info(f"  Total features:    {format_number(n_cols)}")
    logger.info(f"  Fraud (1):         {format_number(n_fraud)} ({format_pct(fraud_rate)})")
    logger.info(f"  Legitimate (0):    {format_number(n_legit)} ({format_pct(1 - fraud_rate)})")
    logger.info(f"  Imbalance ratio:   1:{imbalance_ratio:.1f}")
    logger.info(f"  Numeric features:  {n_numeric}")
    logger.info(f"  Categorical:       {n_categorical}")
    logger.info(f"  Cols no missing:   {cols_no_missing}")
    logger.info(f"  Cols high missing: {cols_high_missing} (will be dropped)")
    logger.info(f"  Memory usage:      {summary['memory_mb']:.1f} MB")
    logger.info("=" * 60)

    return summary


if __name__ == "__main__":
    df = load_raw_data(sample_frac=0.1)
    summary = get_dataset_summary(df)
    print(f"\nFraud rate: {format_pct(summary['fraud_rate'])}")
