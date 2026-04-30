"""
Helpers for normalizing uploaded inference inputs into the full feature shape
expected by the trained fraud models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import (
    ID_COL,
    TARGET_COL,
    TRAIN_TRANSACTION,
    TRAIN_IDENTITY,
    TEST_TRANSACTION,
    TEST_IDENTITY,
)

TRANSACTION_SIGNAL_COLS = {
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "addr1",
    "dist1",
    "C1",
    "D1",
}
IDENTITY_SIGNAL_COLS = {"DeviceType", "DeviceInfo"}
IDENTITY_PREFIX = "id_"


@dataclass(frozen=True)
class DatasetPaths:
    train_transaction: Path
    train_identity: Path
    test_transaction: Path
    test_identity: Path


@dataclass(frozen=True)
class ResolvedInput:
    frame: pd.DataFrame
    mode: str
    split: str | None
    note: str | None


def resolve_dataset_paths(data_dir: str | None = None) -> DatasetPaths:
    if data_dir:
        base_dir = Path(data_dir).expanduser().resolve()
        return DatasetPaths(
            train_transaction=base_dir / "train_transaction.csv",
            train_identity=base_dir / "train_identity.csv",
            test_transaction=base_dir / "test_transaction.csv",
            test_identity=base_dir / "test_identity.csv",
        )
    return DatasetPaths(
        train_transaction=TRAIN_TRANSACTION,
        train_identity=TRAIN_IDENTITY,
        test_transaction=TEST_TRANSACTION,
        test_identity=TEST_IDENTITY,
    )


def _has_transaction_features(df: pd.DataFrame) -> bool:
    return any(col in df.columns for col in TRANSACTION_SIGNAL_COLS)


def _has_identity_features(df: pd.DataFrame) -> bool:
    if any(col in df.columns for col in IDENTITY_SIGNAL_COLS):
        return True
    return any(str(col).startswith(IDENTITY_PREFIX) for col in df.columns)


def _sample_ids(df: pd.DataFrame, limit: int = 1000) -> pd.Series:
    if ID_COL not in df.columns:
        return pd.Series(dtype="int64")
    return df[ID_COL].dropna().head(limit)


def _count_overlap(sample_ids: pd.Series, csv_path: Path) -> int:
    if sample_ids.empty or not csv_path.exists():
        return -1
    ref_ids = pd.read_csv(csv_path, usecols=[ID_COL])[ID_COL]
    return int(ref_ids.isin(sample_ids).sum())


def _choose_split(sample_ids: pd.Series, train_path: Path, test_path: Path) -> str:
    train_overlap = _count_overlap(sample_ids, train_path)
    test_overlap = _count_overlap(sample_ids, test_path)
    if test_overlap >= train_overlap:
        return "test"
    return "train"


def _missing_source_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def expand_model_input(df: pd.DataFrame, data_dir: str | None = None) -> ResolvedInput:
    if ID_COL not in df.columns:
        return ResolvedInput(df, mode="passthrough", split=None, note=None)

    paths = resolve_dataset_paths(data_dir)
    non_id_cols = [col for col in df.columns if col not in {ID_COL, TARGET_COL}]
    sample_ids = _sample_ids(df)

    if not non_id_cols:
        split = _choose_split(sample_ids, paths.train_transaction, paths.test_transaction)
        tx_path = getattr(paths, f"{split}_transaction")
        id_path = getattr(paths, f"{split}_identity")
        missing = _missing_source_paths([tx_path, id_path])
        if missing:
            missing_names = ", ".join(path.name for path in missing)
            return ResolvedInput(
                df,
                mode="id_only_missing_sources",
                split=split,
                note=(
                    f"Detected id-only input, but missing source files for auto-expansion: {missing_names}. "
                    "Using uploaded input as-is."
                ),
            )
        tx_df = pd.read_csv(tx_path)
        id_df = pd.read_csv(id_path)
        full_df = pd.merge(tx_df, id_df, on=ID_COL, how="left")
        merged = pd.merge(df[[ID_COL]], full_df, on=ID_COL, how="left")
        return ResolvedInput(
            merged,
            mode="id_only",
            split=split,
            note=f"Detected id-only input. Expanded with {split}_transaction.csv + {split}_identity.csv by TransactionID.",
        )

    has_tx = _has_transaction_features(df)
    has_id = _has_identity_features(df)
    if has_tx and has_id:
        return ResolvedInput(df, mode="full_features", split=None, note=None)

    if has_id and not has_tx:
        split = _choose_split(sample_ids, paths.train_transaction, paths.test_transaction)
        tx_path = getattr(paths, f"{split}_transaction")
        missing = _missing_source_paths([tx_path])
        if missing:
            return ResolvedInput(
                df,
                mode="identity_only_missing_sources",
                split=split,
                note=(
                    f"Detected identity-only input, but missing source file for auto-expansion: {tx_path.name}. "
                    "Using uploaded input as-is."
                ),
            )
        tx_df = pd.read_csv(tx_path)
        merged = pd.merge(df, tx_df, on=ID_COL, how="left")
        return ResolvedInput(
            merged,
            mode="identity_only",
            split=split,
            note=f"Detected identity-only input. Joined {split}_transaction.csv by TransactionID before scoring.",
        )

    if has_tx and not has_id:
        split = _choose_split(sample_ids, paths.train_identity, paths.test_identity)
        id_path = getattr(paths, f"{split}_identity")
        missing = _missing_source_paths([id_path])
        if missing:
            return ResolvedInput(
                df,
                mode="transaction_only_missing_sources",
                split=split,
                note=(
                    f"Detected transaction-only input, but missing source file for auto-expansion: {id_path.name}. "
                    "Using uploaded input as-is."
                ),
            )
        id_df = pd.read_csv(id_path)
        merged = pd.merge(df, id_df, on=ID_COL, how="left")
        return ResolvedInput(
            merged,
            mode="transaction_only",
            split=split,
            note=f"Detected transaction-only input. Joined {split}_identity.csv by TransactionID before scoring.",
        )

    return ResolvedInput(df, mode="passthrough", split=None, note=None)
