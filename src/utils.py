"""
utils.py — Shared Utilities for PROJECT_P
==========================================
Logging setup, plot theme, Timer context manager,
memory optimization, and common formatting helpers.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import COLORS, DPI


# ─────────────────────── Logging ───────────────────────
def get_logger(name: str) -> logging.Logger:
    """Create a standardized logger with a readable console formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# ─────────────────────── Plot Style ───────────────────────
def set_plot_style():
    """Apply a professional dark theme to all matplotlib plots."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg_dark"],
        "axes.facecolor":    COLORS["bg_card"],
        "axes.edgecolor":    COLORS["grid"],
        "axes.labelcolor":   COLORS["text"],
        "text.color":        COLORS["text"],
        "xtick.color":       COLORS["text"],
        "ytick.color":       COLORS["text"],
        "grid.color":        COLORS["grid"],
        "grid.alpha":        0.3,
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.labelsize":    12,
        "figure.dpi":        DPI,
        "savefig.dpi":       DPI,
        "savefig.bbox":      "tight",
        "savefig.facecolor": COLORS["bg_dark"],
    })


def save_figure(fig, filepath: str, tight: bool = True):
    """Save a matplotlib figure and close it to free memory."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(
        filepath,
        bbox_inches="tight" if tight else None,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)


# ─────────────────────── Formatting ───────────────────────
def format_number(n: int) -> str:
    """Format large integers with commas: 590540 → '590,540'."""
    return f"{n:,}"


def format_pct(val: float, decimals: int = 2) -> str:
    """Format a float as percentage: 0.035 → '3.50%'."""
    return f"{val * 100:.{decimals}f}%"


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


# ─────────────────────── Timer ───────────────────────
class Timer:
    """Context manager for timing code blocks with automatic logging."""

    def __init__(self, description: str = "", logger=None):
        self.description = description
        self.logger = logger or get_logger("Timer")

    def __enter__(self):
        self.start = time.time()
        if self.description:
            self.logger.info(f"⏳ Starting: {self.description}")
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.description:
            self.logger.info(
                f"✅ Finished: {self.description} ({format_duration(self.elapsed)})"
            )


# ─────────────────────── Memory Optimization ───────────────────────
def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce DataFrame memory by downcasting numeric columns.

    Essential for the large IEEE-CIS dataset (~1.7 GB raw).
    Downcasts integers and floats to the smallest type that
    preserves all values without information loss.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        series = df[col]
        col_type = series.dtype

        if pd.api.types.is_bool_dtype(col_type):
            df[col] = series.astype(np.int8)
            continue

        if pd.api.types.is_integer_dtype(col_type):
            try:
                c_min = series.min(skipna=True)
                c_max = series.max(skipna=True)
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = series.astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = series.astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = series.astype(np.int32)
            except (TypeError, ValueError, OverflowError):
                continue

        elif pd.api.types.is_float_dtype(col_type):
            try:
                c_min = series.min(skipna=True)
                c_max = series.max(skipna=True)
                if np.isfinite(c_min) and np.isfinite(c_max):
                    if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                        df[col] = series.astype(np.float32)
            except (TypeError, ValueError, OverflowError):
                continue

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    reduction = 100 * (start_mem - end_mem) / max(start_mem, 1)
    print(f"  Memory: {start_mem:.1f} MB → {end_mem:.1f} MB ({reduction:.1f}% reduction)")
    return df
