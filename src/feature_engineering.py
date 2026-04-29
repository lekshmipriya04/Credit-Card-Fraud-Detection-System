"""
feature_engineering.py — Stage 2: Feature Engineering & Selection
==================================================================
Creates domain-specific features from raw IEEE-CIS data, applies PCA
to reduce the 339 anonymized V-features, and provides mutual-information
based feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from src.config import TARGET_COL, PCA_COMPONENTS, RANDOM_STATE
from src.utils import get_logger, Timer

logger = get_logger("FeatureEng")


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from TransactionDT.

    TransactionDT is a timedelta (seconds) from an undisclosed reference.
    We derive hour-of-day and day-of-week with cyclical encoding so that
    hour 23 is close to hour 0 in feature space.
    """
    if "TransactionDT" not in df.columns:
        return df

    df["hour_of_day"] = (df["TransactionDT"] / 3600 % 24).astype(int)
    df["day_of_week"] = (df["TransactionDT"] / 86400 % 7).astype(int)

    # Cyclical encoding — prevents discontinuity at boundaries
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    logger.info("  ✅ Time features created (hour_of_day, day_of_week, cyclical)")
    return df


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from TransactionAmt:
    - Log-transformed amount (reduces right skewness)
    - Decimal part (fraudsters often use round dollar amounts)
    - Is-round flag (binary indicator)
    """
    if "TransactionAmt" not in df.columns:
        return df

    df["TransactionAmt_log"]      = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_decimal"]  = (df["TransactionAmt"] % 1).round(4)
    df["TransactionAmt_is_round"] = (df["TransactionAmt"] % 1 == 0).astype(int)

    logger.info("  ✅ Amount features created (log, decimal, is_round)")
    return df


def create_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create card interaction features:
    - card1 × card2 frequency encoding (unique card identifier proxy)
    - Mean amount per card1 ratio (spending pattern baseline)
    """
    if "card1" in df.columns and "card2" in df.columns:
        df["card1_card2"] = df["card1"].astype(str) + "_" + df["card2"].astype(str)
        freq = df["card1_card2"].value_counts()
        df["card1_card2_freq"] = df["card1_card2"].map(freq)
        df = df.drop(columns=["card1_card2"])

    if "card1" in df.columns and "TransactionAmt" in df.columns:
        card1_mean = df.groupby("card1")["TransactionAmt"].transform("mean")
        df["amt_card1_ratio"] = df["TransactionAmt"] / (card1_mean + 1e-6)

    logger.info("  ✅ Card interaction features created")
    return df


def create_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create email-based features:
    - Email provider match (purchaser vs recipient domain)
    - Email provider frequency
    """
    p_col, r_col = "P_emaildomain", "R_emaildomain"
    if p_col in df.columns and r_col in df.columns:
        df["email_match"] = (df[p_col] == df[r_col]).astype(int)
    if p_col in df.columns:
        freq = df[p_col].value_counts()
        df["email_freq"] = df[p_col].map(freq).fillna(0)

    logger.info("  ✅ Email features created")
    return df


def create_address_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create address-based risk features."""
    if "addr1" in df.columns and TARGET_COL in df.columns:
        addr_fraud = df.groupby("addr1")[TARGET_COL].transform("mean")
        df["addr1_fraud_rate"] = addr_fraud

    if "dist1" in df.columns:
        df["dist1_log"] = np.log1p(df["dist1"].fillna(0))

    logger.info("  ✅ Address features created")
    return df


def create_c_d_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate C and D feature groups into summary statistics.
    C-features are counting variables, D-features are time-deltas.
    """
    c_cols = [c for c in df.columns if c.startswith("C") and c[1:].isdigit()]
    d_cols = [c for c in df.columns if c.startswith("D") and c[1:].isdigit()]

    if c_cols:
        df["C_sum"]  = df[c_cols].sum(axis=1)
        df["C_mean"] = df[c_cols].mean(axis=1)
        df["C_std"]  = df[c_cols].std(axis=1).fillna(0)
        df["C_max"]  = df[c_cols].max(axis=1)

    if d_cols:
        df["D_sum"]       = df[d_cols].fillna(0).sum(axis=1)
        df["D_mean"]      = df[d_cols].fillna(0).mean(axis=1)
        df["D_nullcount"] = df[d_cols].isnull().sum(axis=1)

    logger.info(
        f"  ✅ C-feature aggregates ({len(c_cols)} cols) and "
        f"D-feature aggregates ({len(d_cols)} cols) created"
    )
    return df


def apply_pca_to_v_features(df: pd.DataFrame, n_components: int = PCA_COMPONENTS):
    """
    Apply PCA to the V1–V339 feature block.

    The 339 V-features are anonymized, highly correlated Vesta features.
    PCA reduces them to the top principal components while preserving
    most of the variance.

    Returns
    -------
    tuple of (pd.DataFrame, PCA or None)
    """
    v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]

    if len(v_cols) < n_components:
        logger.info(f"  ⚠️ Only {len(v_cols)} V-features found, skipping PCA")
        return df, None

    v_data = df[v_cols].fillna(0).values

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    v_pca = pca.fit_transform(v_data)

    df = df.drop(columns=v_cols)
    pca_cols = [f"V_PCA_{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(v_pca, columns=pca_cols, index=df.index)
    df = pd.concat([df, pca_df], axis=1)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(
        f"  ✅ PCA: {len(v_cols)} V-features → {n_components} components "
        f"({explained:.1%} variance explained)"
    )
    return df, pca


def select_features_mi(X: pd.DataFrame, y: pd.Series, top_k: int = 100):
    """
    Select top-K features using Mutual Information with the target.

    MI captures non-linear dependencies, which is more informative
    than simple Pearson correlation for fraud detection.

    Returns
    -------
    tuple of (list[str], pd.Series)
        Selected feature names and full MI scores.
    """
    logger.info(f"  Computing mutual information for {X.shape[1]} features...")
    mi_scores = mutual_info_classif(X.fillna(0), y, random_state=RANDOM_STATE, n_neighbors=5)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    selected = mi_series.head(top_k).index.tolist()
    logger.info(f"  ✅ Selected top {top_k} features by Mutual Information")
    logger.info(f"  Top 10: {selected[:10]}")

    return selected, mi_series


def drop_high_correlation(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove one of each pair of features with correlation > threshold."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
    drop_cols = [c for c in drop_cols if c != TARGET_COL]
    df = df.drop(columns=drop_cols, errors="ignore")
    logger.info(f"  ✅ Dropped {len(drop_cols)} highly correlated features (r > {threshold})")
    return df


def run_feature_engineering(df: pd.DataFrame):
    """
    Execute the full feature engineering pipeline.

    Steps:
      1. Create time, amount, card, email, address, C/D aggregate features
      2. Apply PCA to V1–V339 (339 → 50 components)
      3. Drop highly correlated features (> 0.95)

    Returns
    -------
    tuple of (pd.DataFrame, PCA or None)
    """
    logger.info("=" * 60)
    logger.info("     FEATURE ENGINEERING")
    logger.info("=" * 60)

    with Timer("Creating time features", logger):
        df = create_time_features(df)

    with Timer("Creating amount features", logger):
        df = create_amount_features(df)

    with Timer("Creating card features", logger):
        df = create_card_features(df)

    with Timer("Creating email features", logger):
        df = create_email_features(df)

    with Timer("Creating address features", logger):
        df = create_address_features(df)

    with Timer("Creating C/D aggregate features", logger):
        df = create_c_d_aggregates(df)

    with Timer("Applying PCA to V-features", logger):
        df, pca = apply_pca_to_v_features(df)

    with Timer("Dropping highly correlated features", logger):
        df = drop_high_correlation(df)

    logger.info(f"  📊 Final feature count: {df.shape[1]} columns")
    return df, pca


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    df = load_raw_data(sample_frac=0.1)
    df, pca = run_feature_engineering(df)
    print(f"Final shape: {df.shape}")
