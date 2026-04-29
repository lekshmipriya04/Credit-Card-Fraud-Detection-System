import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def drop_missing_value_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drops columns with a ratio of missing values greater than the threshold.

    Args:
        df (pd.DataFrame): The input dataframe.
        threshold (float): The threshold for missing value ratio.

    Returns:
        pd.DataFrame: The dataframe with columns dropped.
    """
    logging.info(f"Dropping columns with more than {threshold * 100}% missing values.")
    missing_values = df.isnull().sum() / len(df)
    cols_to_drop = missing_values[missing_values > threshold].index
    df = df.drop(columns=cols_to_drop)
    logging.info(f"Dropped columns: {list(cols_to_drop)}")
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Label Encoding on categorical features.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with categorical features encoded.
    """
    logging.info("Performing Label Encoding on categorical features.")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle missing values before fitting the encoder
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        
    logging.info(f"Encoded categorical columns: {list(categorical_cols)}")
    return df
