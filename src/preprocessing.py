"""
preprocessing.py
Feature normalization and selection for IoMT network traffic data.

Pipeline:
  1. Z-score normalization: x' = (x - mu) / sigma
  2. One-hot encoding for categorical features (protocol_type, service, flag)
  3. Correlation-based feature selection (drop redundant features above threshold)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize(X: np.ndarray):
    """Z-score normalize all features."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def encode_categorical(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """One-hot encode categorical columns e.g. protocol_type, service, flag."""
    return pd.get_dummies(df, columns=cat_cols)


def remove_correlated(X: pd.DataFrame, threshold: float = 0.95):
    """Drop features with pairwise correlation above threshold."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > threshold)]
    return X.drop(columns=drop_cols), drop_cols


def preprocess(df: pd.DataFrame, label_col: str = 'label',
               cat_cols: list = None):
    if cat_cols is None:
        cat_cols = ['protocol_type', 'service', 'flag']
    df = encode_categorical(df, cat_cols)
    X = df.drop(columns=[label_col])
    y = (df[label_col] != 'normal').astype(int).values  # binary: 0=normal, 1=attack
    X, dropped = remove_correlated(X)
    X_norm, scaler = normalize(X.values)
    print(f"Features after selection: {X_norm.shape[1]} (dropped {len(dropped)})")
    return X_norm, y, scaler


if __name__ == '__main__':
    print("Pass your IDS dataset CSV to preprocess().")
    print("Expected columns: protocol_type, service, flag, label, + numeric features")
