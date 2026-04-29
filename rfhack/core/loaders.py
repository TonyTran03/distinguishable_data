"""
Dataset loaders for synthetic data evaluation.
    {
        "dataset": str   — short name used as a key throughout
        "category": str  — domain label
        "X": np.ndarray   — shape (n, p), float32
        "y": np.ndarray   — shape (n,), int {0, 1}
        "feature_names": list[str] — length p
    }

All loaders:
    - X is a 2-D float32 numpy array (never a DataFrame)
    - y is a 1-D int numpy array with values in {0, 1}
    - feature_names is a plain Python list of strings
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder


def _to_numpy_X(X):
    """Convert X to a float32 numpy array regardless of input type."""
    if hasattr(X, "values"):          # pandas DataFrame / Series
        X = X.values
    return np.asarray(X, dtype=np.float32)


def _to_numpy_y(y):
    """Convert y to a 1-D int numpy array with values in {0, 1}."""
    if hasattr(y, "values"):
        y = y.values
    y = np.asarray(y)

    if y.dtype.kind in {"i", "u", "b"}:
        return y.astype(int)

    # string / object labels → encode to 0/1
    le = LabelEncoder()
    return le.fit_transform(y).astype(int)


# ── Breast Cancer (sklearn)

def load_breast():
    raw = load_breast_cancer()
    return {
        "dataset":       "breast_cancer",
        "category":      "clinical_tabular",
        "X":             _to_numpy_X(raw.data),
        "y":             _to_numpy_y(raw.target),
        "feature_names": list(raw.feature_names),
    }


# ── Pima Indians Diabetes (OpenML) 

def load_diabetes():
    from sklearn.datasets import fetch_openml
    raw = fetch_openml("diabetes", version=1, as_frame=False)
    return {
        "dataset":       "diabetes",
        "category":      "metabolic_tabular",
        "X":             _to_numpy_X(raw.data),
        "y":             _to_numpy_y(raw.target),
        "feature_names": list(raw.feature_names),
    }


# ── HIV (RData) 

def load_HIV():
    import pyreadr
    raw          = pyreadr.read_r("rfhack/data/allSyntheticData.RData")
    X_df         = raw["x"]
    y_raw        = raw["y"]
    feature_names = list(X_df.columns)

    return {
        "dataset":       "HIV",           # fixed: was incorrectly "diabetes"
        "category":      "clinical_tabular",
        "X":             _to_numpy_X(X_df),
        "y":             _to_numpy_y(y_raw),
        "feature_names": feature_names,
    }