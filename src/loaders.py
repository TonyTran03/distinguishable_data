from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder

from src.data import load_rdata_xy_names


def _to_numpy_X(X):
    """Convert X to a float32 numpy array regardless of input type."""
    if hasattr(X, "values"):
        X = X.values
    return np.asarray(X, dtype=np.float32)


def _to_numpy_y(y):
    """Convert y to a 1-D int numpy array with values in {0, 1}."""
    if hasattr(y, "values"):
        y = y.values
    y = np.asarray(y).reshape(-1)

    if y.dtype.kind in {"i", "u", "b"}:
        return y.astype(int)

    le = LabelEncoder()
    return le.fit_transform(y).astype(int)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_breast():
    raw = load_breast_cancer()
    return {
        "dataset": "breast_cancer",
        "category": "clinical_tabular",
        "X": _to_numpy_X(raw.data),
        "y": _to_numpy_y(raw.target),
        "feature_names": list(raw.feature_names),
    }


def load_diabetes():
    from sklearn.datasets import fetch_openml

    raw = fetch_openml("diabetes", version=1, as_frame=False)
    return {
        "dataset": "diabetes",
        "category": "metabolic_tabular",
        "X": _to_numpy_X(raw.data),
        "y": _to_numpy_y(raw.target),
        "feature_names": list(raw.feature_names),
    }


def load_HIV():
    X, y, feature_names = load_rdata_xy_names(_repo_root() / "data" / "allSyntheticData.RData")
    return {
        "dataset": "HIV",
        "category": "clinical_tabular",
        "X": _to_numpy_X(X),
        "y": _to_numpy_y(y),
        "feature_names": feature_names,
    }
