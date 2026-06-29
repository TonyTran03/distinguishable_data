from __future__ import annotations

from typing import Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

from util.transformation import make_transform
import pyreadr

def load_rdata_xy_names(
    rdata_path: Path,
    x_key: str = "x",
    y_key: str = "y"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads X, y, and feature names from an .RData file.

    Expected:
      - x: (N, D)
      - y: (N,) or (N,1) with binary {0,1}
    """
    obj = pyreadr.read_r(str(rdata_path))

    if x_key not in obj or y_key not in obj:
        raise KeyError(
            f"Missing keys. Found: {list(obj.keys())}. Expected '{x_key}' and '{y_key}'."
        )

    x_df = obj[x_key]
    y_df = obj[y_key]

    X = np.asarray(x_df).astype(np.float32)
    y = np.asarray(y_df).reshape(-1).astype(np.int64)
    feature_names = list(x_df.columns)

    uniq = np.unique(y)
    if not set(uniq).issubset({0, 1}):
        raise ValueError(f"y must be binary {{0,1}}. Found unique values: {uniq}")

    return X, y, feature_names

def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    batch_size: int,
    seed: int,
    num_classes: int = 2,
    x_transform: str = "none", 
    torch_generator: torch.Generator | None = None,
) -> Tuple[DataLoader, DataLoader, StandardScaler]:

    # Split first 
    y = np.asarray(y)
    classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    if np.min(class_counts) < 2:
        raise ValueError(
            "Stratified train/validation split requires at least two samples "
            f"per class; got counts {dict(zip(classes.tolist(), class_counts.tolist()))}."
        )
    if isinstance(test_size, float):
        val_size = int(np.ceil(test_size * len(y)))
    else:
        val_size = int(test_size)
    val_size = max(n_classes, val_size)
    val_size = min(val_size, len(y) - n_classes)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )

    # Instantiate transform
    transform = make_transform(x_transform)

    # Apply transform in raw space 
    X_train = transform.forward(X_train)
    X_val   = transform.forward(X_val)

    # Standardize in transformed space 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)

    # Tensors 
    X_train_t = torch.tensor(X_train)
    X_val_t   = torch.tensor(X_val)

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t   = torch.tensor(y_val, dtype=torch.long)

    c_train = F.one_hot(y_train_t, num_classes=num_classes).float()
    c_val   = F.one_hot(y_val_t,   num_classes=num_classes).float()

    train_loader = DataLoader(
        TensorDataset(X_train_t, c_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch_generator,
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, c_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, scaler
