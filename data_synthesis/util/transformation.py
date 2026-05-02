# transformation.py
from __future__ import annotations
import numpy as np

class XTransform:
    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def inverse(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class IdentityTransform(XTransform):
    def forward(self, X: np.ndarray) -> np.ndarray:
        return X
    def inverse(self, X: np.ndarray) -> np.ndarray:
        return X

class Log1pTransform(XTransform):
    """
    Applies log(1 + x).
    """
    def forward(self, X: np.ndarray) -> np.ndarray:
        if np.min(X) < -1.0:
            raise ValueError(f"log1p invalid: min(X) < -1")
        return np.log1p(X)

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return np.expm1(X)

def make_transform(name: str) -> XTransform:
    name = name.lower()
    if name in ("none", "identity"):
        return IdentityTransform()
    if name in ("log1p", "log"):
        return Log1pTransform()
    raise ValueError(f"Unknown transform: {name}")