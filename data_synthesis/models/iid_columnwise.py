import numpy as np


def sample_columnwise(X, y, n0, n1, seed=42):
    """
    Column-wise bootstrap: each feature is resampled independently.

    For each class, generates n samples where feature j of sample i
    is drawn (with replacement) from all real values of feature j
    within that class. This breaks inter-feature correlations.
    """
    rng = np.random.default_rng(seed)
    p = X.shape[1]

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    X0 = np.empty((n0, p))
    for j in range(p):
        donors = X[idx0, j]
        X0[:, j] = rng.choice(donors, size=n0, replace=True)

    X1 = np.empty((n1, p))
    for j in range(p):
        donors = X[idx1, j]
        X1[:, j] = rng.choice(donors, size=n1, replace=True)

    X_syn = np.vstack([X0, X1])
    y_syn = np.concatenate([
        np.zeros(n0, dtype=int),
        np.ones(n1, dtype=int),
    ])

    return X_syn, y_syn
