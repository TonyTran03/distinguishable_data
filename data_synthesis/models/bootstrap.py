import numpy as np

def sample_bootstrap(X, y, n0, n1, seed=42):
    rng = np.random.default_rng(seed)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    take0 = rng.choice(idx0, size=n0, replace=True)
    take1 = rng.choice(idx1, size=n1, replace=True)

    X0 = X[take0]
    X1 = X[take1]

    X_syn = np.vstack([X0, X1])
    y_syn = np.concatenate([
        np.zeros(n0, dtype=int),
        np.ones(n1, dtype=int),
    ])

    return X_syn, y_syn