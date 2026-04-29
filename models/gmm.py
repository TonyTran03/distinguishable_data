# models/gmm.py
import numpy as np
from sklearn.mixture import GaussianMixture


def sample_gmm(X, y, n0, n1, seed=42, n_components=2, reg_covar=1e-4):
    """
    Fit a per-class GMM on real data and sample synthetic observations.
    ----------
    X            : np.ndarray, shape (n, p)
    y            : np.ndarray, shape (n,), values in {0, 1}
    n0, n1       : number of synthetic samples per class
    seed         : random state
    n_components : desired number of GMM components per class;
                   automatically clamped to min(n_components, n_class_samples // 2)
                   so we never fit more components than the data supports
    reg_covar    : regularisation added to the diagonal of each covariance matrix;
                   prevents singular matrices when classes are small or features
                   are nearly collinear (default 1e-4 is safe for standardised data)
    """

    X = np.asarray(X, dtype=np.float64)

    X0 = X[y == 0]
    X1 = X[y == 1]

    # clamp n_components so every component can have at least 2 samples
    k0 = max(1, min(n_components, len(X0) // 2))
    k1 = max(1, min(n_components, len(X1) // 2))

    gmm0 = GaussianMixture(n_components=k0, reg_covar=reg_covar, random_state=seed)
    gmm1 = GaussianMixture(n_components=k1, reg_covar=reg_covar, random_state=seed)

    gmm0.fit(X0)
    gmm1.fit(X1)

    X_syn0, _ = gmm0.sample(n0)
    X_syn1, _ = gmm1.sample(n1)

    X_syn = np.vstack([X_syn0, X_syn1]).astype(np.float32)
    y_syn = np.concatenate([
        np.zeros(n0, dtype=int),
        np.ones(n1,  dtype=int),
    ])

    return X_syn, y_syn