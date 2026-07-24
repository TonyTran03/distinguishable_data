# models/gmm.py
import numpy as np
from sklearn.mixture import GaussianMixture


# Table S5: class-specific component counts selected by minimum AIC among
# K = 2, 3, 4, and 5 full-covariance candidate models.
AIC_COMPONENTS_BY_DATASET = {
    "Breast Cancer": (5, 5),
    "Diabetes": (5, 5),
    "HIV": (2, 3),
}

# Class-specific component counts selected by minimum BIC over K = 2,...,5.
BIC_COMPONENTS_BY_DATASET = {
    "Breast Cancer": (2, 2),
    "Diabetes": (4, 2),
    "HIV": (2, 2),
}


def sample_gmm(
    X,
    y,
    n0,
    n1,
    seed=42,
    n_components=2,
    reg_covar=1e-4,
    covariance_type="full",
):
    """
    Fit a per-class GMM on real data and sample synthetic observations.
    ----------
    X            : np.ndarray, shape (n, p)
    y            : np.ndarray, shape (n,), values in {0, 1}
    n0, n1       : number of synthetic samples per class
    seed         : random state
    n_components : desired number of GMM components. An integer applies the
                   same K to both classes; a (class_0, class_1) pair supports
                   the class-specific AIC selections reported in Table S5;
                   automatically clamped to min(n_components, n_class_samples // 2)
                   so we never fit more components than the data supports
    reg_covar    : regularisation added to the diagonal of each covariance matrix;
                   prevents singular matrices when classes are small or features
                   are nearly collinear (default 1e-4 is safe for standardised data)
    """

    X = np.asarray(X, dtype=np.float64)

    X0 = X[y == 0]
    X1 = X[y == 1]

    if np.isscalar(n_components):
        requested_k0 = requested_k1 = int(n_components)
    else:
        requested_k0, requested_k1 = map(int, n_components)

    # Clamp K so every component can have at least two observations.
    k0 = max(1, min(requested_k0, len(X0) // 2))
    k1 = max(1, min(requested_k1, len(X1) // 2))

    gmm0 = GaussianMixture(
        n_components=k0,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=seed,
    )
    gmm1 = GaussianMixture(
        n_components=k1,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        random_state=seed,
    )

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
