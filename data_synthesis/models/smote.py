"""Small, dependency-free SMOTE samplers for tabular binary data."""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def _interpolate_class(X_class, n_samples, rng, k_neighbors=5, groups=None):
    X_class = np.asarray(X_class, dtype=np.float64)
    if len(X_class) == 0:
        raise ValueError("SMOTE cannot sample an empty class.")
    if len(X_class) == 1:
        return np.repeat(X_class, n_samples, axis=0)

    output = np.empty((n_samples, X_class.shape[1]), dtype=np.float64)
    for row in range(n_samples):
        anchor = int(rng.integers(len(X_class)))
        candidates = np.arange(len(X_class))
        if groups is not None:
            same_group = candidates[groups == groups[anchor]]
            if len(same_group) > 1:
                candidates = same_group

        pool = X_class[candidates]
        k = min(int(k_neighbors) + 1, len(pool))
        neighbors = NearestNeighbors(n_neighbors=k).fit(pool)
        local = neighbors.kneighbors(
            X_class[anchor].reshape(1, -1), return_distance=False
        )[0]
        neighbor_candidates = candidates[local]
        neighbor_candidates = neighbor_candidates[neighbor_candidates != anchor]
        if len(neighbor_candidates) == 0:
            neighbor = int(rng.choice(candidates))
        else:
            neighbor = int(rng.choice(neighbor_candidates))
        weight = float(rng.random())
        output[row] = X_class[anchor] + weight * (
            X_class[neighbor] - X_class[anchor]
        )
    return output


def _sample_by_class(X, y, n0, n1, seed, sampler):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    X0 = sampler(X[y == 0], int(n0), rng)
    X1 = sampler(X[y == 1], int(n1), rng)
    return (
        np.vstack([X0, X1]).astype(np.float32),
        np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)],
    )


def sample_smote(X, y, n0, n1, seed=42, k_neighbors=5):
    """Generate the requested number of observations within each class."""

    return _sample_by_class(
        X,
        y,
        n0,
        n1,
        seed,
        lambda values, count, rng: _interpolate_class(
            values, count, rng, k_neighbors=k_neighbors
        ),
    )


def sample_gmm_guided_smote(
    X,
    y,
    n0,
    n1,
    seed=42,
    n_components=3,
    k_neighbors=5,
    reg_covar=1e-4,
):
    """SMOTE interpolation restricted to a fitted GMM component when possible."""

    def sample_one_class(values, count, rng):
        components = max(1, min(int(n_components), len(values) // 2))
        gmm = GaussianMixture(
            n_components=components,
            covariance_type="full",
            reg_covar=reg_covar,
            random_state=seed,
        ).fit(values)
        groups = gmm.predict(values)
        return _interpolate_class(
            values,
            count,
            rng,
            k_neighbors=k_neighbors,
            groups=groups,
        )

    return _sample_by_class(X, y, n0, n1, seed, sample_one_class)
