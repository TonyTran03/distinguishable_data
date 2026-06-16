"""Reusable statistical metrics used across revision figures."""

from src.revision.common import *

def one_run_origin_auc(X_real, y_real, X_syn, y_syn, seed):
    """Class-balanced real-vs-synthetic RF probe for one random split."""
    rng = np.random.default_rng(seed)
    X_real = np.asarray(X_real, dtype=np.float64)
    X_syn = np.asarray(X_syn, dtype=np.float64)
    y_real = np.asarray(y_real, dtype=int)
    y_syn = np.asarray(y_syn, dtype=int)
    real_neg, real_pos = np.where(y_real == 0)[0], np.where(y_real == 1)[0]
    syn_neg, syn_pos = np.where(y_syn == 0)[0], np.where(y_syn == 1)[0]
    n_neg = min(len(real_neg), len(syn_neg))
    n_pos = min(len(real_pos), len(syn_pos))
    real_idx = np.r_[rng.choice(real_neg, n_neg, replace=False), rng.choice(real_pos, n_pos, replace=False)]
    syn_idx = np.r_[rng.choice(syn_neg, n_neg, replace=False), rng.choice(syn_pos, n_pos, replace=False)]
    Xr, Xs = standardize_pair(X_real[real_idx], X_syn[syn_idx])
    X = np.vstack([Xr, Xs])
    origin = np.r_[np.zeros(len(Xr), dtype=int), np.ones(len(Xs), dtype=int)]
    X_train, X_test, y_train, y_test = train_test_split(X, origin, test_size=0.25, stratify=origin, random_state=seed)
    rf = RandomForestClassifier(n_estimators=500, random_state=seed, class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    return float(max(auc, 1.0 - auc))

def mean_kld_by_feature(X_real, X_syn, bins=30):
    X_real = np.asarray(X_real, dtype=float)
    X_syn = np.asarray(X_syn, dtype=float)
    vals = []
    for j in range(X_real.shape[1]):
        lo = float(min(np.nanmin(X_real[:, j]), np.nanmin(X_syn[:, j])))
        hi = float(max(np.nanmax(X_real[:, j]), np.nanmax(X_syn[:, j])))
        if not np.isfinite(lo + hi) or hi <= lo:
            vals.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(X_real[:, j], bins=edges, density=True)
        q, _ = np.histogram(X_syn[:, j], bins=edges, density=True)
        p = p + 1e-10
        q = q + 1e-10
        vals.append(float(entropy(p / p.sum(), q / q.sum())))
    return np.asarray(vals, dtype=float)

def nn_distance_mean(X_real, X_syn):
    Xr, Xs = standardize_pair(X_real, X_syn)
    nn = NearestNeighbors(n_neighbors=1).fit(Xr)
    dists, _ = nn.kneighbors(Xs)
    return float(np.mean(dists))

def tstr_values(X_real, y_real, X_syn, y_syn, seed=SEED, repeats=TSTR_REPEATS):
    Xr, Xs = standardize_pair(X_real, X_syn)
    y_real = np.asarray(y_real, dtype=int)
    y_syn = np.asarray(y_syn, dtype=int)
    n_splits = max(2, min(5, int((y_real == 0).sum()), int((y_real == 1).sum())))
    out = []
    trtr = []
    for r in range(repeats):
        run_seed = seed + r * 1009
        clf = RandomForestClassifier(n_estimators=300, random_state=run_seed, class_weight="balanced", n_jobs=-1)
        clf.fit(Xs, y_syn)
        out.append(float(f1_score(y_real, clf.predict(Xr), zero_division=0)))
        fold_scores = []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run_seed)
        for train_idx, test_idx in cv.split(Xr, y_real):
            rf = RandomForestClassifier(n_estimators=300, random_state=run_seed, class_weight="balanced", n_jobs=-1)
            rf.fit(Xr[train_idx], y_real[train_idx])
            fold_scores.append(float(f1_score(y_real[test_idx], rf.predict(Xr[test_idx]), zero_division=0)))
        trtr.append(float(np.mean(fold_scores)))
    return np.asarray(out), np.asarray(trtr)

def target_rf_importance(X, y, seed=SEED):
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    rf = RandomForestClassifier(n_estimators=800, random_state=seed, class_weight="balanced", n_jobs=-1)
    rf.fit(Xs, np.asarray(y, dtype=int))
    return rf.feature_importances_

def corr_matrix(X):
    C = np.corrcoef(np.asarray(X, dtype=float), rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 1.0)
    return C

def upper_triangle_values(C):
    idx = np.triu_indices_from(C, k=1)
    return C[idx]

def corr_preservation_summary(real_corr, syn_corr):
    x = upper_triangle_values(real_corr)
    y = upper_triangle_values(syn_corr)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan
    mae = float(np.mean(np.abs(x - y)))
    return x, y, r, mae

def correlation_frobenius_discrepancy(real_corr, syn_corr):
    return float(np.linalg.norm(np.asarray(real_corr, dtype=float) - np.asarray(syn_corr, dtype=float), ord="fro"))

def rank_discriminating_features(X_real, X_syn, seed=SEED):
    Xr, Xs = standardize_pair(X_real, X_syn)
    X = np.vstack([Xr, Xs])
    y = np.r_[np.zeros(len(Xr), dtype=int), np.ones(len(Xs), dtype=int)]
    rf = RandomForestClassifier(n_estimators=500, random_state=seed, class_weight="balanced", n_jobs=-1)
    rf.fit(X, y)
    return np.argsort(rf.feature_importances_)[::-1]

def ablation_grid(n_features, points=8):
    if n_features <= 2:
        return np.array([0], dtype=int)
    return np.unique(np.rint(np.linspace(0, n_features - 2, points)).astype(int))

def stratified_subsample(X, y, n0, n1, seed=SEED):
    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    take0 = rng.choice(idx0, size=max(1, min(n0, len(idx0))), replace=False)
    take1 = rng.choice(idx1, size=max(1, min(n1, len(idx1))), replace=False)
    idx = np.r_[take0, take1]
    rng.shuffle(idx)
    return X[idx], y[idx]
