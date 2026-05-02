# metrics.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import mannwhitneyu, entropy
from rfhack.core.plots import *


# ── helpers ───────────────────────────────────────────────────────────────────

def strat_samp(idx0, idx1, n0, n1, rng):
    a = rng.choice(idx0, size=n0, replace=False)
    b = rng.choice(idx1, size=n1, replace=False)
    return np.concatenate([a, b])


def stratified_subsample(X, y, n0, n1, seed=42):
    """
    Draw n0 class-0 and n1 class-1 rows from X without replacement.

    If the dataset has fewer samples than requested for either class,
    the request is clamped to the available count and a warning is printed.
    This prevents ValueError on small / imbalanced datasets (e.g. HIV).
    """
    rng  = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n0_avail = len(idx0)
    n1_avail = len(idx1)

    if n0 > n0_avail:
        print(f"  [subsample] n0={n0} > available {n0_avail} for class 0 — clamping to {n0_avail}")
        n0 = n0_avail
    if n1 > n1_avail:
        print(f"  [subsample] n1={n1} > available {n1_avail} for class 1 — clamping to {n1_avail}")
        n1 = n1_avail

    idx = strat_samp(idx0, idx1, n0, n1, rng)
    rng.shuffle(idx)
    return X[idx], y[idx], idx


def print_dataset_summary(load_fns):
    """
    Print class counts for each dataset so you can choose valid (n0, n1) pairs.
    Call this once before running experiments.
    """
    print(f"{'Dataset':<20} {'p':>5} {'Class 0':>10} {'Class 1':>10} {'Total':>8}")
    print("-" * 56)
    for fn in load_fns:
        d  = fn()
        y  = d["y"]
        c0 = int((y == 0).sum())
        c1 = int((y == 1).sum())
        print(f"{d['dataset']:<20} {d['X'].shape[1]:>5} {c0:>10} {c1:>10} {len(y):>8}")


# ── RF discriminator (indistinguishability) ───────────────────────────────────

def one_stochastic_experiment(
    X_real, y_real,
    X_syn,  y_syn,
    holdout_neg=3, holdout_pos=12,
    train_neg=20,  train_pos=20,
    seed=42,
    n_estimators=5,
):
    rng = np.random.default_rng(seed)

    real_neg = np.where(y_real == 0)[0]
    real_pos = np.where(y_real == 1)[0]
    syn_neg  = np.where(y_syn  == 0)[0]
    syn_pos  = np.where(y_syn  == 1)[0]

    # clamp holdout sizes to what's available
    holdout_neg = min(holdout_neg, len(real_neg) // 2, len(syn_neg) // 2)
    holdout_pos = min(holdout_pos, len(real_pos) // 2, len(syn_pos) // 2)
    holdout_neg = max(1, holdout_neg)
    holdout_pos = max(1, holdout_pos)

    test_real_idx = strat_samp(real_neg, real_pos, holdout_neg, holdout_pos, rng)
    test_syn_idx  = strat_samp(syn_neg,  syn_pos,  holdout_neg, holdout_pos, rng)

    rem_real = np.setdiff1d(np.arange(X_real.shape[0]), test_real_idx)
    rem_syn  = np.setdiff1d(np.arange(X_syn.shape[0]),  test_syn_idx)

    rem_real_neg = rem_real[y_real[rem_real] == 0]
    rem_real_pos = rem_real[y_real[rem_real] == 1]
    rem_syn_neg  = rem_syn[y_syn[rem_syn]   == 0]
    rem_syn_pos  = rem_syn[y_syn[rem_syn]   == 1]

    # clamp train sizes to what remains
    train_neg = min(train_neg, len(rem_real_neg), len(rem_syn_neg))
    train_pos = min(train_pos, len(rem_real_pos), len(rem_syn_pos))
    train_neg = max(1, train_neg)
    train_pos = max(1, train_pos)

    train_real_idx = strat_samp(rem_real_neg, rem_real_pos, train_neg, train_pos, rng)
    train_syn_idx  = strat_samp(rem_syn_neg,  rem_syn_pos,  train_neg, train_pos, rng)

    X_train = np.vstack([X_real[train_real_idx], X_syn[train_syn_idx]])
    s_train = np.concatenate([
        np.zeros(len(train_real_idx), dtype=int),
        np.ones(len(train_syn_idx),  dtype=int),
    ])

    X_test = np.vstack([X_real[test_real_idx], X_syn[test_syn_idx]])
    s_test = np.concatenate([
        np.zeros(len(test_real_idx), dtype=int),
        np.ones(len(test_syn_idx),  dtype=int),
    ])

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    rf.fit(X_train, s_train)

    p_syn  = rf.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(s_test, p_syn)
    sep    = max(auc, 1 - auc)
    y_pred = rf.predict(X_test)
    disc_f1 = f1_score(s_test, y_pred, average="binary", zero_division=0)

    return {"rf_auc_raw": auc, "rf_auc_sep": sep, "disc_f1": disc_f1}


def run_many_rf_trials(X_real, y_real, X_syn, y_syn, trials=10):
    aucs, seps, f1s = [], [], []

    for t in range(trials):
        if t == 0 or (t + 1) % 10 == 0 or t == trials - 1:
            print(f"  RF trial {t+1}/{trials}")
        out = one_stochastic_experiment(
            X_real, y_real, X_syn, y_syn,
            holdout_neg=3, holdout_pos=12,
            train_neg=20,  train_pos=20,
            seed=t, n_estimators=5,
        )
        aucs.append(out["rf_auc_raw"])
        seps.append(out["rf_auc_sep"])
        f1s.append(out["disc_f1"])

    return {
        "rf_auc_mean":  np.mean(aucs),
        "rf_auc_sd":    np.std(aucs),
        "rf_sep_mean":  np.mean(seps),
        "rf_sep_sd":    np.std(seps),
        "disc_f1_mean": np.mean(f1s),
        "disc_f1_sd":   np.std(f1s),
    }


# ── TSTR utility ──────────────────────────────────────────────────────────────

def tstr_f1(X_real, y_real, X_syn, y_syn, seed=42, n_estimators=100):
    """Train on synthetic, test on real. Also computes TRTR baseline via CV."""
    rf_syn = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    rf_syn.fit(X_syn, y_syn)
    tstr = f1_score(y_real, rf_syn.predict(X_real), average="binary", zero_division=0)

    n_splits = min(5, int((y_real == 0).sum()), int((y_real == 1).sum()))
    n_splits = max(2, n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    trtr_scores = []
    for train_idx, test_idx in skf.split(X_real, y_real):
        rf_real = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        rf_real.fit(X_real[train_idx], y_real[train_idx])
        trtr_scores.append(
            f1_score(y_real[test_idx], rf_real.predict(X_real[test_idx]),
                     average="binary", zero_division=0)
        )

    trtr = np.mean(trtr_scores)
    return {"tstr_f1": tstr, "trtr_f1": trtr, "utility_gap": trtr - tstr}


# ── Structural fidelity ───────────────────────────────────────────────────────

def correlation_diff(X_real, X_syn):
    corr_real = np.corrcoef(X_real, rowvar=False)
    corr_syn  = np.corrcoef(X_syn,  rowvar=False)
    diff = np.abs(corr_real - corr_syn)
    return {
        "corr_mean_abs_diff": float(diff.mean()),
        "corr_max_abs_diff":  float(diff.max()),
    }


def kld_per_feature(X_real, X_syn, bins=30):
    """KL(real || syn) per feature via histogram. Returns scalars + full array."""
    klds = []
    for j in range(X_real.shape[1]):
        lo = min(X_real[:, j].min(), X_syn[:, j].min())
        hi = max(X_real[:, j].max(), X_syn[:, j].max())
        if hi == lo:
            klds.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(X_real[:, j], bins=edges, density=True)
        q, _ = np.histogram(X_syn[:, j],  bins=edges, density=True)
        p = p + 1e-10;  p /= p.sum()
        q = q + 1e-10;  q /= q.sum()
        klds.append(float(entropy(p, q)))
    klds = np.array(klds)
    return {
        "kld_mean":        float(klds.mean()),
        "kld_max":         float(klds.max()),
        "kld_per_feature": klds,
    }


def per_feature_tests(X_real, X_syn, alpha=0.05):
    pvals = []
    for j in range(X_real.shape[1]):
        _, p = mannwhitneyu(X_real[:, j], X_syn[:, j], alternative="two-sided")
        pvals.append(p)
    pvals = np.array(pvals)
    return {
        "pval_mean":        float(pvals.mean()),
        "pval_median":      float(np.median(pvals)),
        "prop_significant": float((pvals < alpha).mean()),
    }


# ── Master evaluation ─────────────────────────────────────────────────────────

def evaluate_all(X_real, y_real, X_syn, y_syn):
    metrics = {}
    figs    = {}

    metrics.update(run_many_rf_trials(X_real, y_real, X_syn, y_syn))
    metrics.update(tstr_f1(X_real, y_real, X_syn, y_syn))
    metrics.update(correlation_diff(X_real, X_syn))
    metrics.update(kld_per_feature(X_real, X_syn))
    metrics.update(per_feature_tests(X_real, X_syn))

    figs["corr"]    = plot_corr_matrices(X_real, X_syn)
    figs["pca"]     = plot_pca_projection(X_real, y_real, X_syn, y_syn)
    figs["overlap"] = plot_flat_overlap(X_real, X_syn)

    return metrics, figs


# ── Ablation summary ──────────────────────────────────────────────────────────

def evaluate_abl(df):
    figs = {}
    for dataset in df["dataset"].unique():
        for mode in ["forward", "reverse"]:
            sub = df[(df["dataset"] == dataset) & (df["feature_mode"] == mode)]
            if sub.empty:
                continue
            for metric_col, error_col in [
                ("rf_sep_mean", "rf_sep_sd"),
                ("kld_mean",    None),
                ("utility_gap", None),
            ]:
                if metric_col not in df.columns:
                    continue
                fig = plot_ablation_curve(
                    df, dataset=dataset,
                    feature_mode=mode,
                    metric_col=metric_col,
                    error_col=error_col,
                )
                figs[f"ablation_{dataset}_{mode}_{metric_col}"] = fig
    return figs
# metrics.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import mannwhitneyu, entropy
from rfhack.core.plots import *


# ── helpers ───────────────────────────────────────────────────────────────────

def strat_samp(idx0, idx1, n0, n1, rng):
    a = rng.choice(idx0, size=n0, replace=False)
    b = rng.choice(idx1, size=n1, replace=False)
    return np.concatenate([a, b])


def stratified_subsample(X, y, n0, n1, seed=42):
    """
    Draw n0 class-0 and n1 class-1 rows from X without replacement.

    If the dataset has fewer samples than requested for either class,
    the request is clamped to the available count and a warning is printed.
    This prevents ValueError on small / imbalanced datasets (e.g. HIV).
    """
    rng  = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    n0_avail = len(idx0)
    n1_avail = len(idx1)

    if n0 > n0_avail:
        print(f"  [subsample] n0={n0} > available {n0_avail} for class 0 — clamping to {n0_avail}")
        n0 = n0_avail
    if n1 > n1_avail:
        print(f"  [subsample] n1={n1} > available {n1_avail} for class 1 — clamping to {n1_avail}")
        n1 = n1_avail

    idx = strat_samp(idx0, idx1, n0, n1, rng)
    rng.shuffle(idx)
    return X[idx], y[idx], idx


def print_dataset_summary(load_fns):
    """
    Print class counts for each dataset so you can choose valid (n0, n1) pairs.
    Call this once before running experiments.
    """
    print(f"{'Dataset':<20} {'p':>5} {'Class 0':>10} {'Class 1':>10} {'Total':>8}")
    print("-" * 56)
    for fn in load_fns:
        d  = fn()
        y  = d["y"]
        c0 = int((y == 0).sum())
        c1 = int((y == 1).sum())
        print(f"{d['dataset']:<20} {d['X'].shape[1]:>5} {c0:>10} {c1:>10} {len(y):>8}")


# ── RF discriminator (indistinguishability) ───────────────────────────────────

def one_stochastic_experiment(
    X_real, y_real,
    X_syn,  y_syn,
    holdout_neg=3, holdout_pos=12,
    train_neg=20,  train_pos=20,
    seed=42,
    n_estimators=5,
):
    rng = np.random.default_rng(seed)

    real_neg = np.where(y_real == 0)[0]
    real_pos = np.where(y_real == 1)[0]
    syn_neg  = np.where(y_syn  == 0)[0]
    syn_pos  = np.where(y_syn  == 1)[0]

    # clamp holdout sizes to what's available
    holdout_neg = min(holdout_neg, len(real_neg) // 2, len(syn_neg) // 2)
    holdout_pos = min(holdout_pos, len(real_pos) // 2, len(syn_pos) // 2)
    holdout_neg = max(1, holdout_neg)
    holdout_pos = max(1, holdout_pos)

    test_real_idx = strat_samp(real_neg, real_pos, holdout_neg, holdout_pos, rng)
    test_syn_idx  = strat_samp(syn_neg,  syn_pos,  holdout_neg, holdout_pos, rng)

    rem_real = np.setdiff1d(np.arange(X_real.shape[0]), test_real_idx)
    rem_syn  = np.setdiff1d(np.arange(X_syn.shape[0]),  test_syn_idx)

    rem_real_neg = rem_real[y_real[rem_real] == 0]
    rem_real_pos = rem_real[y_real[rem_real] == 1]
    rem_syn_neg  = rem_syn[y_syn[rem_syn]   == 0]
    rem_syn_pos  = rem_syn[y_syn[rem_syn]   == 1]

    # clamp train sizes to what remains
    train_neg = min(train_neg, len(rem_real_neg), len(rem_syn_neg))
    train_pos = min(train_pos, len(rem_real_pos), len(rem_syn_pos))
    train_neg = max(1, train_neg)
    train_pos = max(1, train_pos)

    train_real_idx = strat_samp(rem_real_neg, rem_real_pos, train_neg, train_pos, rng)
    train_syn_idx  = strat_samp(rem_syn_neg,  rem_syn_pos,  train_neg, train_pos, rng)

    X_train = np.vstack([X_real[train_real_idx], X_syn[train_syn_idx]])
    s_train = np.concatenate([
        np.zeros(len(train_real_idx), dtype=int),
        np.ones(len(train_syn_idx),  dtype=int),
    ])

    X_test = np.vstack([X_real[test_real_idx], X_syn[test_syn_idx]])
    s_test = np.concatenate([
        np.zeros(len(test_real_idx), dtype=int),
        np.ones(len(test_syn_idx),  dtype=int),
    ])

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    rf.fit(X_train, s_train)

    p_syn  = rf.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(s_test, p_syn)
    sep    = max(auc, 1 - auc)
    y_pred = rf.predict(X_test)
    disc_f1 = f1_score(s_test, y_pred, average="binary", zero_division=0)

    return {"rf_auc_raw": auc, "rf_auc_sep": sep, "disc_f1": disc_f1}


def run_many_rf_trials(X_real, y_real, X_syn, y_syn, trials=10):
    aucs, seps, f1s = [], [], []

    for t in range(trials):
        if t == 0 or (t + 1) % 10 == 0 or t == trials - 1:
            print(f"  RF trial {t+1}/{trials}")
        out = one_stochastic_experiment(
            X_real, y_real, X_syn, y_syn,
            holdout_neg=3, holdout_pos=12,
            train_neg=20,  train_pos=20,
            seed=t, n_estimators=5,
        )
        aucs.append(out["rf_auc_raw"])
        seps.append(out["rf_auc_sep"])
        f1s.append(out["disc_f1"])

    return {
        "rf_auc_mean":  np.mean(aucs),
        "rf_auc_sd":    np.std(aucs),
        "rf_sep_mean":  np.mean(seps),
        "rf_sep_sd":    np.std(seps),
        "disc_f1_mean": np.mean(f1s),
        "disc_f1_sd":   np.std(f1s),
    }


# ── TSTR utility ──────────────────────────────────────────────────────────────

def tstr_f1(X_real, y_real, X_syn, y_syn, seed=42, n_estimators=100):
    """Train on synthetic, test on real. Also computes TRTR baseline via CV."""
    rf_syn = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    rf_syn.fit(X_syn, y_syn)
    tstr = f1_score(y_real, rf_syn.predict(X_real), average="binary", zero_division=0)

    n_splits = min(5, int((y_real == 0).sum()), int((y_real == 1).sum()))
    n_splits = max(2, n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    trtr_scores = []
    for train_idx, test_idx in skf.split(X_real, y_real):
        rf_real = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        rf_real.fit(X_real[train_idx], y_real[train_idx])
        trtr_scores.append(
            f1_score(y_real[test_idx], rf_real.predict(X_real[test_idx]),
                     average="binary", zero_division=0)
        )

    trtr = np.mean(trtr_scores)
    return {"tstr_f1": tstr, "trtr_f1": trtr, "utility_gap": trtr - tstr}


# ── Structural fidelity ───────────────────────────────────────────────────────

def correlation_diff(X_real, X_syn):
    corr_real = np.corrcoef(X_real, rowvar=False)
    corr_syn  = np.corrcoef(X_syn,  rowvar=False)
    diff = np.abs(corr_real - corr_syn)
    return {
        "corr_mean_abs_diff": float(diff.mean()),
        "corr_max_abs_diff":  float(diff.max()),
    }


def kld_per_feature(X_real, X_syn, bins=30):
    """KL(real || syn) per feature via histogram. Returns scalars + full array."""
    klds = []
    for j in range(X_real.shape[1]):
        lo = min(X_real[:, j].min(), X_syn[:, j].min())
        hi = max(X_real[:, j].max(), X_syn[:, j].max())
        if hi == lo:
            klds.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        p, _ = np.histogram(X_real[:, j], bins=edges, density=True)
        q, _ = np.histogram(X_syn[:, j],  bins=edges, density=True)
        p = p + 1e-10;  p /= p.sum()
        q = q + 1e-10;  q /= q.sum()
        klds.append(float(entropy(p, q)))
    klds = np.array(klds)
    return {
        "kld_mean":        float(klds.mean()),
        "kld_max":         float(klds.max()),
        "kld_per_feature": klds,
    }


def per_feature_tests(X_real, X_syn, alpha=0.05):
    pvals = []
    for j in range(X_real.shape[1]):
        _, p = mannwhitneyu(X_real[:, j], X_syn[:, j], alternative="two-sided")
        pvals.append(p)
    pvals = np.array(pvals)
    return {
        "pval_mean":        float(pvals.mean()),
        "pval_median":      float(np.median(pvals)),
        "prop_significant": float((pvals < alpha).mean()),
    }


# ── Master evaluation ─────────────────────────────────────────────────────────

def evaluate_all(X_real, y_real, X_syn, y_syn):
    metrics = {}
    figs    = {}

    metrics.update(run_many_rf_trials(X_real, y_real, X_syn, y_syn))
    metrics.update(tstr_f1(X_real, y_real, X_syn, y_syn))
    metrics.update(correlation_diff(X_real, X_syn))
    metrics.update(kld_per_feature(X_real, X_syn))
    metrics.update(per_feature_tests(X_real, X_syn))

    figs["corr"]    = plot_corr_matrices(X_real, X_syn)
    figs["pca"]     = plot_pca_projection(X_real, y_real, X_syn, y_syn)
 
    # KLD per feature — replaces the flat overlap histogram
    kld_result = kld_per_feature(X_real, X_syn)
    metrics.update(kld_result)
    figs["kld"]        = plot_kld_per_feature(X_real, X_syn)
    figs["kld_array"]  = kld_result["kld_per_feature"] 
    
    return metrics, figs


# ── Ablation summary ──────────────────────────────────────────────────────────

def evaluate_abl(df):
    figs = {}
    for dataset in df["dataset"].unique():
        for mode in ["forward", "reverse"]:
            sub = df[(df["dataset"] == dataset) & (df["feature_mode"] == mode)]
            if sub.empty:
                continue
            for metric_col, error_col in [
                ("rf_sep_mean", "rf_sep_sd"),
                ("kld_mean",    None),
                ("utility_gap", None),
            ]:
                if metric_col not in df.columns:
                    continue
                fig = plot_ablation_curve(
                    df, dataset=dataset,
                    feature_mode=mode,
                    metric_col=metric_col,
                    error_col=error_col,
                )
                figs[f"ablation_{dataset}_{mode}_{metric_col}"] = fig
    return figs
