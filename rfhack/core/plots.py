# plots.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator


# ── Shared style ──────────────────────────────────────────────────────────────

METHOD_COLORS = {"bootstrap": "#4878CF", "gmm": "#F28E2B", "cvae": "#59A14F"}
METHOD_ORDER  = ["bootstrap", "gmm", "cvae"]


# ── 1. Correlation matrices + difference panel ────────────────────────────────

def plot_corr_matrices(X_real, X_syn):
    """
    Three-panel figure:
      Left  — real correlation matrix
      Middle — synthetic correlation matrix
      Right  — absolute difference (the actual finding)
    """
    corr_real = np.corrcoef(X_real, rowvar=False)
    corr_syn  = np.corrcoef(X_syn,  rowvar=False)
    diff      = np.abs(corr_real - corr_syn)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    im0 = axs[0].imshow(corr_real, vmin=-1, vmax=1, cmap="RdBu_r")
    axs[0].set_title("Real", fontsize=12)

    im1 = axs[1].imshow(corr_syn, vmin=-1, vmax=1, cmap="RdBu_r")
    axs[1].set_title("Synthetic", fontsize=12)

    # difference panel — separate colormap and colorbar
    im2 = axs[2].imshow(diff, vmin=0, vmax=1, cmap="Oranges")
    axs[2].set_title("Absolute difference\n(synthetic − real)", fontsize=12)

    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    fig.suptitle("Pairwise correlation structure", fontsize=13, y=1.02)
    return fig


# ── 2. PCA projection ─────────────────────────────────────────────────────────

def add_confidence_ellipse(ax, x, y, edgecolor, n_std=2.0, lw=2, label=None):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if cov.shape != (2, 2):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle  = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h   = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(
        (np.mean(x), np.mean(y)), width=w, height=h, angle=angle,
        facecolor="none", edgecolor=edgecolor, lw=lw, label=label,
    )
    ax.add_patch(ell)


def plot_pca_projection(X_real, y_real, X_syn, y_syn, max_syn_display=300):
    """
    PCA scatter with confidence ellipses.

    max_syn_display caps the number of synthetic points plotted so real
    data is not visually buried — synthetic geometry is captured by the
    ellipses regardless.
    """
    pca = PCA(n_components=2)
    pca.fit(X_real)

    Z_real = pca.transform(X_real)
    Z_syn  = pca.transform(X_syn)

    # subsample synthetic for display only
    if len(Z_syn) > max_syn_display:
        rng  = np.random.default_rng(0)
        idx  = rng.choice(len(Z_syn), size=max_syn_display, replace=False)
        Z_syn_plot  = Z_syn[idx]
        y_syn_plot  = y_syn[idx]
    else:
        Z_syn_plot  = Z_syn
        y_syn_plot  = y_syn

    df_plot = pd.DataFrame({
        "PC1":    np.concatenate([Z_real[:, 0], Z_syn_plot[:, 0]]),
        "PC2":    np.concatenate([Z_real[:, 1], Z_syn_plot[:, 1]]),
        "class":  np.concatenate([y_real, y_syn_plot]).astype(str),
        "source": ["Real"] * len(Z_real) + ["Synthetic"] * len(Z_syn_plot),
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=df_plot, x="PC1", y="PC2",
        hue="class", style="source",
        palette={"0": "#4878CF", "1": "#E15759"},
        markers={"Real": "o", "Synthetic": "^"},
        alpha=0.55, s=55, ax=ax,
    )

    ellipse_style = {
        ("Real",      "0"): "#4878CF",
        ("Real",      "1"): "#E15759",
        ("Synthetic", "0"): "#76A8E0",
        ("Synthetic", "1"): "#F0908F",
    }
    for (src, cls), color in ellipse_style.items():
        d = df_plot[(df_plot["source"] == src) & (df_plot["class"] == cls)]
        add_confidence_ellipse(ax, d["PC1"], d["PC2"], edgecolor=color,
                               n_std=2.0, lw=2)

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("PCA projection — real vs synthetic by class", fontsize=12)
    ax.axhline(0, color="lightgray", lw=0.8)
    ax.axvline(0, color="lightgray", lw=0.8)
    ax.grid(True, alpha=0.2)
    return fig


# ── 3. KLD per feature (replaces flat overlap histogram) ─────────────────────

def plot_kld_per_feature(X_real, X_syn, feature_names=None, bins=30, top_n=None):
    """
    Bar chart of KL(real || syn) per feature, sorted descending.
    Directly answers: which features are hardest to synthesize?

    Parameters
    ----------
    top_n : int or None — only show the top_n worst features (useful for wide datasets)
    """
    from scipy.stats import entropy

    p = X_real.shape[1]
    klds = []
    for j in range(p):
        lo = min(X_real[:, j].min(), X_syn[:, j].min())
        hi = max(X_real[:, j].max(), X_syn[:, j].max())
        if hi == lo:
            klds.append(0.0)
            continue
        edges = np.linspace(lo, hi, bins + 1)
        pv, _ = np.histogram(X_real[:, j], bins=edges, density=True)
        qv, _ = np.histogram(X_syn[:, j],  bins=edges, density=True)
        pv = pv + 1e-10;  pv /= pv.sum()
        qv = qv + 1e-10;  qv /= qv.sum()
        klds.append(float(entropy(pv, qv)))

    klds = np.array(klds)
    names = feature_names if feature_names is not None else [f"f{j}" for j in range(p)]

    order = np.argsort(klds)[::-1]
    if top_n is not None:
        order = order[:top_n]

    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.45), 4), constrained_layout=True)
    bars = ax.bar(
        range(len(order)),
        klds[order],
        color="#E15759", alpha=0.8, edgecolor="white", linewidth=0.5,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("KL divergence  KL(real ∥ syn)", fontsize=11)
    ax.set_title("Per-feature distributional divergence (sorted worst → best)", fontsize=12)
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_kld_per_feature_by_method(kld_dict, feature_names=None, top_n=None):
    """
    Side-by-side KLD bars for multiple methods on the same feature set.

    Parameters
    ----------
    kld_dict : dict — {method_name: kld_array}, e.g. from evaluate_all outputs
    top_n    : int  — show only top_n features ranked by mean KLD across methods
    """
    methods = [m for m in METHOD_ORDER if m in kld_dict]
    if not methods:
        raise ValueError("kld_dict has no recognised method keys.")

    p      = len(next(iter(kld_dict.values())))
    names  = feature_names if feature_names is not None else [f"f{j}" for j in range(p)]
    mean_k = np.mean([kld_dict[m] for m in methods], axis=0)
    order  = np.argsort(mean_k)[::-1]
    if top_n is not None:
        order = order[:top_n]

    x      = np.arange(len(order))
    width  = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.55), 4.5), constrained_layout=True)

    for i, method in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            kld_dict[method][order],
            width=width * 0.9,
            label=method,
            color=METHOD_COLORS.get(method, f"C{i}"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([names[i] for i in order], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("KL divergence  KL(real ∥ syn)", fontsize=11)
    ax.set_title("Per-feature KLD by method (sorted by mean, worst → best)", fontsize=12)
    ax.legend(title="Method", fontsize=10)
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    return fig


# ── 4. Ablation curve ─────────────────────────────────────────────────────────

def plot_ablation_curve(
    df,
    dataset,
    feature_mode="forward",
    metric_col="rf_sep_mean",
    error_col="rf_sep_sd",
):
    """
    RF Sep (or any metric) vs. k for forward / reverse ablation.

    Changes vs original:
    - y-axis always spans [0.5, 1.0] with a dashed reference line at 0.5
      so the reader has an anchor for "indistinguishable"
    - consistent method colors
    - x-axis label reflects forward vs reverse
    """
    sub = df[
        (df["dataset"] == dataset) &
        (df["feature_mode"] == feature_mode)
    ].copy()

    if sub.empty:
        raise ValueError(f"No rows for dataset={dataset}, feature_mode={feature_mode}")

    x_col = "subset_param" if "subset_param" in sub.columns else "k"
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub = sub.dropna(subset=[x_col, metric_col])

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    methods_present = [m for m in METHOD_ORDER if m in sub["method"].unique()]
    for method in methods_present:
        d = sub[sub["method"] == method].sort_values(x_col)
        color = METHOD_COLORS.get(method, None)

        if error_col and error_col in d.columns:
            ax.errorbar(
                d[x_col], d[metric_col], yerr=d[error_col],
                marker="o", capsize=4, label=method,
                color=color, linewidth=2, markersize=6,
            )
        else:
            ax.plot(
                d[x_col], d[metric_col],
                marker="o", label=method,
                color=color, linewidth=2, markersize=6,
            )

    # reference line — indistinguishable baseline
    ax.axhline(0.5, color="black", lw=1.2, ls="--", alpha=0.6, label="chance (0.5)")

    # fixed y-axis so all plots are comparable across datasets
    ax.set_ylim(0.48, 1.02)

    if feature_mode == "forward":
        ax.set_xlabel("Features kept (top k by RF importance)", fontsize=11)
    elif feature_mode == "reverse":
        ax.set_xlabel("Top-k features dropped", fontsize=11)
    else:
        ax.set_xlabel(x_col, fontsize=11)

    ylabel = metric_col.replace("_", " ")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{dataset} — {feature_mode} ablation", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    return fig
