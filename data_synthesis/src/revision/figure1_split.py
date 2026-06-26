"""Figure 1 metric strip for one selected dataset."""

import ast
import contextlib
import io

from src.revision.common import *
from src.revision.stats import ks_by_feature, mean_kld_by_feature


SCORE_METRIC_NAMES = ["AUC"]
KLD_METRIC_NAME = "KLD"
KS_METRIC_NAME = "KS"
METRIC_SPACING = 1.95
GENERATOR_OFFSETS = np.linspace(-0.62, 0.62, len(METHOD_ORDER))
GENERATOR_VIOLIN_WIDTH = 0.24
METRIC_LABEL_Y = -0.2
GROUP_SHADE_COLOR = "#FFFFFF"
METHOD_TICK_LABELS = {
    "Bootstrap": "Bootstrap",
    "Column-wise": "Column-\nwise",
    "GMM": "GMM",
    "CVAE": "CVAE",
}


def _coerce_values(values):
    if isinstance(values, str):
        values = ast.literal_eval(values)
    return np.asarray(values, dtype=float)


def _dataset_metric_table(metric_table, dataset):
    return (
        metric_table[metric_table["dataset"] == dataset]
        .set_index("method")
        .reindex(METHOD_ORDER)
        .reset_index()
    )


def _feature_kld_values(dataset, seed=SEED, cvae_epochs=CVAE_EPOCHS):
    data = require_datasets()[dataset]
    X_real = np.asarray(data["X"], dtype=np.float32)
    out = []
    for method in METHOD_ORDER:
        with contextlib.redirect_stdout(io.StringIO()):
            X_syn, _ = sample_synthetic(dataset, data, method, seed=seed, cvae_epochs=cvae_epochs)
        out.append(mean_kld_by_feature(X_real, X_syn))
    return out


def _feature_ks_values(dataset, seed=SEED, cvae_epochs=CVAE_EPOCHS):
    data = require_datasets()[dataset]
    X_real = np.asarray(data["X"], dtype=np.float32)
    out = []
    for method in METHOD_ORDER:
        with contextlib.redirect_stdout(io.StringIO()):
            X_syn, _ = sample_synthetic(dataset, data, method, seed=seed, cvae_epochs=cvae_epochs)
        out.append(ks_by_feature(X_real, X_syn))
    return out


def _draw_distribution_at(ax, pos, vals, method):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return

    if len(vals) < 2 or np.isclose(np.nanmin(vals), np.nanmax(vals)):
        ax.scatter(
            pos,
            float(np.nanmean(vals)),
            s=38,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        return

    violin = ax.violinplot(
        [vals],
        positions=[pos],
        widths=GENERATOR_VIOLIN_WIDTH,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    body = violin["bodies"][0]
    body.set_facecolor(METHOD_PASTELS[method])
    body.set_edgecolor(METHOD_COLORS[method])
    body.set_alpha(0.92)
    body.set_linewidth(1.25)
    body.set_zorder(2)

    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    mean = np.mean(vals)
    ax.vlines(pos, q1, q3, color=METHOD_COLORS[method], linewidth=2.1, alpha=0.90, zorder=3)
    ax.scatter(pos, med, s=28, color="white", edgecolor=METHOD_COLORS[method], linewidth=1.2, zorder=4)
    ax.scatter(pos, mean, s=20, color=METHOD_COLORS[method], edgecolor="white", linewidth=0.6, zorder=5)


def _set_kld_limits(ax, kld_values):
    finite = np.concatenate([np.asarray(v, dtype=float)[np.isfinite(v)] for v in kld_values])
    if len(finite) == 0:
        ax.set_ylim(0, 1)
        return
    upper = float(np.nanmax(finite))
    ax.set_ylim(0, upper * 1.10 if upper > 0 else 1)


def _set_positive_metric_limits(ax, values, upper_bound=None):
    finite_groups = [np.asarray(v, dtype=float)[np.isfinite(v)] for v in values]
    finite_groups = [v for v in finite_groups if len(v) > 0]
    if not finite_groups:
        ax.set_ylim(0, upper_bound or 1)
        return
    finite = np.concatenate(finite_groups)
    upper = float(np.nanmax(finite))
    if upper_bound is not None:
        ax.set_ylim(0, upper_bound)
    else:
        ax.set_ylim(0, upper * 1.10 if upper > 0 else 1)


def _draw_metric_label(ax, x, label):
    ax.text(
        x,
        METRIC_LABEL_Y,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=10.5,
        weight="semibold",
        clip_on=False,
    )


def plot_figure1_proportion_strip(metric_table, dataset, seed=SEED, cvae_epochs=CVAE_EPOCHS):
    """Plot discriminator AUC distributions for one selected dataset."""
    sub = _dataset_metric_table(metric_table, dataset)
    score_centers = np.arange(len(SCORE_METRIC_NAMES), dtype=float) * METRIC_SPACING
    score_values = {
        "AUC": [_coerce_values(v) for v in sub["rf_auc_values"]],
    }

    fig, score_ax = plt.subplots(1, 1, figsize=(4.8, 3.75), constrained_layout=False)
    score_positions = []
    score_labels = []
    left_edge = score_centers[0] - METRIC_SPACING / 2
    right_edge = score_centers[-1] + METRIC_SPACING / 2
    group_edges = [left_edge]
    group_edges.extend((score_centers[:-1] + score_centers[1:]) / 2)
    group_edges.append(right_edge)

    for center_i, metric in enumerate(SCORE_METRIC_NAMES):
        center = score_centers[center_i]
        if center_i % 2 == 0:
            score_ax.axvspan(group_edges[center_i], group_edges[center_i + 1], color=GROUP_SHADE_COLOR, zorder=0)
        for offset, method, vals in zip(GENERATOR_OFFSETS, METHOD_ORDER, score_values[metric]):
            pos = center + offset
            score_positions.append(pos)
            score_labels.append(METHOD_TICK_LABELS[method])
            _draw_distribution_at(score_ax, pos, vals, method)

    score_ax.set_xlim(left_edge, right_edge)
    score_ax.set_ylim(-0.02, 1.03)
    score_ax.set_xticks(score_positions)
    score_ax.set_xticklabels(score_labels, fontsize=7.2, linespacing=0.9)
    for center, metric in zip(score_centers, SCORE_METRIC_NAMES):
        _draw_metric_label(score_ax, center, metric)
    score_ax.set_ylabel("Score", labelpad=6)

    clean_axis(score_ax, grid_axis="y")
    for spine in score_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    score_ax.tick_params(axis="y", labelsize=8.8, width=1.2, length=4)
    score_ax.tick_params(axis="x", length=0, pad=4)

    fig.subplots_adjust(left=0.13, right=0.97, top=0.9, bottom=0.33)
    return fig


def plot_figure1_marginal_metrics(dataset, seed=SEED, cvae_epochs=CVAE_EPOCHS):
    """Plot feature-wise KLD and KS distributions for one selected dataset."""
    kld_values = _feature_kld_values(dataset, seed=seed, cvae_epochs=cvae_epochs)
    ks_values = _feature_ks_values(dataset, seed=seed, cvae_epochs=cvae_epochs)

    fig, (kld_ax, ks_ax) = plt.subplots(
        1,
        2,
        figsize=(7.4, 3.75),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.24},
    )

    kld_center = 0
    kld_positions = []
    kld_labels = []
    for offset, method, vals in zip(GENERATOR_OFFSETS, METHOD_ORDER, kld_values):
        pos = kld_center + offset
        kld_positions.append(pos)
        kld_labels.append(METHOD_TICK_LABELS[method])
        _draw_distribution_at(kld_ax, pos, vals, method)

    ks_center = 0
    ks_positions = []
    ks_labels = []
    for offset, method, vals in zip(GENERATOR_OFFSETS, METHOD_ORDER, ks_values):
        pos = ks_center + offset
        ks_positions.append(pos)
        ks_labels.append(METHOD_TICK_LABELS[method])
        _draw_distribution_at(ks_ax, pos, vals, method)

    kld_ax.set_xlim(kld_center - METRIC_SPACING / 2, kld_center + METRIC_SPACING / 2)
    _set_kld_limits(kld_ax, kld_values)
    kld_ax.set_xticks(kld_positions)
    kld_ax.set_xticklabels(kld_labels, fontsize=7.2, linespacing=0.9)
    _draw_metric_label(kld_ax, kld_center, KLD_METRIC_NAME)
    kld_ax.set_ylabel("KLD", labelpad=4)
    kld_ax.yaxis.set_label_position("left")
    kld_ax.yaxis.tick_left()

    ks_ax.set_xlim(ks_center - METRIC_SPACING / 2, ks_center + METRIC_SPACING / 2)
    _set_positive_metric_limits(ks_ax, ks_values, upper_bound=1.0)
    ks_ax.set_xticks(ks_positions)
    ks_ax.set_xticklabels(ks_labels, fontsize=7.2, linespacing=0.9)
    _draw_metric_label(ks_ax, ks_center, KS_METRIC_NAME)
    ks_ax.set_ylabel("KS statistic", labelpad=4)
    ks_ax.yaxis.set_label_position("left")
    ks_ax.yaxis.tick_left()

    for ax in (kld_ax, ks_ax):
        clean_axis(ax, grid_axis="y")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
        ax.tick_params(axis="y", labelsize=8.8, width=1.2, length=4)
        ax.tick_params(axis="x", length=0, pad=4)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.33)
    return fig


def plot_feature_ks_violin(dataset, seed=SEED, cvae_epochs=CVAE_EPOCHS):
    """Plot feature-wise two-sample KS statistics across generators."""
    ks_values = _feature_ks_values(dataset, seed=seed, cvae_epochs=cvae_epochs)

    fig, ax = plt.subplots(figsize=(5.4, 3.75), constrained_layout=False)
    positions = []
    labels = []
    center = 0
    for offset, method, vals in zip(GENERATOR_OFFSETS, METHOD_ORDER, ks_values):
        pos = center + offset
        positions.append(pos)
        labels.append(METHOD_TICK_LABELS[method])
        _draw_distribution_at(ax, pos, vals, method)

    ax.set_xlim(center - METRIC_SPACING / 2, center + METRIC_SPACING / 2)
    _set_positive_metric_limits(ax, ks_values, upper_bound=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7.2, linespacing=0.9)
    _draw_metric_label(ax, center, KS_METRIC_NAME)
    ax.set_ylabel("KS statistic", labelpad=4)
    clean_axis(ax, grid_axis="y")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.tick_params(axis="y", labelsize=8.8, width=1.2, length=4)
    ax.tick_params(axis="x", length=0, pad=4)

    fig.subplots_adjust(left=0.16, right=0.97, top=0.86, bottom=0.30)
    return fig
