"""Figure 1 metric strip for one selected dataset."""

import ast
import contextlib
import io

from src.revision.common import *
from src.revision.stats import mean_kld_by_feature


METRIC_NAMES = ["AUC", "Utility", "KLD"]
METRIC_SPACING = 1.95
GENERATOR_OFFSETS = np.linspace(-0.62, 0.62, len(METHOD_ORDER))
GENERATOR_VIOLIN_WIDTH = 0.24
METRIC_LABEL_Y = -0.2
GROUP_SHADE_COLOR = "#FFFBFB"
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


def plot_figure1_proportion_strip(metric_table, dataset, seed=SEED, cvae_epochs=CVAE_EPOCHS):
    """Plot AUC, TSTR utility, and bounded feature-wise KLD on one 0-1 axis."""
    sub = _dataset_metric_table(metric_table, dataset)
    centers = np.arange(len(METRIC_NAMES), dtype=float) * METRIC_SPACING
    metric_values = {
        "AUC": [_coerce_values(v) for v in sub["rf_auc_values"]],
        "Utility": [_coerce_values(v) for v in sub["tstr_f1_values"]],
        "KLD": [v / (1.0 + v) for v in _feature_kld_values(dataset, seed=seed, cvae_epochs=cvae_epochs)],
    }

    fig, ax = plt.subplots(figsize=(11.4, 3.75), constrained_layout=False)
    all_positions = []
    all_labels = []
    left_edge = centers[0] - METRIC_SPACING / 2
    right_edge = centers[-1] + METRIC_SPACING / 2
    group_edges = [left_edge]
    group_edges.extend((centers[:-1] + centers[1:]) / 2)
    group_edges.append(right_edge)

    for center_i, metric in enumerate(METRIC_NAMES):
        center = centers[center_i]
        if center_i % 2 == 0:
            ax.axvspan(group_edges[center_i], group_edges[center_i + 1], color=GROUP_SHADE_COLOR, zorder=0)
            # ax.axvspan(group_edges[center_i], group_edges[center_i + 1], zorder=0)
        for offset, method, vals in zip(GENERATOR_OFFSETS, METHOD_ORDER, metric_values[metric]):
            pos = center + offset
            all_positions.append(pos)
            all_labels.append(METHOD_TICK_LABELS[method])
            _draw_distribution_at(ax, pos, vals, method)

    ax.set_xlim(left_edge, right_edge)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, fontsize=7.2, linespacing=0.9)

    for center, metric in zip(centers, METRIC_NAMES):
        ax.text(
            center,
            METRIC_LABEL_Y,
            metric,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10.5,
            weight="semibold",
            clip_on=False,
        )

    ax.set_ylabel("Proportion", labelpad=12)
    clean_axis(ax, grid_axis="y")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.tick_params(axis="y", labelsize=8.8, width=1.2, length=4)
    ax.tick_params(axis="x", length=0, pad=4)
    fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.33)
    return fig
