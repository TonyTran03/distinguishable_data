"""Expanded, t-SNE-free follow-up evaluation for the original paper."""

from __future__ import annotations

import contextlib
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import ttest_ind
from sklearn.covariance import graphical_lasso
from sklearn.decomposition import PCA

from models.bootstrap import sample_bootstrap
from models.cvae import sample_cvae
from models.gmm import sample_gmm
from models.iid_columnwise import sample_columnwise
from models.smote import sample_gmm_guided_smote, sample_smote
from models.wgan_gp import sample_wgan_gp
from src.revision.common import (
    Config,
    DATASET_COLORS,
    add_confidence_ellipse,
    apply_notebook_figure_style,
    class_counts,
    standardize_pair,
)
from src.revision.figure4_graphical_lasso import FIGURE4_ALPHAS
from src.revision.figure4_graphical_lasso_plots import (
    STATUS_COLORS,
    build_edge_status_matrix,
    compute_edge_recovery,
    compute_frobenius_deviation,
    compute_synthetic_only_rate,
    fit_glasso_precision,
    get_edge_set,
    get_real_structure_order,
    precision_to_partial_corr,
)
from src.revision.stats import (
    ablation_grid,
    ks_by_feature,
    mean_kld_by_feature,
    nn_distance_mean,
    one_run_origin_auc,
    rank_discriminating_features,
    stratified_subsample,
    tstr_values,
)

METHOD_ORDER = [
    "Bootstrap",
    "Column-wise",
    "GMM",
    "SMOTE",
    "GMM-guided SMOTE",
    "CVAE",
]
METHOD_COLORS = {
    "Bootstrap": "#6A5ACD",
    "Column-wise": "#CC79A7",
    "GMM": "#009E73",
    "SMOTE": "#0072B2",
    "GMM-guided SMOTE": "#56B4E9",
    "CVAE": "#D55E00",
    "WGAN-GP": "#8C564B",
}

PANEL_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def label_figure_panels(fig, axes=None, start=0):
    """Give every supplied Matplotlib axes its own sequential panel letter."""
    axes = list(fig.axes if axes is None else axes)
    if start < 0 or start + len(axes) > len(PANEL_LABELS):
        raise ValueError("Panel labels must fit within A-Z.")

    # Replace older group-level labels so a composite cannot contain duplicate
    # letters after this per-axes labelling pass.
    for ax in fig.axes:
        for text_artist in list(ax.texts):
            if text_artist.get_text() in set(PANEL_LABELS):
                text_artist.remove()

    for offset, ax in enumerate(axes):
        panel_label = ax.text(
            -0.055,
            1.045,
            PANEL_LABELS[start + offset],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            weight="bold",
            color="#111111",
            clip_on=False,
            zorder=20,
        )
        panel_label.set_gid("panel-letter")
    return fig


def sample_method(
    X,
    y,
    method,
    seed=42,
    cvae_epochs=50,
    wgan_epochs=50,
):
    n0, n1 = class_counts(y)
    if method == "Bootstrap":
        return sample_bootstrap(X, y, n0, n1, seed=seed)
    if method == "Column-wise":
        return sample_columnwise(X, y, n0, n1, seed=seed)
    if method == "GMM":
        return sample_gmm(X, y, n0, n1, seed=seed)
    if method == "SMOTE":
        return sample_smote(X, y, n0, n1, seed=seed)
    if method == "GMM-guided SMOTE":
        return sample_gmm_guided_smote(X, y, n0, n1, seed=seed)
    if method == "CVAE":
        with contextlib.redirect_stdout(io.StringIO()):
            return sample_cvae(
                X,
                y,
                n0,
                n1,
                seed=seed,
                cfg=Config(
                    seed=seed,
                    epochs=cvae_epochs,
                    x_transform="none",
                    latent_prior="normal",
                ),
            )
    if method == "WGAN-GP":
        return sample_wgan_gp(X, y, n0, n1, seed=seed, epochs=wgan_epochs)
    raise ValueError(f"Unknown method: {method}")


def generate_cohorts(
    datasets,
    methods=METHOD_ORDER,
    seed=42,
    cvae_epochs=50,
    wgan_epochs=50,
):
    cohorts = {}
    for dataset, data in datasets.items():
        cohorts[dataset] = {}
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=int)
        for offset, method in enumerate(methods):
            print(f"[generate] {dataset} - {method}")
            cohorts[dataset][method] = sample_method(
                X,
                y,
                method,
                seed=seed + 101 * offset,
                cvae_epochs=cvae_epochs,
                wgan_epochs=wgan_epochs,
            )
    return cohorts


def compute_origin_auc(datasets, cohorts, repeats=5, seed=42):
    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"])
        y_real = np.asarray(datasets[dataset]["y"], dtype=int)
        for method, (X_syn, y_syn) in method_data.items():
            for repeat in range(int(repeats)):
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "run": repeat,
                        "separability_auc": one_run_origin_auc(
                            X_real,
                            y_real,
                            X_syn,
                            y_syn,
                            seed=seed + 1009 * repeat,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_figure1_auc_boxplots(auc_runs):
    """Plot repeated direction-invariant origin AUC values by dataset and method."""

    datasets = list(dict.fromkeys(auc_runs["dataset"]))
    methods = [
        method for method in METHOD_ORDER
        if method in set(auc_runs["method"])
    ]
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.0 * len(datasets), 4.4),
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, dataset, panel in zip(axes, datasets, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        subset = auc_runs[auc_runs["dataset"] == dataset]
        values = [
            subset.loc[
                subset["method"] == method, "separability_auc"
            ].dropna().to_numpy(dtype=float)
            for method in methods
        ]
        boxes = ax.boxplot(
            values,
            positions=np.arange(len(methods)),
            widths=0.62,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#222222",
                "markersize": 4.5,
            },
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#555555", "linewidth": 1.1},
            capprops={"color": "#555555", "linewidth": 1.1},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#777777",
                "markeredgecolor": "none",
                "markersize": 3.2,
                "alpha": 0.45,
            },
        )
        for box, method in zip(boxes["boxes"], methods):
            box.set_facecolor(METHOD_COLORS[method])
            box.set_edgecolor(METHOD_COLORS[method])
            box.set_alpha(0.58)
            box.set_linewidth(1.4)

        # Show individual repetitions without obscuring the box summaries.
        rng = np.random.default_rng(42)
        for position, method_values, method in zip(
            np.arange(len(methods)), values, methods
        ):
            jitter = rng.uniform(-0.11, 0.11, size=len(method_values))
            ax.scatter(
                position + jitter,
                method_values,
                s=12,
                color=METHOD_COLORS[method],
                edgecolors="white",
                linewidths=0.25,
                alpha=0.45,
                zorder=3,
            )

        ax.axhline(
            0.5,
            color="#777777",
            linestyle="--",
            linewidth=1.1,
            label="Chance",
        )
        ax.set_title(
            dataset,
            color=DATASET_COLORS[dataset],
            fontsize=12,
            weight="bold",
        )
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.47, 1.02)
        ax.set_xlabel("Synthetic generator")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.5)
        ax.text(
            -0.08,
            1.04,
            panel,
            transform=ax.transAxes,
            fontsize=15,
            weight="bold",
            va="top",
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.1)

    axes[0].set_ylabel(r"Distinguishability score, $D_r$")
    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.93,
        bottom=0.29,
        wspace=0.12,
    )
    return apply_notebook_figure_style(fig)


def compute_feature_kld_table(datasets, cohorts):
    """Return one KLD value per dataset, method, and feature."""

    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"], dtype=float)
        names = list(
            datasets[dataset].get(
                "feature_names",
                [f"feature_{index + 1}" for index in range(X_real.shape[1])],
            )
        )
        for method, (X_syn, _) in method_data.items():
            values = mean_kld_by_feature(X_real, X_syn)
            for feature_index, value in enumerate(values):
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "feature_index": feature_index,
                        "feature": names[feature_index],
                        "kld": float(value),
                    }
                )
    return pd.DataFrame(rows)


def plot_feature_kld_boxplots(feature_kld):
    """Plot feature-level KLD distributions for every dataset and generator."""

    datasets = list(dict.fromkeys(feature_kld["dataset"]))
    methods = [
        method for method in METHOD_ORDER
        if method in set(feature_kld["method"])
    ]
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.0 * len(datasets), 4.4),
        sharey=False,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, dataset, panel in zip(axes, datasets, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        subset = feature_kld[feature_kld["dataset"] == dataset]
        values = [
            subset.loc[subset["method"] == method, "kld"]
            .dropna()
            .to_numpy(dtype=float)
            for method in methods
        ]
        boxes = ax.boxplot(
            values,
            positions=np.arange(len(methods)),
            widths=0.62,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#222222",
                "markersize": 4.5,
            },
            medianprops={"color": "#111111", "linewidth": 1.5},
            whiskerprops={"color": "#555555", "linewidth": 1.1},
            capprops={"color": "#555555", "linewidth": 1.1},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#777777",
                "markeredgecolor": "none",
                "markersize": 3.0,
                "alpha": 0.35,
            },
        )
        for box, method in zip(boxes["boxes"], methods):
            box.set_facecolor(METHOD_COLORS[method])
            box.set_edgecolor(METHOD_COLORS[method])
            box.set_alpha(0.58)
            box.set_linewidth(1.4)

        ax.set_title(
            dataset,
            color=DATASET_COLORS[dataset],
            fontsize=12,
            weight="bold",
        )
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("Synthetic generator")
        ax.set_ylabel("Feature KLD")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.5)
        ax.text(
            -0.08,
            1.04,
            panel,
            transform=ax.transAxes,
            fontsize=15,
            weight="bold",
            va="top",
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.1)

    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.93,
        bottom=0.29,
        wspace=0.20,
    )
    return apply_notebook_figure_style(fig)


def plot_figure1_fidelity_grid(
    auc_runs,
    feature_kld,
    marginal_tests,
    tstr_runs,
    dataset_order=None,
    method_order=None,
    jitter_seed=42,
):
    """Plot AUC, feature KLD, and utility-gap distributions in a 3 x 3 grid."""
    dataset_order = list(dataset_order or dict.fromkeys(auc_runs["dataset"]))
    method_order = [
        method
        for method in (method_order or METHOD_ORDER)
        if method in set(auc_runs["method"])
    ]
    if len(dataset_order) != 3:
        raise ValueError("Figure 1 requires exactly three datasets")
    if "utility_gap_abs" not in tstr_runs.columns:
        tstr_runs = tstr_runs.assign(
            utility_gap_abs=(tstr_runs["trtr_f1"] - tstr_runs["tstr_f1"]).abs()
        )

    row_specs = [
        (auc_runs, "separability_auc", "AUC"),
        (feature_kld, "kld", "KLD"),
        (tstr_runs, "utility_gap_abs", "Utility gap"),
    ]
    # Portrait proportions allow the complete 3 x 3 figure and its caption to
    # occupy an A4 journal page when included at \textwidth.
    fig, axes = plt.subplots(3, 3, figsize=(9.1, 10.4), squeeze=False)

    def consistent_jitter(count, width=0.105):
        if count <= 1:
            return np.zeros(count, dtype=float)
        offsets = np.linspace(-width, width, count)
        return offsets[np.random.default_rng(jitter_seed + count).permutation(count)]

    for row, (table, value_column, row_label) in enumerate(row_specs):
        for col, dataset in enumerate(dataset_order):
            ax = axes[row, col]
            subset = table[table["dataset"] == dataset]
            values = [
                subset.loc[subset["method"] == method, value_column]
                .dropna()
                .to_numpy(dtype=float)
                for method in method_order
            ]
            boxes = ax.boxplot(
                values,
                positions=np.arange(len(method_order)),
                widths=0.62,
                patch_artist=True,
                showmeans=True,
                showfliers=False,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "#333333",
                    "markeredgewidth": 0.8,
                    "markersize": 4.2,
                },
                medianprops={"color": "#111111", "linewidth": 1.35},
                whiskerprops={"color": "#666666", "linewidth": 1.0},
                capprops={"color": "#666666", "linewidth": 1.0},
            )
            for box, method in zip(boxes["boxes"], method_order):
                box.set_facecolor(METHOD_COLORS[method])
                box.set_edgecolor(METHOD_COLORS[method])
                box.set_alpha(0.58)
                box.set_linewidth(1.25)

            for position, method_values, method in zip(
                np.arange(len(method_order)), values, method_order
            ):
                ax.scatter(
                    position + consistent_jitter(len(method_values)),
                    method_values,
                    s=11,
                    color=METHOD_COLORS[method],
                    edgecolors="white",
                    linewidths=0.22,
                    alpha=0.38,
                    zorder=3,
                )

            if row == 0:
                ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0)
                ax.set_ylim(0.47, 1.02)
                ax.set_title(
                    dataset,
                    color=DATASET_COLORS[dataset],
                    fontsize=12.5,
                    weight="bold",
                    pad=10,
                )
            elif row == 1:
                ax.set_ylim(bottom=0.0)
            elif row == 2:
                ax.set_ylim(0.0, 0.40)
                ax.set_yticks(np.arange(0.0, 0.401, 0.10))
                ax.tick_params(axis="y", labelleft=col == 0)

            if col == 0:
                ax.set_ylabel(row_label, fontsize=11, weight="semibold")
            else:
                ax.set_ylabel("")
            ax.set_xticks(np.arange(len(method_order)))
            ax.set_xticklabels(method_order, rotation=45, ha="center", fontsize=8.0)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.55)
            ax.tick_params(direction="out", width=0.8)
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color("#333333")

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.955,
        bottom=0.105,
        wspace=0.16,
        hspace=0.48,
    )
    for letter, ax in zip("ABCDEFGHI", axes.ravel()):
        position = ax.get_position()
        fig.text(
            position.x0 - 0.027,
            position.y1 + 0.006,
            letter,
            ha="left",
            va="bottom",
            fontsize=14,
            weight="bold",
        )
    return apply_notebook_figure_style(fig)


def plot_ks_supplement(
    marginal_tests,
    dataset_order=None,
    method_order=None,
    jitter_seed=42,
):
    """Preserve the former Figure 1 KS row as a supplementary figure."""
    dataset_order = list(dataset_order or dict.fromkeys(marginal_tests["dataset"]))
    method_order = [
        method for method in (method_order or METHOD_ORDER)
        if method in set(marginal_tests["method"])
    ]
    fig, axes = plt.subplots(
        1,
        len(dataset_order),
        figsize=(9.1, 3.65),
        sharey=True,
        squeeze=False,
    )

    def consistent_jitter(count, width=0.105):
        if count <= 1:
            return np.zeros(count, dtype=float)
        offsets = np.linspace(-width, width, count)
        return offsets[
            np.random.default_rng(jitter_seed + count).permutation(count)
        ]

    for col, dataset in enumerate(dataset_order):
        ax = axes[0, col]
        subset = marginal_tests[marginal_tests["dataset"] == dataset]
        values = [
            subset.loc[subset["method"] == method, "ks_statistic"]
            .dropna()
            .to_numpy(dtype=float)
            for method in method_order
        ]
        boxes = ax.boxplot(
            values,
            positions=np.arange(len(method_order)),
            widths=0.62,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#333333",
                "markeredgewidth": 0.8,
                "markersize": 4.2,
            },
            medianprops={"color": "#111111", "linewidth": 1.35},
            whiskerprops={"color": "#666666", "linewidth": 1.0},
            capprops={"color": "#666666", "linewidth": 1.0},
        )
        for box, method in zip(boxes["boxes"], method_order):
            box.set_facecolor(METHOD_COLORS[method])
            box.set_edgecolor(METHOD_COLORS[method])
            box.set_alpha(0.58)
            box.set_linewidth(1.25)
        for position, method_values, method in zip(
            np.arange(len(method_order)), values, method_order
        ):
            ax.scatter(
                position + consistent_jitter(len(method_values)),
                method_values,
                s=11,
                color=METHOD_COLORS[method],
                edgecolors="white",
                linewidths=0.22,
                alpha=0.38,
                zorder=3,
            )
        ax.set_ylim(0.0, 0.65)
        ax.set_title(dataset, fontsize=11.0, weight="semibold", pad=7)
        ax.set_xticks(np.arange(len(method_order)))
        ax.set_xticklabels(method_order, rotation=45, ha="center", fontsize=7.6)
        ax.set_ylabel("KS score" if col == 0 else "", fontsize=10.0, weight="semibold")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.55)
        ax.tick_params(direction="out", width=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#333333")
        ax.text(
            -0.10,
            1.035,
            chr(ord("A") + col),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12.0,
            weight="bold",
            clip_on=False,
        )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.90,
        bottom=0.26,
        wspace=0.16,
    )
    return apply_notebook_figure_style(fig)


def compute_tstr_runs(datasets, cohorts, repeats=3, seed=42):
    """Return repeat-level TSTR and TRTR F1 values for plotting and summaries."""
    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"])
        y_real = np.asarray(datasets[dataset]["y"], dtype=int)
        for method, (X_syn, y_syn) in method_data.items():
            tstr, trtr = tstr_values(
                X_real,
                y_real,
                X_syn,
                y_syn,
                seed=seed,
                repeats=repeats,
            )
            rows.extend(
                {
                    "dataset": dataset,
                    "method": method,
                    "repeat": repeat,
                    "tstr_f1": float(tstr_value),
                    "trtr_f1": float(trtr_value),
                    "utility_gap_abs": float(abs(trtr_value - tstr_value)),
                }
                for repeat, (tstr_value, trtr_value) in enumerate(zip(tstr, trtr))
            )
    return pd.DataFrame(rows)


def compute_metric_table(
    datasets,
    cohorts,
    auc_runs,
    tstr_repeats=3,
    seed=42,
    tstr_runs=None,
):
    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"])
        y_real = np.asarray(datasets[dataset]["y"], dtype=int)
        for method, (X_syn, y_syn) in method_data.items():
            if tstr_runs is None:
                tstr, trtr = tstr_values(
                    X_real,
                    y_real,
                    X_syn,
                    y_syn,
                    seed=seed,
                    repeats=tstr_repeats,
                )
            else:
                repeat_values = tstr_runs.query(
                    "dataset == @dataset and method == @method"
                ).sort_values("repeat")
                tstr = repeat_values["tstr_f1"].to_numpy(dtype=float)
                trtr = repeat_values["trtr_f1"].to_numpy(dtype=float)
            auc = auc_runs.query(
                "dataset == @dataset and method == @method"
            )["separability_auc"].to_numpy()
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "origin_auc_mean": float(np.mean(auc)),
                    "origin_auc_sd": float(np.std(auc)),
                    "tstr_f1_mean": float(np.mean(tstr)),
                    "tstr_f1_sd": float(np.std(tstr)),
                    "utility_gap_abs_mean": float(np.mean(np.abs(trtr - tstr))),
                    "mean_feature_kld": float(
                        np.mean(mean_kld_by_feature(X_real, X_syn))
                    ),
                    "mean_feature_ks": float(
                        np.nanmean(ks_by_feature(X_real, X_syn))
                    ),
                    "nn_distance_mean": nn_distance_mean(X_real, X_syn),
                }
            )
    return pd.DataFrame(rows)


def plot_pca_grid(datasets, cohorts, dataset):
    methods = [method for method in METHOD_ORDER if method in cohorts[dataset]]
    columns = 3
    rows = math.ceil(len(methods) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.6 / 1.18, (3.7 * rows) / 1.18),
        squeeze=False,
    )
    axes = np.atleast_1d(axes).ravel()
    X_real = np.asarray(datasets[dataset]["X"])

    for ax, method in zip(axes, methods):
        X_syn = cohorts[dataset][method][0]
        Xr, Xs = standardize_pair(X_real, X_syn)
        pca = PCA(n_components=2, random_state=42).fit(Xr)
        Zr, Zs = pca.transform(Xr), pca.transform(Xs)
        ax.scatter(
            Zr[:, 0], Zr[:, 1], s=8, facecolors="none",
            edgecolors="#777777", linewidths=0.8, alpha=0.55, label="Real",
        )
        ax.scatter(
            Zs[:, 0], Zs[:, 1], s=8, color=METHOD_COLORS[method],
            edgecolors="none", alpha=0.6, label=method,
        )
        add_confidence_ellipse(
            ax,
            Zr,
            "#777777",
            linestyle="-",
            linewidth=2.0,
        )
        add_confidence_ellipse(
            ax,
            Zs,
            METHOD_COLORS[method],
            linestyle="-",
            linewidth=2.2,
        )
        ax.set_title(
            method,
            color=METHOD_COLORS[method],
            fontsize=11.0,
            weight="bold",
            pad=5.0,
        )
        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
            fontsize=9.0,
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
            fontsize=9.0,
        )
        ax.tick_params(axis="both", labelsize=8.0, direction="out")
        ax.legend(
            frameon=False,
            fontsize=8.5,
            markerscale=1.25,
            handletextpad=0.45,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#444444")
        ax.grid(False)
    for ax in axes[len(methods):]:
        ax.axis("off")
    fig.tight_layout(h_pad=1.15, w_pad=1.0)
    return apply_notebook_figure_style(fig)


def plot_pca_all_datasets_a4(
    datasets,
    cohorts,
    dataset_order=("HIV", "Breast Cancer", "Diabetes"),
):
    """Plot all PCA comparisons on one balanced A4 portrait page.

    Each dataset receives exactly one third of the figure and contains its six
    synthesis methods in a 2 x 3 grid. Axis limits are shared within a dataset
    so its method panels can be compared directly.
    """
    dataset_order = [dataset for dataset in dataset_order if dataset in cohorts]
    if not dataset_order:
        raise ValueError("No requested datasets are present in cohorts.")

    # apply_notebook_figure_style scales figures by FIGURE_SIZE_SCALE (1.18),
    # so begin at the reciprocal dimensions to finish at true A4 portrait.
    fig = plt.figure(figsize=(8.27 / 1.18, 11.69 / 1.18), facecolor="white")
    outer = fig.add_gridspec(
        len(dataset_order),
        1,
        left=0.075,
        right=0.985,
        top=0.985,
        bottom=0.045,
        hspace=0.12,
    )

    panel_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    display_names = {"GMM-guided SMOTE": "GMM-SMOTE"}

    for dataset_index, dataset in enumerate(dataset_order):
        methods = [method for method in METHOD_ORDER if method in cohorts[dataset]]
        if len(methods) > 6:
            raise ValueError("The A4 PCA layout supports at most six methods per dataset.")

        section = outer[dataset_index].subgridspec(
            3,
            3,
            height_ratios=[0.14, 1.0, 1.0],
            hspace=0.46,
            wspace=0.25,
        )
        heading_ax = fig.add_subplot(section[0, :])
        heading_ax.axis("off")
        heading_ax.text(
            0.5,
            0.55,
            f"{panel_letters[dataset_index]}. {dataset}",
            ha="center",
            va="center",
            fontsize=12.0,
            weight="bold",
            color="#222222",
            family="DejaVu Sans",
        )

        X_real = np.asarray(datasets[dataset]["X"])
        payloads = {}
        all_coordinates = []
        for method in methods:
            X_syn = cohorts[dataset][method][0]
            Xr, Xs = standardize_pair(X_real, X_syn)
            pca = PCA(n_components=2, random_state=42).fit(Xr)
            Zr, Zs = pca.transform(Xr), pca.transform(Xs)
            payloads[method] = (Zr, Zs, pca.explained_variance_ratio_)
            all_coordinates.extend((Zr, Zs))

        combined = np.vstack(all_coordinates)
        x_min, y_min = np.nanmin(combined, axis=0)
        x_max, y_max = np.nanmax(combined, axis=0)
        x_pad = max((x_max - x_min) * 0.08, 0.5)
        y_pad = max((y_max - y_min) * 0.08, 0.5)
        x_limits = (x_min - x_pad, x_max + x_pad)
        y_limits = (y_min - y_pad, y_max + y_pad)

        for method_index, method in enumerate(methods):
            row, col = divmod(method_index, 3)
            ax = fig.add_subplot(section[row + 1, col])
            Zr, Zs, explained_variance = payloads[method]
            method_label = display_names.get(method, method)

            ax.scatter(
                Zr[:, 0],
                Zr[:, 1],
                s=6.5,
                marker="o",
                facecolors="none",
                edgecolors="#777777",
                linewidths=0.55,
                alpha=0.48,
                label="Real",
                zorder=2,
            )
            ax.scatter(
                Zs[:, 0],
                Zs[:, 1],
                s=6.5,
                marker="o",
                color=METHOD_COLORS[method],
                edgecolors="none",
                alpha=0.58,
                label="Synthetic",
                zorder=3,
            )
            add_confidence_ellipse(ax, Zr, "#777777", linewidth=1.25)
            add_confidence_ellipse(
                ax,
                Zs,
                METHOD_COLORS[method],
                linewidth=1.45,
            )

            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
            ax.set_title(
                method_label,
                color=METHOD_COLORS[method],
                fontsize=7.8,
                weight="semibold",
                pad=2.5,
            )
            if row == 1:
                ax.set_xlabel(
                    f"PC1 ({explained_variance[0] * 100:.1f}%)",
                    fontsize=6.3,
                    labelpad=1.5,
                )
            else:
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel(
                    f"PC2 ({explained_variance[1] * 100:.1f}%)",
                    fontsize=6.3,
                    labelpad=1.5,
                )
            else:
                ax.set_ylabel("")
            ax.tick_params(
                axis="both",
                labelsize=5.8,
                direction="out",
                length=2.5,
                width=0.75,
                pad=1.5,
            )
            ax.tick_params(axis="x", labelbottom=row == 1)
            ax.tick_params(axis="y", labelleft=col == 0)
            ax.legend(
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="#BDBDBD",
                framealpha=0.82,
                prop={"size": 6.2, "weight": "semibold", "family": "DejaVu Sans"},
                markerscale=1.35,
                borderpad=0.30,
                labelspacing=0.20,
                handlelength=1.0,
                handletextpad=0.35,
            )
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#444444")
            ax.grid(False)

        for method_index in range(len(methods), 6):
            row, col = divmod(method_index, 3)
            empty_ax = fig.add_subplot(section[row + 1, col])
            empty_ax.axis("off")

    return apply_notebook_figure_style(fig)


def plot_metric_summary(metric_table):
    specifications = [
        ("origin_auc_mean", "Origin AUC"),
        ("tstr_f1_mean", "TSTR F1"),
        ("utility_gap_abs_mean", "|TRTR - TSTR|"),
        ("mean_feature_ks", "Mean feature KS"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0))
    for ax, (column, title) in zip(axes.ravel(), specifications):
        pivot = metric_table.pivot(index="method", columns="dataset", values=column)
        pivot = pivot.reindex(METHOD_ORDER)
        pivot.plot.bar(ax=ax, color=[DATASET_COLORS[c] for c in pivot.columns])
        ax.set_title(title, weight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return apply_notebook_figure_style(fig)


def compute_structure_metrics(datasets, cohorts, threshold=1e-7):
    rows = []
    for dataset, method_data in cohorts.items():
        alpha = FIGURE4_ALPHAS[dataset]
        theta_real = fit_glasso_precision(datasets[dataset]["X"], alpha)
        partial_real = precision_to_partial_corr(theta_real)
        real_edges = get_edge_set(partial_real, threshold=threshold)
        for method, (X_syn, _) in method_data.items():
            theta_syn = fit_glasso_precision(X_syn, alpha)
            partial_syn = precision_to_partial_corr(theta_syn)
            syn_edges = get_edge_set(partial_syn, threshold=threshold)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "frobenius_deviation": compute_frobenius_deviation(
                        theta_real, theta_syn
                    ),
                    "edge_recovery": compute_edge_recovery(real_edges, syn_edges),
                    "synthetic_only_rate": compute_synthetic_only_rate(
                        real_edges, syn_edges
                    ),
                    "real_edges": len(real_edges),
                    "synthetic_edges": len(syn_edges),
                }
            )
    return pd.DataFrame(rows)


def plot_structure_metrics(structure_table):
    metrics = [
        ("frobenius_deviation", "Precision Frobenius deviation"),
        ("edge_recovery", "Real-edge recovery"),
        ("synthetic_only_rate", "Synthetic-only edge rate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    for ax, (column, title) in zip(axes, metrics):
        pivot = structure_table.pivot(
            index="method", columns="dataset", values=column
        ).reindex(METHOD_ORDER)
        pivot.plot.bar(ax=ax, color=[DATASET_COLORS[c] for c in pivot.columns])
        ax.set_title(title, weight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return apply_notebook_figure_style(fig)


def plot_glasso_regularization_paths(
    datasets,
    selected_alphas=None,
    dataset_order=None,
    n_alphas=40,
    alpha_min_ratio=0.1,
    alpha_max_ratio=2.5,
    max_edges=10,
):
    """Plot the strongest off-diagonal Graphical Lasso coefficient paths.

    Features are standardized exactly as in the structural comparison.  The
    returned table contains the plotted (rather than all possible) edges so it
    can be exported alongside the figure without producing an unwieldy file.
    """
    selected_alphas = dict(selected_alphas or FIGURE4_ALPHAS)
    dataset_order = list(dataset_order or datasets.keys())
    if n_alphas < 2:
        raise ValueError("n_alphas must be at least 2")
    if not 0 < alpha_min_ratio < alpha_max_ratio:
        raise ValueError("alpha ratios must satisfy 0 < min < max")
    if max_edges < 1:
        raise ValueError("max_edges must be positive")

    fig, axes = plt.subplots(
        1, len(dataset_order), figsize=(6.2 * len(dataset_order), 5.4), squeeze=False
    )
    rows = []
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, min(max_edges, 10)))

    for ax, dataset in zip(axes.ravel(), dataset_order):
        X = np.asarray(datasets[dataset]["X"], dtype=np.float64)
        names = [str(name) for name in datasets[dataset]["feature_names"]]
        Xs, _ = standardize_pair(X, X)
        emp_cov = Xs.T @ Xs / Xs.shape[0]
        selected_alpha = float(selected_alphas[dataset])
        selected_precision = fit_glasso_precision(X, selected_alpha)
        selected_partial = precision_to_partial_corr(selected_precision)
        heatmap_order = get_real_structure_order(selected_partial)
        heatmap_number = np.empty(Xs.shape[1], dtype=int)
        heatmap_number[heatmap_order] = np.arange(1, Xs.shape[1] + 1)
        off_diagonal_cov = emp_cov.copy()
        np.fill_diagonal(off_diagonal_cov, 0.0)
        zero_solution_alpha = float(np.max(np.abs(off_diagonal_cov)))
        alpha_grid = np.geomspace(
            selected_alpha * alpha_min_ratio,
            max(selected_alpha * alpha_max_ratio, zero_solution_alpha * 1.05),
            num=n_alphas,
        )

        edge_i, edge_j = np.triu_indices(Xs.shape[1], k=1)
        path = np.empty((n_alphas, len(edge_i)), dtype=float)
        for index, alpha in enumerate(alpha_grid):
            if alpha >= zero_solution_alpha:
                precision = np.diag(1.0 / np.diag(emp_cov))
            else:
                # LARS is more stable near the all-zero end of these paths;
                # coordinate descent is faster for the denser solutions.
                mode = "lars" if alpha >= 0.5 * zero_solution_alpha else "cd"
                _, precision = graphical_lasso(
                    emp_cov,
                    alpha=float(alpha),
                    mode=mode,
                    max_iter=1000,
                    tol=1e-3,
                )
            path[index] = precision[edge_i, edge_j]

        strengths = np.max(np.abs(path), axis=0)
        plotted = np.flatnonzero(strengths > 1e-10)
        plotted = plotted[np.argsort(strengths[plotted])[::-1][:max_edges]]
        for line_index, edge_index in enumerate(plotted):
            i, j = int(edge_i[edge_index]), int(edge_j[edge_index])
            edge_a = int(heatmap_number[i])
            edge_b = int(heatmap_number[j])
            display_label = f"Edge ({edge_a}, {edge_b})"
            color = colors[line_index % len(colors)]
            ax.plot(
                alpha_grid,
                path[:, edge_index],
                color=color,
                linewidth=1.7,
                label=display_label,
            )
            rows.extend(
                {
                    "dataset": dataset,
                    "alpha": float(alpha),
                    "selected_alpha": selected_alpha,
                    "feature_a_matrix_index": edge_a,
                    "feature_b_matrix_index": edge_b,
                    "feature_a": names[i],
                    "feature_b": names[j],
                    "precision_coefficient": float(coefficient),
                }
                for alpha, coefficient in zip(alpha_grid, path[:, edge_index])
            )

        ax.axvline(
            selected_alpha,
            color="#222222",
            linestyle="--",
            linewidth=1.4,
            label=rf"analysis $\alpha={selected_alpha:g}$",
        )
        ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.7)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_title(dataset, weight="bold")
        ax.set_xlabel(r"Regularization parameter $\alpha$")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            frameon=False,
            fontsize=7.2,
            ncol=1,
        )

    axes[0, 0].set_ylabel("Precision-matrix coefficient")
    fig.tight_layout()
    return apply_notebook_figure_style(fig), pd.DataFrame(rows)


def compute_marginal_tests(datasets, cohorts):
    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"], dtype=float)
        names = datasets[dataset].get(
            "feature_names", [f"feature_{i + 1}" for i in range(X_real.shape[1])]
        )
        for method, (X_syn, _) in method_data.items():
            ks_values = ks_by_feature(X_real, X_syn)
            for index, name in enumerate(names):
                test = ttest_ind(
                    X_real[:, index],
                    np.asarray(X_syn)[:, index],
                    equal_var=False,
                    nan_policy="omit",
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "feature": name,
                        "t_statistic": float(test.statistic),
                        "p_value": float(test.pvalue),
                        "ks_statistic": float(ks_values[index]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_marginal_tests(table):
    marked = table.assign(p_below_0_05=table["p_value"] < 0.05)
    return (
        marked.groupby(["dataset", "method"], as_index=False)
        .agg(
            features_tested=("feature", "count"),
            features_p_below_0_05=("p_below_0_05", "sum"),
            median_p_value=("p_value", "median"),
            mean_ks_statistic=("ks_statistic", "mean"),
        )
        .assign(
            percent_p_below_0_05=lambda x:
            100 * x["features_p_below_0_05"] / x["features_tested"]
        )
    )


def plot_hiv_marginal_overlap_supplement(
    datasets,
    cohorts,
    marginal_tests,
    dataset="HIV",
    method_order=None,
    top_n=8,
    feature_start=0,
):
    """Plot one title-free supplement page of HIV marginal overlaps.

    Features are ranked by their largest KS statistic across methods. Each row
    uses one real-derived standardization and one pooled 1st--99th percentile
    range so the six synthetic-method panels are directly comparable. Use
    ``feature_start`` to paginate the complete ranked feature list.
    """
    method_order = [
        method
        for method in (method_order or METHOD_ORDER)
        if method in cohorts[dataset]
    ]
    if len(method_order) != 6:
        raise ValueError(
            "The HIV marginal-overlap supplement requires all six methods."
        )
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    if feature_start < 0:
        raise ValueError("feature_start cannot be negative.")

    X_real = np.asarray(datasets[dataset]["X"], dtype=np.float64)
    feature_names = list(
        datasets[dataset].get(
            "feature_names",
            [f"feature_{index + 1}" for index in range(X_real.shape[1])],
        )
    )
    name_to_index = {str(name): index for index, name in enumerate(feature_names)}
    subset = marginal_tests[marginal_tests["dataset"] == dataset].copy()
    ranked_features = (
        subset.groupby("feature", sort=False)["ks_statistic"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()[feature_start:feature_start + int(top_n)]
    )
    if not ranked_features:
        raise ValueError(f"No marginal-test results are available for {dataset!r}.")

    # The reciprocal dimensions finish at true A4 portrait after the shared
    # notebook style applies its 1.18 figure-size multiplier.
    fig, axes = plt.subplots(
        len(ranked_features),
        len(method_order),
        figsize=(8.27 / 1.18, 11.69 / 1.18),
        squeeze=False,
        sharex="row",
        sharey="row",
    )

    for row, feature in enumerate(ranked_features):
        feature_index = name_to_index[str(feature)]
        real_values = X_real[:, feature_index]
        real_values = real_values[np.isfinite(real_values)]
        center = float(np.mean(real_values))
        scale = float(np.std(real_values, ddof=1))
        if not np.isfinite(scale) or np.isclose(scale, 0.0):
            scale = 1.0
        real_z = (real_values - center) / scale

        synthetic_z = {}
        for method in method_order:
            values = np.asarray(
                cohorts[dataset][method][0][:, feature_index], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            synthetic_z[method] = (values - center) / scale

        pooled = np.concatenate([real_z, *synthetic_z.values()])
        x_low, x_high = np.quantile(pooled, [0.01, 0.99])
        x_low, x_high = float(x_low), float(x_high)
        if np.isclose(x_low, x_high):
            pad = max(0.5, abs(x_low) * 0.05)
            x_low, x_high = x_low - pad, x_high + pad
        visible = pooled[(pooled >= x_low) & (pooled <= x_high)]
        candidate_edges = np.histogram_bin_edges(visible, bins="fd")
        n_bins = int(np.clip(len(candidate_edges) - 1, 10, 22))
        edges = np.linspace(x_low, x_high, n_bins + 1)
        axis_pad = 0.07 * (x_high - x_low)

        for col, method in enumerate(method_order):
            ax = axes[row, col]
            method_color = METHOD_COLORS[method]
            ax.hist(
                real_z,
                bins=edges,
                density=True,
                histtype="stepfilled",
                color="#777777",
                alpha=0.34,
                linewidth=0,
            )
            ax.hist(
                synthetic_z[method],
                bins=edges,
                density=True,
                histtype="stepfilled",
                color=method_color,
                alpha=0.34,
                linewidth=0,
            )
            ax.hist(
                real_z,
                bins=edges,
                density=True,
                histtype="step",
                color="#333434",
                linewidth=0.85,
            )
            ax.hist(
                synthetic_z[method],
                bins=edges,
                density=True,
                histtype="step",
                color=method_color,
                linewidth=0.85,
            )
            ks_values = subset.loc[
                (subset["method"] == method) & (subset["feature"] == feature),
                "ks_statistic",
            ]
            if not ks_values.empty:
                ax.text(
                    0.97,
                    0.09,
                    f"KS {float(ks_values.iloc[0]):.2f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=4.8,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.80,
                        "pad": 0.7,
                    },
                )

            if row == 0:
                ax.set_title(method, fontsize=7.3, weight="semibold", pad=4.0)
            if col == 0:
                feature_label = str(feature).replace("_", " ")
                if len(feature_label) > 21:
                    feature_label = feature_label[:18] + "..."
                ax.set_ylabel(
                    feature_label,
                    fontsize=3.8,
                    weight="semibold",
                    labelpad=2.0,
                )
            else:
                ax.tick_params(axis="y", labelleft=False)
            if row == len(ranked_features) - 1:
                ax.set_xlabel("Standardized value", fontsize=5.7)
            else:
                ax.tick_params(axis="x", labelbottom=False)

            ax.set_xlim(x_low - axis_pad, x_high + axis_pad)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.45)
            ax.tick_params(labelsize=4.8, width=0.6, length=2.0, pad=1.0)
            for spine in ax.spines.values():
                spine.set_linewidth(0.65)
                spine.set_color("#555555")

    fig.subplots_adjust(
        left=0.072,
        right=0.998,
        top=0.955,
        bottom=0.075,
        wspace=0.035,
        hspace=0.12,
    )
    return apply_notebook_figure_style(fig)


def compute_noise_sensitivity(
    datasets,
    cohorts,
    sigmas=(0.0, 0.2, 0.5, 1.0),
    repeats=3,
    seed=42,
):
    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"], dtype=float)
        y_real = np.asarray(datasets[dataset]["y"], dtype=int)
        scale = np.where(X_real.std(axis=0) == 0, 1.0, X_real.std(axis=0))
        for method, (base_syn, y_syn) in method_data.items():
            for sigma in sigmas:
                values = []
                for repeat in range(int(repeats)):
                    rng = np.random.default_rng(seed + 1009 * repeat)
                    X_syn = np.asarray(base_syn, dtype=float)
                    if sigma:
                        X_syn = X_syn + rng.normal(size=X_syn.shape) * scale * sigma
                    values.append(
                        one_run_origin_auc(
                            X_real, y_real, X_syn, y_syn, seed + 101 * repeat
                        )
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "sigma": float(sigma),
                        "sep_mean": float(np.mean(values)),
                        "sep_sd": float(np.std(values)),
                    }
                )
    return pd.DataFrame(rows)


def plot_noise_sensitivity(table):
    columns = 3
    rows = math.ceil(len(METHOD_ORDER) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(13.2, 3.5 * rows), sharey=True)
    axes = np.asarray(axes).ravel()
    for ax, method in zip(axes, METHOD_ORDER):
        for dataset in table["dataset"].unique():
            values = table.query(
                "method == @method and dataset == @dataset"
            ).sort_values("sigma")
            ax.errorbar(
                values["sigma"], values["sep_mean"], yerr=values["sep_sd"],
                marker="o", color=DATASET_COLORS[dataset], label=dataset,
            )
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(method, color=METHOD_COLORS[method], weight="bold")
        ax.set_xlabel("Noise sigma")
        ax.set_ylabel("Origin AUC")
    for ax in axes[len(METHOD_ORDER):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return apply_notebook_figure_style(fig)


def plot_hiv_noise_sensitivity_supplement(
    noise_table,
    dataset="HIV",
    method_order=None,
):
    """Render E2's HIV noise-sensitivity panel as an A4 supplement page."""
    method_order = [
        method
        for method in (method_order or METHOD_ORDER)
        if method in set(noise_table["method"])
    ]
    if len(method_order) != 6:
        raise ValueError(
            "The HIV noise-sensitivity supplement requires all six methods."
        )

    # The reciprocal dimensions finish at true A4 landscape after the shared
    # notebook style applies its 1.18 figure-size multiplier.
    fig, ax = plt.subplots(figsize=(11.69 / 1.18, 8.27 / 1.18))
    display_names = {"GMM-guided SMOTE": "GMM-SMOTE"}
    for method in method_order:
        values = noise_table.query(
            "dataset == @dataset and method == @method"
        ).sort_values("sigma")
        if values.empty:
            continue
        color = METHOD_COLORS[method]
        ax.plot(
            values["sigma"],
            values["sep_mean"],
            color=color,
            marker="o",
            markersize=4.2,
            linewidth=1.8,
            label=display_names.get(method, method),
        )
        if "sep_sd" in values:
            lower = np.clip(values["sep_mean"] - values["sep_sd"], 0.0, 1.0)
            upper = np.clip(values["sep_mean"] + values["sep_sd"], 0.0, 1.0)
            ax.fill_between(
                values["sigma"],
                lower,
                upper,
                color=color,
                alpha=0.14,
                linewidth=0,
            )

    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.15)
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel(r"Noise level $\sigma$", fontsize=11.0)
    ax.set_ylabel(r"$\langle\mathrm{AUC}\rangle$", fontsize=11.0)
    ax.tick_params(axis="both", labelsize=9.0, direction="out")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.75, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="center right",
        frameon=False,
        fontsize=8.2,
        ncol=2,
        columnspacing=1.0,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.965, bottom=0.10)
    return apply_notebook_figure_style(fig)


def compute_reverse_ablation(datasets, cohorts, repeats=3, seed=42):
    rows = []
    for dataset, method_data in cohorts.items():
        X_real = np.asarray(datasets[dataset]["X"])
        y_real = np.asarray(datasets[dataset]["y"], dtype=int)
        for method, (X_syn, y_syn) in method_data.items():
            ranking = rank_discriminating_features(X_real, X_syn, seed=seed)
            for removed in ablation_grid(X_real.shape[1]):
                keep = ranking[int(removed):]
                values = [
                    one_run_origin_auc(
                        X_real[:, keep],
                        y_real,
                        np.asarray(X_syn)[:, keep],
                        y_syn,
                        seed=seed + 1009 * repeat,
                    )
                    for repeat in range(int(repeats))
                ]
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "n_features_removed": int(removed),
                        "percent_removed": 100 * removed / X_real.shape[1],
                        "auc_mean": float(np.mean(values)),
                        "auc_sd": float(np.std(values)),
                    }
                )
    return pd.DataFrame(rows)


def plot_reverse_ablation(table):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1), sharey=True)
    for ax, dataset in zip(axes, table["dataset"].unique()):
        for method in METHOD_ORDER:
            values = table.query(
                "dataset == @dataset and method == @method"
            ).sort_values("percent_removed")
            ax.plot(
                values["percent_removed"],
                values["auc_mean"],
                color=METHOD_COLORS[method],
                marker="o",
                markersize=3,
                label=method,
            )
            if "auc_sd" in values:
                lower = np.clip(values["auc_mean"] - values["auc_sd"], 0.0, 1.0)
                upper = np.clip(values["auc_mean"] + values["auc_sd"], 0.0, 1.0)
                ax.fill_between(
                    values["percent_removed"],
                    lower,
                    upper,
                    color=METHOD_COLORS[method],
                    alpha=0.13,
                    linewidth=0,
                )
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(dataset, color=DATASET_COLORS[dataset], weight="bold")
        ax.set_xlabel("Features removed (%)")
        ax.set_ylabel("Origin AUC")
    axes[-1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return apply_notebook_figure_style(fig)


def _plot_hiv_experimental_1_with_path(
    datasets,
    cohorts,
    ablation_table,
    edge_status,
    dataset="HIV",
):
    """Legacy Experimental 1 layout retained for reproducibility."""
    methods = [method for method in METHOD_ORDER if method in cohorts[dataset]]
    structures = edge_status.structures[dataset]
    path_table = edge_status.regularization_path
    if path_table is None or path_table.empty:
        raise ValueError("edge_status must contain the HIV regularization path.")

    fig = plt.figure(figsize=(8.27 / 1.18, 11.30 / 1.18), facecolor="white")
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.16, 0.90, 0.72],
        left=0.075,
        right=0.985,
        top=0.975,
        bottom=0.125,
        hspace=0.23,
    )

    # A: wide PCA comparison using the open 2 x 3 design.
    pca_grid = outer[0].subgridspec(2, 3, wspace=0.24, hspace=0.42)
    X_real = np.asarray(datasets[dataset]["X"])
    pca_axes = []
    for index, method in enumerate(methods):
        ax = fig.add_subplot(pca_grid[index // 3, index % 3])
        pca_axes.append(ax)
        X_syn = np.asarray(cohorts[dataset][method][0])
        Xr, Xs = standardize_pair(X_real, X_syn)
        pca = PCA(n_components=2, random_state=42).fit(Xr)
        Zr, Zs = pca.transform(Xr), pca.transform(Xs)
        ax.scatter(
            Zr[:, 0],
            Zr[:, 1],
            s=5.0,
            facecolors="none",
            edgecolors="#777777",
            linewidths=0.55,
            alpha=0.50,
            label="Real",
        )
        ax.scatter(
            Zs[:, 0],
            Zs[:, 1],
            s=5.0,
            color=METHOD_COLORS[method],
            edgecolors="none",
            alpha=0.60,
            label=method,
        )
        add_confidence_ellipse(ax, Zr, "#777777", linewidth=1.15)
        add_confidence_ellipse(ax, Zs, METHOD_COLORS[method], linewidth=1.35)
        ax.set_title(
            "GMM-SMOTE" if method == "GMM-guided SMOTE" else method,
            color=METHOD_COLORS[method],
            fontsize=7.5,
            weight="bold",
            pad=2.5,
        )
        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
            fontsize=5.8,
            labelpad=1.0,
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
            fontsize=5.8,
            labelpad=1.0,
        )
        ax.tick_params(axis="both", labelsize=5.2, length=2.0, pad=1.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=5.2,
            markerscale=1.15,
            handletextpad=0.30,
            borderaxespad=0.25,
        )
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("#444444")
    pca_axes[0].text(
        -0.18,
        1.02,
        "A",
        transform=pca_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11.0,
        weight="bold",
        clip_on=False,
    )

    middle = outer[1].subgridspec(1, 2, width_ratios=[0.38, 0.62], wspace=0.18)

    # B: HIV reverse ablation.
    ablation_ax = fig.add_subplot(middle[0, 0])
    for method in methods:
        values = ablation_table.query(
            "dataset == @dataset and method == @method"
        ).sort_values("percent_removed")
        ablation_ax.plot(
            values["percent_removed"],
            values["auc_mean"],
            color=METHOD_COLORS[method],
            marker="o",
            markersize=2.3,
            linewidth=1.05,
            label="GMM-SMOTE" if method == "GMM-guided SMOTE" else method,
        )
        if "auc_sd" in values:
            lower = np.clip(values["auc_mean"] - values["auc_sd"], 0.0, 1.0)
            upper = np.clip(values["auc_mean"] + values["auc_sd"], 0.0, 1.0)
            ablation_ax.fill_between(
                values["percent_removed"],
                lower,
                upper,
                color=METHOD_COLORS[method],
                alpha=0.14,
                linewidth=0,
                zorder=0,
            )
    ablation_ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.85)
    ablation_ax.set_xlabel("Features removed (%)", fontsize=7.0)
    ablation_ax.set_ylabel("AUC", fontsize=7.0)
    ablation_ax.tick_params(axis="both", labelsize=6.2)
    ablation_ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    ablation_ax.legend(
        loc="best",
        frameon=False,
        fontsize=5.5,
        ncol=2,
        columnspacing=0.7,
        handlelength=1.4,
    )
    ablation_ax.spines["top"].set_visible(False)
    ablation_ax.spines["right"].set_visible(False)
    ablation_ax.text(
        -0.15,
        1.035,
        "B",
        transform=ablation_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.0,
        weight="bold",
        clip_on=False,
    )

    # C: six categorical edge-status matrices.
    matrix_grid = middle[0, 1].subgridspec(2, 3, wspace=0.13, hspace=0.24)
    real = structures["real"]
    order = get_real_structure_order(real["partial"])
    n_features = real["partial"].shape[0]
    tick_step = 10 if n_features <= 70 else 20
    ticks = np.arange(0, n_features, tick_step)
    tick_labels = [str(value + 1) for value in ticks]
    status_cmap = ListedColormap([
        STATUS_COLORS["absent"],
        STATUS_COLORS["preserved"],
        STATUS_COLORS["real_only"],
        STATUS_COLORS["synthetic_only"],
    ])
    matrix_axes = []
    for index, method in enumerate(methods):
        row, col = divmod(index, 3)
        ax = fig.add_subplot(matrix_grid[row, col])
        matrix_axes.append(ax)
        syn_edges = structures["synthetic"][method]["edges"]
        status = build_edge_status_matrix(real["edges"], syn_edges, n_features)
        ax.imshow(
            status[np.ix_(order, order)],
            cmap=status_cmap,
            vmin=-0.5,
            vmax=3.5,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(
            "GMM-SMOTE" if method == "GMM-guided SMOTE" else method,
            color=METHOD_COLORS[method],
            fontsize=6.2,
            weight="bold",
            pad=1.8,
        )
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels if row == 1 else [], fontsize=4.6)
        ax.set_yticklabels(tick_labels if col == 0 else [], fontsize=4.6)
        ax.tick_params(axis="both", length=1.5, width=0.55, pad=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("#444444")
    matrix_axes[0].text(
        -0.18,
        1.18,
        "C",
        transform=matrix_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11.0,
        weight="bold",
        clip_on=False,
    )

    status_handles = [
        Patch(facecolor=STATUS_COLORS["preserved"], edgecolor="#333333", label="Preserved"),
        Patch(facecolor=STATUS_COLORS["real_only"], edgecolor="#333333", label="Real-only"),
        Patch(facecolor=STATUS_COLORS["synthetic_only"], edgecolor="#333333", label="Synthetic-only"),
        Patch(facecolor=STATUS_COLORS["absent"], edgecolor="#C9CDD2", label="Absent"),
    ]
    matrix_axes[-2].legend(
        handles=status_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=4,
        frameon=False,
        fontsize=4.8,
        handlelength=1.2,
        handletextpad=0.3,
        columnspacing=0.55,
    )

    # D: full-width HIV regularization path.
    path_ax = fig.add_subplot(outer[2])
    edge_columns = ["feature_a_matrix_index", "feature_b_matrix_index"]
    grouped_paths = list(path_table.groupby(edge_columns, sort=False))
    path_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(grouped_paths)))
    for color, ((edge_a, edge_b), values) in zip(path_colors, grouped_paths):
        values = values.sort_values("alpha")
        path_ax.plot(
            values["alpha"],
            values["precision_coefficient"],
            color=color,
            linewidth=1.25,
            label=f"Edge ({int(edge_a)}, {int(edge_b)})",
        )
    selected_alpha = float(path_table["selected_alpha"].iloc[0])
    path_ax.axvline(
        selected_alpha,
        color="#222222",
        linestyle="--",
        linewidth=1.1,
        label=rf"Analysis $\alpha={selected_alpha:g}$",
    )
    path_ax.axhline(0, color="#777777", linewidth=0.7, alpha=0.75)
    path_ax.set_xscale("log")
    path_ax.invert_xaxis()
    path_ax.set_xlabel(r"Regularization parameter $\alpha$", fontsize=7.5)
    path_ax.set_ylabel("Precision coefficient", fontsize=7.5)
    path_ax.tick_params(axis="both", labelsize=6.4)
    path_ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.38)
    path_ax.text(
        -0.045,
        1.035,
        "D",
        transform=path_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.0,
        weight="bold",
        clip_on=False,
    )
    path_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        frameon=False,
        fontsize=5.5,
        ncol=4,
        handlelength=1.7,
        columnspacing=0.8,
    )
    return apply_notebook_figure_style(fig)


def plot_hiv_experimental_1(
    datasets,
    cohorts,
    ablation_table,
    edge_status,
    dataset="HIV",
):
    """Experimental 1: wide edge matrices above ablation and PCA."""
    methods = [method for method in METHOD_ORDER if method in cohorts[dataset]]
    structures = edge_status.structures[dataset]
    fig = plt.figure(figsize=(8.27 / 1.18, 9.55 / 1.18), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.35, 0.78],
        left=0.055,
        right=0.995,
        top=0.965,
        bottom=0.075,
        hspace=0.25,
    )

    # A: the structural result receives the full figure width.
    matrix_grid = outer[0].subgridspec(2, 3, wspace=0.08, hspace=0.17)
    real = structures["real"]
    order = get_real_structure_order(real["partial"])
    n_features = real["partial"].shape[0]
    ticks = np.arange(0, n_features, 10 if n_features <= 70 else 20)
    tick_labels = [str(value + 1) for value in ticks]
    status_cmap = ListedColormap([
        STATUS_COLORS["absent"],
        STATUS_COLORS["preserved"],
        STATUS_COLORS["real_only"],
        STATUS_COLORS["synthetic_only"],
    ])
    matrix_axes = []
    for index, method in enumerate(methods):
        row, col = divmod(index, 3)
        ax = fig.add_subplot(matrix_grid[row, col])
        matrix_axes.append(ax)
        syn_edges = structures["synthetic"][method]["edges"]
        status = build_edge_status_matrix(real["edges"], syn_edges, n_features)
        ax.imshow(
            status[np.ix_(order, order)],
            cmap=status_cmap,
            vmin=-0.5,
            vmax=3.5,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(
            "GMM-SMOTE" if method == "GMM-guided SMOTE" else method,
            color=METHOD_COLORS[method],
            fontsize=9,
            weight="bold",
            pad=2.5,
        )
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels if row == 1 else [], fontsize=5.0)
        ax.set_yticklabels(tick_labels if col == 0 else [], fontsize=5.0)
        ax.tick_params(axis="both", length=1.7, width=0.6, pad=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("#444444")
    matrix_axes[0].text(
        -0.13, 1.08, "A", transform=matrix_axes[0].transAxes,
        ha="left", va="bottom", fontsize=11.0, weight="bold", clip_on=False,
    )
    status_handles = [
        Patch(facecolor=STATUS_COLORS["preserved"], edgecolor="#333333", label="Preserved"),
        Patch(facecolor=STATUS_COLORS["real_only"], edgecolor="#333333", label="Real-only"),
        Patch(facecolor=STATUS_COLORS["synthetic_only"], edgecolor="#333333", label="Synthetic-only"),
        Patch(facecolor=STATUS_COLORS["absent"], edgecolor="#C9CDD2", label="Absent"),
    ]
    fig.legend(
        handles=status_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.42),
        ncol=4,
        frameon=False,
        fontsize=7,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    lower = outer[1].subgridspec(1, 2, width_ratios=[0.38, 0.62], wspace=0.18)

    # B: reverse ablation.
    ablation_ax = fig.add_subplot(lower[0, 0])
    for method in methods:
        values = ablation_table.query(
            "dataset == @dataset and method == @method"
        ).sort_values("n_features_removed")
        color = METHOD_COLORS[method]
        ablation_ax.plot(
            values["n_features_removed"], values["auc_mean"],
            color=color, marker="o", markersize=2.3, linewidth=1.05,
            label="GMM-SMOTE" if method == "GMM-guided SMOTE" else method,
        )
        if "auc_sd" in values:
            lower_auc = np.clip(values["auc_mean"] - values["auc_sd"], 0.0, 1.0)
            upper_auc = np.clip(values["auc_mean"] + values["auc_sd"], 0.0, 1.0)
            ablation_ax.fill_between(
                values["n_features_removed"], lower_auc, upper_auc,
                color=color, alpha=0.14, linewidth=0,
            )
    ablation_ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.85)
    feature_ticks = values["n_features_removed"].to_numpy(dtype=int)
    ablation_ax.set_xticks(feature_ticks)
    ablation_ax.set_xlabel("Features removed", fontsize=7.0)
    ablation_ax.set_ylabel("AUC", fontsize=7.0)
    ablation_ax.tick_params(axis="both", labelsize=6.2)
    ablation_ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.65)
    ablation_ax.legend(
        loc="best", frameon=False, fontsize=5.4, ncol=2,
        columnspacing=0.7, handlelength=1.4,
    )
    ablation_ax.spines["top"].set_visible(False)
    ablation_ax.spines["right"].set_visible(False)
    ablation_ax.text(
        -0.15, 1.035, "B", transform=ablation_ax.transAxes,
        ha="left", va="bottom", fontsize=11.0, weight="bold", clip_on=False,
    )

    # C: compact 2 x 3 PCA comparison.
    pca_grid = lower[0, 1].subgridspec(2, 3, wspace=0.16, hspace=0.28)
    X_real = np.asarray(datasets[dataset]["X"])
    pca_axes = []
    for index, method in enumerate(methods):
        row, col = divmod(index, 3)
        ax = fig.add_subplot(pca_grid[row, col])
        pca_axes.append(ax)
        X_syn = np.asarray(cohorts[dataset][method][0])
        Xr, Xs = standardize_pair(X_real, X_syn)
        pca = PCA(n_components=2, random_state=42).fit(Xr)
        Zr, Zs = pca.transform(Xr), pca.transform(Xs)
        ax.scatter(
            Zr[:, 0], Zr[:, 1], s=3.2, facecolors="none",
            edgecolors="#777777", linewidths=0.4, alpha=0.45, label="Real",
        )
        ax.scatter(
            Zs[:, 0], Zs[:, 1], s=3.2, color=METHOD_COLORS[method],
            edgecolors="none", alpha=0.58, label=method,
        )
        add_confidence_ellipse(ax, Zr, "#777777", linewidth=0.8)
        add_confidence_ellipse(ax, Zs, METHOD_COLORS[method], linewidth=0.95)
        ax.set_title(
            "GMM-SMOTE" if method == "GMM-guided SMOTE" else method,
            color=METHOD_COLORS[method], fontsize=7, weight="bold", pad=1.5,
        )
        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)" if row == 1 else "",
            fontsize=6.5, labelpad=0.6,
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)" if col == 0 else "",
            fontsize=6.5, labelpad=0.6,
        )
        ax.tick_params(axis="both", labelsize=3.9, length=1.4, pad=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.legend(
            loc="upper left", frameon=True, fontsize=5, facecolor="white",
            markerscale=2, handletextpad=0.2, borderaxespad=0.15, ncol=2, columnspacing=0.8,  framealpha=0.9,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color("#444444")
    pca_axes[0].text(
        -0.16, 1.12, "C", transform=pca_axes[0].transAxes,
        ha="left", va="bottom", fontsize=11.0, weight="bold", clip_on=False,
    )
    return apply_notebook_figure_style(fig)


def build_glasso_feature_order_table(datasets, edge_status, dataset_order=None):
    """Map clustered Graphical Lasso matrix positions to source features."""
    dataset_order = list(dataset_order or datasets.keys())
    rows = []
    for dataset in dataset_order:
        if dataset not in edge_status.structures:
            raise KeyError(f"No Graphical Lasso structure is available for {dataset!r}.")
        feature_names = list(datasets[dataset]["feature_names"])
        real_partial = edge_status.structures[dataset]["real"]["partial"]
        order = get_real_structure_order(real_partial)
        if len(feature_names) != len(order):
            raise ValueError(
                f"{dataset!r} has {len(feature_names)} feature names but "
                f"{len(order)} Graphical Lasso variables."
            )
        rows.extend(
            {
                "dataset": dataset,
                "matrix_index": matrix_index,
                "original_feature_index": int(original_index) + 1,
                "feature_name": feature_names[int(original_index)],
            }
            for matrix_index, original_index in enumerate(order, start=1)
        )
    return pd.DataFrame(rows)


def glasso_feature_order_to_latex(feature_order_table):
    """Render the complete Graphical Lasso feature mapping as a longtable."""
    latex_table = feature_order_table.rename(columns={
        "dataset": "Dataset",
        "matrix_index": "Matrix index",
        "original_feature_index": "Original index",
        "feature_name": "Feature",
    })
    latex = latex_table.to_latex(
        index=False,
        longtable=True,
        escape=True,
        caption=(
            "Feature ordering used in the Graphical Lasso edge-comparison "
            "matrices. For each dataset, matrix positions were obtained by "
            "average-linkage hierarchical clustering of the absolute "
            "partial-correlation structure estimated from the real data, "
            "using dissimilarity $d_{ij}=1-|\\rho^{\\mathrm{partial}}_{ij}|$. "
            "The same real-derived order was applied to every synthetic method."
        ),
        label="tab:glasso_feature_order",
        column_format="lrrp{0.48\\textwidth}",
    )
    return "% Requires \\usepackage{booktabs,longtable}\n" + latex


def plot_glasso_path_supplement(edge_status):
    """Plot every HIV Graphical Lasso edge path for the supplement."""
    path_table = edge_status.regularization_path
    if path_table is None or path_table.empty:
        raise ValueError("edge_status must contain a regularization path.")
    fig = plt.figure(figsize=(8.27 / 1.18, 5.05 / 1.18), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[3.25, 1.15], hspace=0.08)
    ax = fig.add_subplot(grid[0])
    survival_ax = fig.add_subplot(grid[1], sharex=ax)
    edge_columns = ["feature_a_matrix_index", "feature_b_matrix_index"]
    grouped_paths = list(path_table.groupby(edge_columns, sort=False))
    selected_alpha = float(path_table["selected_alpha"].iloc[0])
    if "selected_nonzero" not in path_table.columns:
        selected_flags = {}
        for edge_key, values in grouped_paths:
            nearest = values.iloc[(values["alpha"] - selected_alpha).abs().argmin()]
            selected_flags[edge_key] = abs(float(nearest["precision_coefficient"])) > 1e-7
    else:
        selected_flags = {
            edge_key: bool(values["selected_nonzero"].iloc[0])
            for edge_key, values in grouped_paths
        }

    total_edges = len(grouped_paths)
    retained_edges = sum(selected_flags.values())
    removed_edges = total_edges - retained_edges
    retained_pct = 100.0 * retained_edges / total_edges
    removed_pct = 100.0 * removed_edges / total_edges

    # Draw retained paths first, then place the much more numerous paths that
    # are zero at the selected lambda on top.  A warm retained-path color keeps
    # this panel distinct from the blue retained-area encoding below.
    for selected_state in (True, False):
        for edge_key, values in grouped_paths:
            if selected_flags[edge_key] != selected_state:
                continue
            values = values.sort_values("alpha")
            ax.plot(
                np.log(values["alpha"].to_numpy(dtype=float)),
                values["precision_coefficient"],
                color="#C46A2D" if selected_state else "#6F7782",
                linewidth=0.72 if selected_state else 0.48,
                alpha=0.48 if selected_state else 0.28,
                zorder=1 if selected_state else 2,
            )
    ax.axvline(
        np.log(selected_alpha),
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        zorder=3,
    )
    ax.axhline(0, color="#777777", linewidth=0.7, alpha=0.75)
    ax.set_ylabel("Precision coefficient", fontsize=8.0)
    ax.tick_params(axis="both", labelsize=6.8)
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.38)

    # Show the complete sparsification process directly rather than relying on
    # the overlapping coefficient paths to communicate how many edges survive.
    remaining_by_alpha = {
        float(alpha): int(np.count_nonzero(values.to_numpy(dtype=float) > 1e-7))
        for alpha, values in path_table.assign(
            absolute_coefficient=path_table["precision_coefficient"].abs()
        ).groupby("alpha")["absolute_coefficient"]
    }
    remaining_by_alpha[selected_alpha] = retained_edges
    survival_alphas = np.array(sorted(remaining_by_alpha), dtype=float)
    survival_pct = np.array(
        [100.0 * remaining_by_alpha[alpha] / total_edges for alpha in survival_alphas],
        dtype=float,
    )
    log_survival_alphas = np.log(survival_alphas)
    selected_log_alpha = np.log(selected_alpha)
    # The gray envelope is the complete candidate-edge set (100%); blue is the
    # portion still nonzero.  The gray area left above the curve therefore
    # directly shows the edges removed by regularization.
    survival_ax.fill_between(
        log_survival_alphas, 0, 100,
        color="#D9DEE3", alpha=0.78, linewidth=0, zorder=0,
    )
    survival_ax.fill_between(
        log_survival_alphas, 0, survival_pct,
        color="#C46A2D", alpha=0.78, linewidth=0, zorder=1,
    )
    survival_ax.plot(
        log_survival_alphas, survival_pct,
        color="#C46A2D", linewidth=1.15, zorder=2,
    )
    survival_ax.axvline(
        selected_log_alpha, color="#222222", linestyle="--", linewidth=1.2,
    )
    survival_ax.scatter(
        [selected_log_alpha], [retained_pct], s=24,
        color="#C46A2D", edgecolor="white", linewidth=0.7, zorder=4,
    )
    survival_ax.annotate(
        f"{retained_edges:,} retained ({retained_pct:.1f}%)\n"
        f"{removed_edges:,} removed ({removed_pct:.1f}%)",
        xy=(selected_log_alpha, retained_pct),
        xytext=(selected_log_alpha + 0.22, 58),
        ha="left", va="center", fontsize=6.2, color="#333333",
        bbox=dict(facecolor="white", edgecolor="#D6DADF", alpha=0.94, pad=2.2),
        arrowprops=dict(arrowstyle="-", color="#C46A2D", linewidth=0.8),
    )
    survival_ax.set_ylim(0, 100)
    survival_ax.set_yticks([0, 25, 50, 75, 100])
    survival_ax.set_ylabel("Edges retained (%)", fontsize=7.1)
    survival_ax.set_xlabel(r"$\log(\lambda)$", fontsize=8.0)
    survival_ax.tick_params(axis="both", labelsize=6.6)
    survival_ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.36)

    survival_ax.spines["top"].set_visible(False)
    survival_ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#6F7782", linewidth=1.1, alpha=0.85, label=r"Zero at selected $\lambda$"),
            Line2D([0], [0], color="#C46A2D", linewidth=1.7, label=r"Non-zero at selected $\lambda$"),
            Line2D([0], [0], color="#222222", linewidth=1.2, linestyle="--", label=rf"Selected $\lambda={selected_alpha:g}$"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.60),
        frameon=False,
        fontsize=6.2,
        ncol=3,
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.22)
    return apply_notebook_figure_style(fig)


def plot_hiv_experimental_3(edge_status, dataset="HIV", method_order=None):
    """Experimental 3: a contiguous 2 x 3 categorical edge-status plate."""
    structures = edge_status.structures[dataset]
    methods = [
        method for method in (method_order or METHOD_ORDER)
        if method in structures["synthetic"]
    ]
    if len(methods) != 6:
        raise ValueError("Experimental 3 requires exactly six synthesis methods.")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(8.27 / 1.18, 6.25 / 1.18),
        squeeze=False,
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    real = structures["real"]
    order = get_real_structure_order(real["partial"])
    n_features = real["partial"].shape[0]
    tick_step = 10 if n_features <= 70 else 20
    ticks = np.arange(0, n_features, tick_step)
    tick_labels = [str(value + 1) for value in ticks]
    status_cmap = ListedColormap([
        STATUS_COLORS["absent"],
        STATUS_COLORS["preserved"],
        STATUS_COLORS["real_only"],
        STATUS_COLORS["synthetic_only"],
    ])

    for index, (ax, method) in enumerate(zip(axes.ravel(), methods)):
        row, col = divmod(index, 3)
        syn_edges = structures["synthetic"][method]["edges"]
        status = build_edge_status_matrix(real["edges"], syn_edges, n_features)
        ax.imshow(
            status[np.ix_(order, order)],
            cmap=status_cmap,
            vmin=-0.5,
            vmax=3.5,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=5.8)
        ax.set_yticklabels(tick_labels if col == 0 else [], fontsize=5.8)
        ax.tick_params(
            axis="x",
            top=row == 0,
            labeltop=row == 0,
            bottom=row == 1,
            labelbottom=row == 1,
            direction="out",
            length=2.0,
            width=0.7,
            pad=1.0,
        )
        ax.tick_params(
            axis="y",
            left=col == 0,
            labelleft=col == 0,
            right=False,
            direction="out",
            length=2.0,
            width=0.7,
            pad=1.0,
        )
        ax.text(
            0.025,
            0.025,
            "GMM-SMOTE" if method == "GMM-guided SMOTE" else method,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            weight="bold",
            color=METHOD_COLORS[method],
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
                "pad": 1.2,
            },
            zorder=5,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.85)
            spine.set_color("#333333")

    legend_handles = [
        Patch(facecolor=STATUS_COLORS["preserved"], edgecolor="#333333", label="Preserved edge"),
        Patch(facecolor=STATUS_COLORS["real_only"], edgecolor="#333333", label="Real-only / lost"),
        Patch(facecolor=STATUS_COLORS["synthetic_only"], edgecolor="#333333", label="Synthetic-only"),
        Patch(facecolor=STATUS_COLORS["absent"], edgecolor="#C9CDD2", label="Absent in both"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=4,
        frameon=False,
        fontsize=7.2,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.965, bottom=0.115, wspace=0.0, hspace=0.0)
    return apply_notebook_figure_style(fig)


def plot_hiv_experimental_2(
    datasets,
    cohorts,
    ablation_table,
    noise_table,
    dataset="HIV",
):
    """Experimental 2: wide PCA above ablation and noise sensitivity."""
    methods = [method for method in METHOD_ORDER if method in cohorts[dataset]]
    fig = plt.figure(figsize=(8.27 / 1.18, 8.35 / 1.18), facecolor="white")
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.16, 0.88],
        left=0.075,
        right=0.985,
        top=0.965,
        bottom=0.090,
        hspace=0.25,
    )

    pca_grid = outer[0].subgridspec(2, 3, wspace=0.24, hspace=0.42)
    X_real = np.asarray(datasets[dataset]["X"])
    pca_axes = []
    for index, method in enumerate(methods):
        ax = fig.add_subplot(pca_grid[index // 3, index % 3])
        pca_axes.append(ax)
        X_syn = np.asarray(cohorts[dataset][method][0])
        Xr, Xs = standardize_pair(X_real, X_syn)
        pca = PCA(n_components=2, random_state=42).fit(Xr)
        Zr, Zs = pca.transform(Xr), pca.transform(Xs)
        ax.scatter(
            Zr[:, 0], Zr[:, 1], s=5.0, facecolors="none",
            edgecolors="#777777", linewidths=0.55, alpha=0.50, label="Real",
        )
        ax.scatter(
            Zs[:, 0], Zs[:, 1], s=5.0, color=METHOD_COLORS[method],
            edgecolors="none", alpha=0.60, label=method,
        )
        add_confidence_ellipse(ax, Zr, "#777777", linewidth=1.15)
        add_confidence_ellipse(ax, Zs, METHOD_COLORS[method], linewidth=1.35)
        ax.set_title(
            "GMM-SMOTE" if method == "GMM-guided SMOTE" else method,
            color=METHOD_COLORS[method], fontsize=7.5, weight="bold", pad=2.5,
        )
        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
            fontsize=5.8, labelpad=1.0,
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
            fontsize=5.8, labelpad=1.0,
        )
        ax.tick_params(axis="both", labelsize=5.2, length=2.0, pad=1.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.legend(
            loc="upper right", frameon=False, fontsize=5.2,
            markerscale=1.15, handletextpad=0.30, borderaxespad=0.25,
        )
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("#444444")
    pca_axes[0].text(
        -0.18, 1.02, "A", transform=pca_axes[0].transAxes,
        ha="left", va="bottom", fontsize=11.0, weight="bold", clip_on=False,
    )

    lower = outer[1].subgridspec(1, 2, wspace=0.24)
    ablation_ax = fig.add_subplot(lower[0, 0])
    noise_ax = fig.add_subplot(lower[0, 1])

    for method in methods:
        color = METHOD_COLORS[method]
        label = "GMM-SMOTE" if method == "GMM-guided SMOTE" else method

        ablation_values = ablation_table.query(
            "dataset == @dataset and method == @method"
        ).sort_values("n_features_removed")
        ablation_ax.plot(
            ablation_values["n_features_removed"],
            ablation_values["auc_mean"],
            color=color,
            marker="o",
            markersize=2.4,
            linewidth=1.1,
            label=label,
        )
        if "auc_sd" in ablation_values:
            lower_auc = np.clip(
                ablation_values["auc_mean"] - ablation_values["auc_sd"], 0.0, 1.0
            )
            upper_auc = np.clip(
                ablation_values["auc_mean"] + ablation_values["auc_sd"], 0.0, 1.0
            )
            ablation_ax.fill_between(
                ablation_values["n_features_removed"],
                lower_auc,
                upper_auc,
                color=color,
                alpha=0.14,
                linewidth=0,
            )

        noise_values = noise_table.query(
            "dataset == @dataset and method == @method"
        ).sort_values("sigma")
        noise_ax.plot(
            noise_values["sigma"],
            noise_values["sep_mean"],
            color=color,
            marker="o",
            markersize=2.4,
            linewidth=1.1,
            label=label,
        )
        if "sep_sd" in noise_values:
            lower_auc = np.clip(
                noise_values["sep_mean"] - noise_values["sep_sd"], 0.0, 1.0
            )
            upper_auc = np.clip(
                noise_values["sep_mean"] + noise_values["sep_sd"], 0.0, 1.0
            )
            noise_ax.fill_between(
                noise_values["sigma"],
                lower_auc,
                upper_auc,
                color=color,
                alpha=0.14,
                linewidth=0,
            )

    for ax, panel in ((ablation_ax, "B"), (noise_ax, "C")):
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=0.85)
        ax.set_ylabel("AUC", fontsize=7.2)
        ax.tick_params(axis="both", labelsize=6.4)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.14, 1.035, panel, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11.0, weight="bold", clip_on=False,
        )
    feature_ticks = ablation_values["n_features_removed"].to_numpy(dtype=int)
    ablation_ax.set_xticks(feature_ticks)
    ablation_ax.set_xlabel("Features removed", fontsize=7.2)
    noise_ax.set_xlabel(r"Noise level $\sigma$", fontsize=7.2)
    noise_ax.legend(
        loc="best", frameon=False, fontsize=5.8, ncol=2,
        columnspacing=0.8, handlelength=1.5,
    )
    return apply_notebook_figure_style(fig)


def format_hiv_structural_cd(edge_status):
    """Relabel the standalone structural figure as matrix panel C above path D."""
    fig = edge_status.fig
    axes = fig.axes
    if len(axes) < 7:
        raise ValueError("Expected six matrix axes followed by one path axis.")
    for ax in axes:
        for text_artist in list(ax.texts):
            if text_artist.get_text() in set("ABCDEFG"):
                text_artist.remove()
    axes[0].text(
        -0.10, 1.035, "C", transform=axes[0].transAxes,
        ha="left", va="bottom", fontsize=10.8, weight="bold",
        color="#111111", clip_on=False,
    )
    axes[-1].text(
        -0.035, 1.035, "D", transform=axes[-1].transAxes,
        ha="left", va="bottom", fontsize=10.8, weight="bold",
        color="#111111", clip_on=False,
    )
    return fig


def compute_sample_size_sensitivity(
    datasets,
    methods=METHOD_ORDER,
    fractions=(0.25, 0.5, 1.0),
    repeats=2,
    seed=42,
    cvae_epochs=25,
    wgan_epochs=25,
):
    rows = []
    for dataset, data in datasets.items():
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=int)
        for fraction in fractions:
            n0 = max(2, int(np.sum(y == 0) * fraction))
            n1 = max(2, int(np.sum(y == 1) * fraction))
            X_sub, y_sub = stratified_subsample(X, y, n0, n1, seed=seed)
            for offset, method in enumerate(methods):
                X_syn, y_syn = sample_method(
                    X_sub,
                    y_sub,
                    method,
                    seed=seed + 101 * offset,
                    cvae_epochs=cvae_epochs,
                    wgan_epochs=wgan_epochs,
                )
                for repeat in range(int(repeats)):
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "fraction": float(fraction),
                            "separability_auc": one_run_origin_auc(
                                X_sub,
                                y_sub,
                                X_syn,
                                y_syn,
                                seed=seed + 1009 * repeat,
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def plot_sample_size_sensitivity(table):
    summary = (
        table.groupby(["dataset", "method", "fraction"])["separability_auc"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1), sharey=True)
    for ax, dataset in zip(axes, table["dataset"].unique()):
        for method in METHOD_ORDER:
            values = summary.query(
                "dataset == @dataset and method == @method"
            ).sort_values("fraction")
            ax.errorbar(
                100 * values["fraction"],
                values["mean"],
                yerr=values["std"].fillna(0),
                color=METHOD_COLORS[method],
                marker="o",
                label=method,
            )
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(dataset, color=DATASET_COLORS[dataset], weight="bold")
        ax.set_xlabel("Real data used to train generator (%)")
        ax.set_ylabel("Origin AUC")
    axes[-1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return apply_notebook_figure_style(fig), summary
