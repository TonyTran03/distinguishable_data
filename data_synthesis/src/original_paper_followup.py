"""Expanded, t-SNE-free follow-up evaluation for the original paper."""

from __future__ import annotations

import contextlib
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
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
    class_counts,
    standardize_pair,
)
from src.revision.figure4_graphical_lasso import FIGURE4_ALPHAS
from src.revision.figure4_graphical_lasso_plots import (
    compute_edge_recovery,
    compute_frobenius_deviation,
    compute_synthetic_only_rate,
    fit_glasso_precision,
    get_edge_set,
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
    fig.suptitle(
        "Real-versus-synthetic discriminator AUC",
        fontsize=14,
        weight="bold",
        y=0.97,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.86,
        bottom=0.29,
        wspace=0.12,
    )
    return fig


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

    fig.suptitle(
        "Feature-wise Kullback-Leibler divergence",
        fontsize=14,
        weight="bold",
        y=0.97,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.86,
        bottom=0.29,
        wspace=0.20,
    )
    return fig


def compute_metric_table(datasets, cohorts, auc_runs, tstr_repeats=3, seed=42):
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
                repeats=tstr_repeats,
            )
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
    methods = list(cohorts[dataset])
    columns = 3
    rows = math.ceil(len(methods) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12.6, 3.7 * rows))
    axes = np.atleast_1d(axes).ravel()
    X_real = np.asarray(datasets[dataset]["X"])
    for ax, method in zip(axes, methods):
        X_syn = cohorts[dataset][method][0]
        Xr, Xs = standardize_pair(X_real, X_syn)
        pca = PCA(n_components=2, random_state=42).fit(Xr)
        Zr, Zs = pca.transform(Xr), pca.transform(Xs)
        ax.scatter(
            Zr[:, 0], Zr[:, 1], s=8, facecolors="none",
            edgecolors="#777777", alpha=0.55, label="Real",
        )
        ax.scatter(
            Zs[:, 0], Zs[:, 1], s=8, color=METHOD_COLORS[method],
            alpha=0.6, label=method,
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
        ax.set_title(method, color=METHOD_COLORS[method], weight="bold")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[len(methods):]:
        ax.axis("off")
    fig.suptitle(f"{dataset}: PCA of real and synthetic observations", weight="bold")
    fig.tight_layout()
    return fig


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
    return fig


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
    return fig


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
    return fig


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
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(dataset, color=DATASET_COLORS[dataset], weight="bold")
        ax.set_xlabel("RF-ranked features removed (%)")
        ax.set_ylabel("Origin AUC")
    axes[-1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
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
    return fig, summary
