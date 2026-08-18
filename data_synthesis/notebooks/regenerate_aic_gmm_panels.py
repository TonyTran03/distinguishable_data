"""Regenerate the two overview figures with Table S5 AIC-selected GMMs."""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "data_synthesis"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from models.gmm import AIC_COMPONENTS_BY_DATASET
from src import original_paper_followup as followup
from src.revision import data_io
from src.revision.common import DATASET_COLORS, add_confidence_ellipse, standardize_pair


SEED = 42
GMM_SEED = SEED + 101 * followup.METHOD_ORDER.index("GMM")
CACHE_DIR = PKG_ROOT / "notebooks" / "followup_cache"
EXPORT_DIR = PKG_ROOT / "notebooks" / "followup_exports"
BASE_FINGERPRINT = "fed3894e75"


def load_cache(name):
    csv_names = {
        "origin_auc": "origin_auc_runs_final.csv",
        "feature_kld": "feature_kld_final.csv",
        "marginal_tests": "marginal_tests_final.csv",
        "tstr_runs": "tstr_runs_final.csv",
    }
    if name in csv_names:
        return pd.read_csv(EXPORT_DIR / csv_names[name])
    with (CACHE_DIR / f"{name}_{BASE_FINGERPRINT}.pkl").open("rb") as handle:
        return pickle.load(handle)


def replace_method_rows(base, replacement, method="GMM"):
    return pd.concat(
        [base.loc[base["method"] != method], replacement],
        ignore_index=True,
    )


def build_aic_cohorts(datasets):
    cohorts = copy.deepcopy(load_cache("cohorts"))
    for dataset, data in datasets.items():
        cohorts[dataset]["GMM"] = followup.sample_method(
            np.asarray(data["X"], dtype=np.float32),
            np.asarray(data["y"], dtype=int),
            "GMM",
            seed=GMM_SEED,
            dataset=dataset,
        )
    return cohorts


def plot_pca_rf_overview(datasets, cohorts, auc_runs):
    """CVAE PCA examples above origin-AUC distributions for four generators."""
    dataset_order = list(datasets)
    auc_methods = ["Bootstrap", "Column-wise", "GMM", "CVAE"]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.1), squeeze=False)

    for col, dataset in enumerate(dataset_order):
        pca_ax = axes[0, col]
        auc_ax = axes[1, col]
        X_real = np.asarray(datasets[dataset]["X"])
        X_syn = np.asarray(cohorts[dataset]["CVAE"][0])
        Xr, Xs = standardize_pair(X_real, X_syn)
        pca = PCA(n_components=2, random_state=SEED).fit(Xr)
        Zr, Zs = pca.transform(Xr), pca.transform(Xs)

        pca_ax.scatter(
            Zr[:, 0], Zr[:, 1], s=8, facecolors="none", edgecolors="#777777",
            linewidths=0.55, alpha=0.48, label="Real data",
        )
        pca_ax.scatter(
            Zs[:, 0], Zs[:, 1], s=8, color=DATASET_COLORS[dataset],
            edgecolors="none", alpha=0.62, label="CVAE synthetic data",
        )
        add_confidence_ellipse(pca_ax, Zr, "#888888", linewidth=1.8)
        add_confidence_ellipse(
            pca_ax, Zs, DATASET_COLORS[dataset], linewidth=1.8
        )
        pca_ax.set_title(
            dataset, color=DATASET_COLORS[dataset], weight="bold", fontsize=13
        )
        pca_ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
        pca_ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
        pca_ax.legend(frameon=True, fontsize=7.5, loc="upper left")
        pca_ax.grid(False)
        pca_ax.text(
            -0.12, 1.04, f"{chr(65 + col)}1", transform=pca_ax.transAxes,
            fontsize=14, weight="bold", va="bottom",
        )

        subset = auc_runs[auc_runs["dataset"] == dataset]
        values = [
            subset.loc[subset["method"] == method, "separability_auc"]
            .dropna().to_numpy(dtype=float)
            for method in auc_methods
        ]
        positions = np.arange(len(auc_methods))
        violins = auc_ax.violinplot(
            values, positions=positions, widths=0.78,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for body, method in zip(violins["bodies"], auc_methods):
            body.set_facecolor(followup.METHOD_COLORS[method])
            body.set_edgecolor(followup.METHOD_COLORS[method])
            body.set_alpha(0.42)
        for position, method, method_values in zip(positions, auc_methods, values):
            mean = float(np.mean(method_values))
            sd = float(np.std(method_values, ddof=1))
            auc_ax.errorbar(
                position, mean, yerr=sd, fmt="o", markersize=5,
                color=followup.METHOD_COLORS[method], markerfacecolor="white",
                capsize=3, linewidth=1.4, zorder=4,
            )
        auc_ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0)
        auc_ax.set_ylim(0.47, 1.02)
        auc_ax.set_xticks(positions)
        auc_ax.set_xticklabels(auc_methods, rotation=20, ha="right", fontsize=8)
        auc_ax.set_ylabel("Origin AUC")
        auc_ax.set_title("RF separability", weight="bold", fontsize=11.5)
        auc_ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.55)
        auc_ax.text(
            -0.12, 1.04, f"{chr(65 + col)}2", transform=auc_ax.transAxes,
            fontsize=14, weight="bold", va="bottom",
        )

    fig.subplots_adjust(
        left=0.065, right=0.99, top=0.94, bottom=0.12,
        wspace=0.25, hspace=0.42,
    )
    return fig


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    datasets, _ = data_io.initialize_datasets()
    cohorts = build_aic_cohorts(datasets)
    gmm_only = {dataset: {"GMM": values["GMM"]} for dataset, values in cohorts.items()}

    print("Recomputing GMM origin AUC (50 repeats)...", flush=True)
    gmm_auc = followup.compute_origin_auc(datasets, gmm_only, repeats=50, seed=SEED)
    auc_runs = replace_method_rows(load_cache("origin_auc"), gmm_auc)

    print("Recomputing GMM feature KLD and marginal tests...", flush=True)
    gmm_kld = followup.compute_feature_kld_table(datasets, gmm_only)
    feature_kld = replace_method_rows(load_cache("feature_kld"), gmm_kld)
    gmm_marginal = followup.compute_marginal_tests(datasets, gmm_only)
    marginal_tests = replace_method_rows(load_cache("marginal_tests"), gmm_marginal)

    print("Recomputing GMM utility (20 repeats)...", flush=True)
    gmm_tstr = followup.compute_tstr_runs(datasets, gmm_only, repeats=20, seed=SEED)
    tstr_runs = replace_method_rows(load_cache("tstr_runs"), gmm_tstr)
    # The legacy CSV predates this plotting column; derive it for all methods
    # after merging so non-GMM rows are not left as NaN.
    tstr_runs["utility_gap_abs"] = (
        tstr_runs["trtr_f1"] - tstr_runs["tstr_f1"]
    ).abs()

    fig1 = followup.plot_figure1_fidelity_grid(
        auc_runs,
        feature_kld,
        marginal_tests,
        tstr_runs,
        dataset_order=list(datasets),
        method_order=followup.METHOD_ORDER,
        jitter_seed=SEED,
    )
    fig2 = plot_pca_rf_overview(datasets, cohorts, auc_runs)

    figure1_path = EXPORT_DIR / "figure1_aic_gmm_auc_kld_utility.png"
    figure2_path = EXPORT_DIR / "figure2_aic_gmm_pca_rf_auc.png"
    fig1.savefig(figure1_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig2.savefig(figure2_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    plt.close(fig2)

    auc_runs.to_csv(EXPORT_DIR / "aic_gmm_origin_auc_runs.csv", index=False)
    feature_kld.to_csv(EXPORT_DIR / "aic_gmm_feature_kld.csv", index=False)
    tstr_runs.to_csv(EXPORT_DIR / "aic_gmm_tstr_runs.csv", index=False)
    pd.DataFrame(
        [
            {"dataset": dataset, "class_0_k": ks[0], "class_1_k": ks[1]}
            for dataset, ks in AIC_COMPONENTS_BY_DATASET.items()
        ]
    ).to_csv(EXPORT_DIR / "aic_gmm_components.csv", index=False)

    summary = (
        auc_runs.groupby(["dataset", "method"])["separability_auc"]
        .agg(mean="mean", sd="std")
        .reset_index()
    )
    summary.to_csv(EXPORT_DIR / "aic_gmm_auc_summary.csv", index=False)
    print(f"Saved {figure1_path}")
    print(f"Saved {figure2_path}")


if __name__ == "__main__":
    main()
