"""Screen CVAE latent priors with the existing marginal-fidelity diagnostics.

Run from ``data_synthesis`` so the project's existing imports resolve:

    python -m src.revision.cvae_prior_ablation --epochs 200 --repeats 1

The seed-42 heatmap uses the same raw Welch t-tests as the revision notebook.
CSV outputs also include KS, Wasserstein, and standardized mean differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, wilcoxon

from models.cvae import sample_cvae
from src.revision.common import DATASET_ORDER, class_counts, load_datasets, repo_root
from src.revision.marginal_diagnostics import (
    build_marginal_diagnostics,
    compute_marginal_test_table,
)
from util.config import Config


PRIOR_LABELS = {
    "normal": "Normal",
    "standardized_lognormal": "Log-normal",
    "class_conditional_gmm": "Class GMM",
}


def _augment_effect_metrics(table, real_data, synthetic_data):
    table = table.copy()
    smd = []
    standardized_wasserstein = []
    for row in table.itertuples(index=False):
        feature_index = int(row.feature_index) - 1
        real = np.asarray(real_data[row.dataset])[:, feature_index]
        synthetic = np.asarray(synthetic_data[row.dataset][row.method])[:, feature_index]
        real = real[np.isfinite(real)]
        synthetic = synthetic[np.isfinite(synthetic)]
        real_sd = float(np.std(real, ddof=1)) if len(real) > 1 else np.nan
        scale = real_sd if np.isfinite(real_sd) and real_sd > 0 else 1.0
        smd.append(float((np.mean(synthetic) - np.mean(real)) / scale))
        standardized_wasserstein.append(
            float(wasserstein_distance(real, synthetic) / scale)
        )
    table["standardized_mean_difference"] = smd
    table["standardized_wasserstein"] = standardized_wasserstein
    return table


def _summarize(feature_results):
    grouped = feature_results.groupby(["dataset", "prior"], sort=False)
    summary = grouped.agg(
        repeats=("repeat", "nunique"),
        feature_tests=("p_value", "size"),
        prop_p_below_0_05=("p_value", lambda x: float((x < 0.05).mean())),
        median_p_value=("p_value", "median"),
        mean_abs_standardized_mean_difference=(
            "standardized_mean_difference", lambda x: float(np.mean(np.abs(x)))
        ),
        median_ks=("ks_statistic", "median"),
        mean_ks=("ks_statistic", "mean"),
        mean_standardized_wasserstein=("standardized_wasserstein", "mean"),
    ).reset_index()

    robustness = (
        feature_results.assign(significant=lambda x: x["p_value"] < 0.05)
        .groupby(["dataset", "prior", "feature"], sort=False)["significant"]
        .mean()
        .rename("significant_repeat_fraction")
        .reset_index()
    )
    robust_summary = (
        robustness.groupby(["dataset", "prior"], sort=False)
        .agg(
            prop_features_significant_majority=(
                "significant_repeat_fraction", lambda x: float((x >= 0.5).mean())
            ),
            prop_features_significant_every_repeat=(
                "significant_repeat_fraction", lambda x: float((x == 1.0).mean())
            ),
        )
        .reset_index()
    )
    return summary.merge(robust_summary, on=["dataset", "prior"], how="left"), robustness


def _paired_comparison(feature_results):
    """Compare each alternative with normal on the same seed and feature."""
    keys = ["dataset", "repeat", "feature"]
    normal = feature_results[feature_results["prior"] == "Normal"].set_index(keys)
    metrics = {
        "standardized_mean_difference": "abs_smd",
        "ks_statistic": "ks",
        "standardized_wasserstein": "wasserstein",
    }
    rows = []
    alternatives = feature_results[feature_results["prior"] != "Normal"]
    for (dataset, prior), group in alternatives.groupby(
        ["dataset", "prior"], sort=False
    ):
        alternative = group.set_index(keys)
        baseline = normal.loc[alternative.index]
        row = {"dataset": dataset, "prior": prior}
        for metric, short_name in metrics.items():
            baseline_values = baseline[metric].abs().to_numpy()
            alternative_values = alternative[metric].abs().to_numpy()
            differences = alternative_values - baseline_values
            try:
                paired_p = float(wilcoxon(differences).pvalue)
            except ValueError:
                paired_p = np.nan
            row[f"{short_name}_relative_change"] = float(
                alternative_values.mean() / baseline_values.mean() - 1.0
            )
            row[f"{short_name}_fraction_improved"] = float(
                (differences < 0).mean()
            )
            row[f"{short_name}_wilcoxon_p"] = paired_p
        rows.append(row)
    return pd.DataFrame(rows)


def run_ablation(epochs=200, repeats=1, seed=42, output_dir=None, verbose=False):
    datasets = load_datasets()
    dataset_order = [name for name in DATASET_ORDER if name in datasets]
    prior_order = list(PRIOR_LABELS)
    method_order = [PRIOR_LABELS[prior] for prior in prior_order]

    output_dir = Path(
        output_dir
        or repo_root / "data_synthesis" / "output" / "cvae_prior_ablation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_feature_results = []
    seed_real_data = {}
    seed_synthetic_data = {}
    feature_names = {}

    for repeat in range(repeats):
        run_seed = seed + 1009 * repeat
        for dataset in dataset_order:
            data = datasets[dataset]
            X = np.asarray(data["X"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=int)
            n0, n1 = class_counts(y)
            real_one = {dataset: X}
            synthetic_one = {dataset: {}}
            names_one = {dataset: list(data["feature_names"])}

            for prior in prior_order:
                label = PRIOR_LABELS[prior]
                print(
                    f"[{repeat + 1}/{repeats}] {dataset}: {label} "
                    f"({epochs} epochs, seed={run_seed})",
                    flush=True,
                )
                cfg = Config(
                    seed=run_seed,
                    epochs=epochs,
                    # This is a supplementary sensitivity analysis and retains
                    # the preprocessing used for its existing reported output.
                    x_transform="log1p",
                    latent_prior=prior,
                    prior_components=2,
                )
                X_syn, _ = sample_cvae(
                    X, y, n0, n1, seed=run_seed, cfg=cfg, verbose=verbose
                )
                synthetic_one[dataset][label] = X_syn

            table = compute_marginal_test_table(
                real_data=real_one,
                synthetic_data=synthetic_one,
                feature_names=names_one,
                dataset_order=[dataset],
                method_order=method_order,
            )
            table = _augment_effect_metrics(table, real_one, synthetic_one)
            table.insert(0, "seed", run_seed)
            table.insert(0, "repeat", repeat)
            table = table.rename(columns={"method": "prior"})
            all_feature_results.append(table)

            if repeat == 0:
                seed_real_data[dataset] = X
                seed_synthetic_data[dataset] = synthetic_one[dataset]
                feature_names[dataset] = names_one[dataset]

    feature_results = pd.concat(all_feature_results, ignore_index=True)
    summary, robustness = _summarize(feature_results)
    paired_comparison = _paired_comparison(feature_results)

    feature_results.to_csv(output_dir / "feature_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    robustness.to_csv(output_dir / "feature_robustness.csv", index=False)
    paired_comparison.to_csv(output_dir / "paired_vs_normal.csv", index=False)

    diagnostics = build_marginal_diagnostics(
        real_data=seed_real_data,
        synthetic_data=seed_synthetic_data,
        feature_names=feature_names,
        dataset_order=dataset_order,
        method_order=method_order,
    )
    diagnostics.heatmap.savefig(
        output_dir / f"marginal_pvalue_heatmap_seed{seed}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(diagnostics.heatmap)
    for dataset, figure in diagnostics.overlap_figures.items():
        slug = dataset.lower().replace(" ", "_")
        figure.savefig(
            output_dir / f"marginal_overlap_{slug}_seed{seed}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)

    print("\nMarginal-fidelity summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved results to {output_dir}")
    return feature_results, summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ablation(
        epochs=args.epochs,
        repeats=args.repeats,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
