"""Graph-guided, marginal-preserving repairs for origin-separability tests."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.model_selection import train_test_split

from src.revision.common import METHOD_COLORS, NEUTRAL, SEED
from src.revision.figure4_graphical_lasso import FIGURE4_ALPHAS
from src.revision.figure4_graphical_lasso_plots import (
    fit_glasso_precision,
    precision_to_partial_corr,
)
from src.revision.stats import one_run_origin_auc


ERROR_CATEGORIES = ("real_only", "synthetic_only", "reversed", "changed")
CONTROL_COLORS = {
    "baseline": NEUTRAL,
    "targeted": "#C62828",
    "random": "#7B7B7B",
    "preserved_matched": "#2F6DB3",
    "marginal_only": "#009E73",
}


def _as_float_array(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Expected a two-dimensional feature matrix.")
    return X


def gaussian_rank_scores(X):
    """Map columns to normal scores using ranks, without using feature scale."""
    X = _as_float_array(X)
    n = X.shape[0]
    if n < 2:
        raise ValueError("At least two rows are required.")
    Z = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        ranks = rankdata(X[:, j], method="average")
        probabilities = np.clip((ranks - 0.5) / n, 1e-6, 1 - 1e-6)
        Z[:, j] = norm.ppf(probabilities)
    return Z


def _correlation_matrix(X):
    corr = np.corrcoef(_as_float_array(X), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    return corr


def _matrix_power_psd(matrix, power, eigen_floor=1e-8):
    matrix = (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T) / 2
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, eigen_floor, None) ** power
    return (vectors * values) @ vectors.T


def nearest_correlation(matrix, eigen_floor=1e-6):
    """Project a symmetric matrix to a positive-definite correlation matrix."""
    matrix = (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T) / 2
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, eigen_floor, None)
    projected = (vectors * values) @ vectors.T
    scale = np.sqrt(np.clip(np.diag(projected), eigen_floor, None))
    projected = projected / np.outer(scale, scale)
    projected = (projected + projected.T) / 2
    np.fill_diagonal(projected, 1.0)
    return projected


def _rank_reorder(original, scores):
    """Reorder each original column by scores, preserving its exact multiset."""
    original = _as_float_array(original)
    scores = _as_float_array(scores)
    if original.shape != scores.shape:
        raise ValueError("Original values and ranking scores must have the same shape.")
    repaired = np.empty_like(original)
    for j in range(original.shape[1]):
        order = np.argsort(scores[:, j], kind="mergesort")
        repaired[order, j] = np.sort(original[:, j], kind="mergesort")
    return repaired


def marginal_multiset_error(before, after):
    """Maximum absolute change between sorted columns; zero means exact preservation."""
    before = _as_float_array(before)
    after = _as_float_array(after)
    if before.shape != after.shape:
        return np.inf
    return float(np.max(np.abs(np.sort(before, axis=0) - np.sort(after, axis=0))))


def selected_pair_copula_error(X_real, X_candidate, pairs):
    """Mean absolute Gaussian-rank correlation error over selected pairs."""
    pairs = list(pairs)
    if not pairs:
        return np.nan
    real_corr = _correlation_matrix(gaussian_rank_scores(X_real))
    candidate_corr = _correlation_matrix(gaussian_rank_scores(X_candidate))
    return float(
        np.mean([abs(real_corr[i, j] - candidate_corr[i, j]) for i, j in pairs])
    )


def copula_edge_repair(
    X_apply,
    X_syn_reference,
    X_real_reference,
    pairs: Iterable[tuple[int, int]],
    strength=1.0,
):
    """Repair selected rank-dependence entries while preserving all marginals.

    The dependency transform is learned from reference rows. Selected entries
    of the synthetic Gaussian-copula correlation matrix are moved toward the
    corresponding real entries, then the original values in ``X_apply`` are
    rank-reordered according to the transformed latent scores.
    """
    X_apply = _as_float_array(X_apply)
    X_syn_reference = _as_float_array(X_syn_reference)
    X_real_reference = _as_float_array(X_real_reference)
    pairs = list(dict.fromkeys(tuple(sorted(map(int, pair))) for pair in pairs))
    if not pairs or strength <= 0:
        return X_apply.copy()
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1.")

    syn_corr = _correlation_matrix(gaussian_rank_scores(X_syn_reference))
    real_corr = _correlation_matrix(gaussian_rank_scores(X_real_reference))
    target_corr = syn_corr.copy()
    for i, j in pairs:
        target = (1 - strength) * syn_corr[i, j] + strength * real_corr[i, j]
        target_corr[i, j] = target
        target_corr[j, i] = target
    target_corr = nearest_correlation(target_corr)

    whitening = _matrix_power_psd(syn_corr, -0.5)
    coloring = _matrix_power_psd(target_corr, 0.5)
    transformed_scores = gaussian_rank_scores(X_apply) @ whitening @ coloring
    return _rank_reorder(X_apply, transformed_scores)


def quantile_match_marginals(X_apply, X_real_reference):
    """Match each marginal to the real reference without targeting dependence."""
    X_apply = _as_float_array(X_apply)
    X_real_reference = _as_float_array(X_real_reference)
    n = X_apply.shape[0]
    repaired = np.empty_like(X_apply)
    probabilities = (np.arange(n) + 0.5) / n
    for j in range(X_apply.shape[1]):
        target_values = np.quantile(X_real_reference[:, j], probabilities)
        order = np.argsort(X_apply[:, j], kind="mergesort")
        repaired[order, j] = target_values
    return repaired


def build_edge_discrepancy_table(
    X_real,
    X_syn,
    feature_names=None,
    alpha=None,
    edge_threshold=1e-7,
    changed_threshold=0.05,
):
    """Classify Graphical Lasso edges and rank structural discrepancies."""
    X_real = _as_float_array(X_real)
    X_syn = _as_float_array(X_syn)
    if X_real.shape[1] != X_syn.shape[1]:
        raise ValueError("Real and synthetic matrices must have the same feature count.")
    if alpha is None:
        raise ValueError("Provide the Graphical Lasso alpha used for the dataset.")

    p = X_real.shape[1]
    feature_names = list(feature_names or [f"feature_{i + 1}" for i in range(p)])
    real_partial = precision_to_partial_corr(fit_glasso_precision(X_real, alpha))
    syn_partial = precision_to_partial_corr(fit_glasso_precision(X_syn, alpha))

    rows = []
    for i in range(p):
        for j in range(i + 1, p):
            real_value = float(real_partial[i, j])
            syn_value = float(syn_partial[i, j])
            real_present = abs(real_value) > edge_threshold
            syn_present = abs(syn_value) > edge_threshold
            delta = abs(real_value - syn_value)

            if real_present and not syn_present:
                category = "real_only"
            elif syn_present and not real_present:
                category = "synthetic_only"
            elif real_present and syn_present and np.sign(real_value) != np.sign(syn_value):
                category = "reversed"
            elif real_present and syn_present and delta >= changed_threshold:
                category = "changed"
            elif real_present and syn_present:
                category = "preserved"
            else:
                category = "absent"

            rows.append(
                {
                    "i": i,
                    "j": j,
                    "feature_i": feature_names[i],
                    "feature_j": feature_names[j],
                    "category": category,
                    "real_partial": real_value,
                    "synthetic_partial": syn_value,
                    "abs_partial_error": delta,
                    "real_abs_strength": abs(real_value),
                    "synthetic_abs_strength": abs(syn_value),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["abs_partial_error", "real_abs_strength"],
        ascending=[False, False],
        ignore_index=True,
    )


def _pairs(frame):
    return [tuple(row) for row in frame[["i", "j"]].to_numpy(dtype=int)]


def select_target_pairs(edge_table, dose):
    candidates = edge_table[edge_table["category"].isin(ERROR_CATEGORIES)]
    return _pairs(candidates.head(int(dose)))


def select_random_pairs(edge_table, target_pairs, dose, seed):
    target_pairs = set(target_pairs)
    candidates = edge_table[
        ~edge_table.apply(lambda row: (int(row["i"]), int(row["j"])) in target_pairs, axis=1)
    ]
    if candidates.empty or dose <= 0:
        return []
    return _pairs(candidates.sample(n=min(int(dose), len(candidates)), random_state=seed))


def select_preserved_matched_pairs(edge_table, target_pairs, dose):
    """Match controls to target pairs by real absolute partial-correlation size."""
    preserved = edge_table[edge_table["category"] == "preserved"].copy()
    if preserved.empty or dose <= 0:
        return []
    target_lookup = edge_table.set_index(["i", "j"])
    available = preserved.copy()
    selected = []
    for pair in list(target_pairs)[: int(dose)]:
        if available.empty:
            break
        target_strength = float(target_lookup.loc[pair, "real_abs_strength"])
        index = (available["real_abs_strength"] - target_strength).abs().idxmin()
        row = available.loc[index]
        selected.append((int(row["i"]), int(row["j"])))
        available = available.drop(index)
    return selected


def _stratified_discovery_indices(y, discovery_fraction, seed):
    indices = np.arange(len(y))
    discovery, evaluation = train_test_split(
        indices,
        train_size=discovery_fraction,
        stratify=np.asarray(y, dtype=int),
        random_state=seed,
    )
    return np.asarray(discovery), np.asarray(evaluation)


def _evaluate_auc_repeats(X_real, y_real, X_syn, y_syn, repeats, seed):
    values = []
    for repeat in range(repeats):
        values.append(
            one_run_origin_auc(
                X_real,
                y_real,
                X_syn,
                y_syn,
                seed=seed + repeat * 1009,
            )
        )
    return values


def run_graph_guided_repair_experiment(
    dataset_name,
    method,
    X_real,
    y_real,
    X_syn,
    y_syn,
    feature_names=None,
    doses=(0, 1, 5, 10, 20),
    split_repeats=3,
    auc_repeats=10,
    discovery_fraction=0.5,
    edge_threshold=1e-7,
    changed_threshold=0.05,
    repair_strength=1.0,
    seed=SEED,
):
    """Run cross-fitted targeted repairs and negative controls."""
    X_real = _as_float_array(X_real)
    X_syn = _as_float_array(X_syn)
    y_real = np.asarray(y_real, dtype=int)
    y_syn = np.asarray(y_syn, dtype=int)
    alpha = FIGURE4_ALPHAS[dataset_name]
    rows = []
    edge_tables = []

    for split in range(split_repeats):
        split_seed = seed + split * 7919
        real_discovery, real_evaluation = _stratified_discovery_indices(
            y_real, discovery_fraction, split_seed
        )
        syn_discovery, syn_evaluation = _stratified_discovery_indices(
            y_syn, discovery_fraction, split_seed + 37
        )
        Xr_disc, Xr_eval = X_real[real_discovery], X_real[real_evaluation]
        Xs_disc, Xs_eval = X_syn[syn_discovery], X_syn[syn_evaluation]
        yr_eval, ys_eval = y_real[real_evaluation], y_syn[syn_evaluation]

        edge_table = build_edge_discrepancy_table(
            Xr_disc,
            Xs_disc,
            feature_names=feature_names,
            alpha=alpha,
            edge_threshold=edge_threshold,
            changed_threshold=changed_threshold,
        ).assign(split=split)
        edge_tables.append(edge_table)

        maximum_dose = max(int(dose) for dose in doses)
        full_target_pairs = select_target_pairs(edge_table, maximum_dose)
        baseline_auc_values = _evaluate_auc_repeats(
            Xr_eval,
            yr_eval,
            Xs_eval,
            ys_eval,
            repeats=auc_repeats,
            seed=split_seed,
        )
        marginal_repaired = quantile_match_marginals(Xs_eval, Xr_disc)
        marginal_auc_values = _evaluate_auc_repeats(
            Xr_eval,
            yr_eval,
            marginal_repaired,
            ys_eval,
            repeats=auc_repeats,
            seed=split_seed,
        )

        for dose in doses:
            dose = int(dose)
            target_pairs = full_target_pairs[:dose]
            random_pairs = select_random_pairs(
                edge_table, full_target_pairs, dose, split_seed + dose
            )
            preserved_pairs = select_preserved_matched_pairs(
                edge_table, target_pairs, dose
            )
            variants = {
                "targeted": copula_edge_repair(
                    Xs_eval, Xs_eval, Xr_disc, target_pairs, strength=repair_strength
                ),
                "random": copula_edge_repair(
                    Xs_eval, Xs_eval, Xr_disc, random_pairs, strength=repair_strength
                ),
                "preserved_matched": copula_edge_repair(
                    Xs_eval, Xs_eval, Xr_disc, preserved_pairs, strength=repair_strength
                ),
            }
            condition_auc_values = {
                "baseline": baseline_auc_values,
                "marginal_only": marginal_auc_values,
            }
            for condition, repaired in variants.items():
                condition_auc_values[condition] = (
                    baseline_auc_values
                    if dose == 0
                    else _evaluate_auc_repeats(
                        Xr_eval,
                        yr_eval,
                        repaired,
                        ys_eval,
                        repeats=auc_repeats,
                        seed=split_seed,
                    )
                )

            all_variants = {
                "baseline": Xs_eval,
                **variants,
                "marginal_only": marginal_repaired,
            }
            for condition, repaired in all_variants.items():
                condition_pairs = {
                    "targeted": target_pairs,
                    "random": random_pairs,
                    "preserved_matched": preserved_pairs,
                }.get(condition, [])
                pair_error_before = selected_pair_copula_error(
                    Xr_disc, Xs_eval, condition_pairs
                )
                pair_error_after = selected_pair_copula_error(
                    Xr_disc, repaired, condition_pairs
                )
                for repeat, auc in enumerate(condition_auc_values[condition]):
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "method": method,
                            "split": split,
                            "dose": dose,
                            "condition": condition,
                            "auc_repeat": repeat,
                            "auc": auc,
                            "auc_reduction": np.nan,
                            "n_pairs": len(condition_pairs),
                            "selected_pair_copula_error_before": pair_error_before,
                            "selected_pair_copula_error_after": pair_error_after,
                            "selected_pair_copula_error_reduction": (
                                pair_error_before - pair_error_after
                                if np.isfinite(pair_error_before)
                                and np.isfinite(pair_error_after)
                                else np.nan
                            ),
                            "marginal_multiset_error": (
                                np.nan
                                if condition == "marginal_only"
                                else marginal_multiset_error(Xs_eval, repaired)
                            ),
                        }
                    )

    results = pd.DataFrame(rows)
    baseline = (
        results[results["condition"] == "baseline"]
        .set_index(["split", "dose", "auc_repeat"])["auc"]
        .rename("baseline_auc")
    )
    results = results.join(baseline, on=["split", "dose", "auc_repeat"])
    results["auc_reduction"] = results["baseline_auc"] - results["auc"]
    return results, pd.concat(edge_tables, ignore_index=True)


def summarize_repair_results(results):
    return (
        results.groupby(["dataset", "method", "condition", "dose"], as_index=False)
        .agg(
            mean_auc=("auc", "mean"),
            sd_auc=("auc", "std"),
            mean_auc_reduction=("auc_reduction", "mean"),
            sd_auc_reduction=("auc_reduction", "std"),
            runs=("auc", "size"),
            max_marginal_error=("marginal_multiset_error", "max"),
            mean_pair_error_reduction=(
                "selected_pair_copula_error_reduction",
                "mean",
            ),
        )
        .sort_values(["dataset", "method", "condition", "dose"])
    )


def plot_edge_discrepancy_map(edge_table, feature_names, title=None):
    """Plot preserved and erroneous Graphical Lasso relationships."""
    feature_names = list(feature_names)
    p = len(feature_names)
    category_order = [
        "absent",
        "preserved",
        "real_only",
        "synthetic_only",
        "reversed",
        "changed",
    ]
    category_colors = {
        "absent": "#FFFFFF",
        "preserved": "#2F6DB3",
        "real_only": "#C43C39",
        "synthetic_only": "#E88925",
        "reversed": "#8E44AD",
        "changed": "#46A5A5",
    }
    code = {category: index for index, category in enumerate(category_order)}
    matrix = np.zeros((p, p), dtype=int)
    for row in edge_table.itertuples():
        value = code[row.category]
        matrix[int(row.i), int(row.j)] = value
        matrix[int(row.j), int(row.i)] = value

    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    ax.imshow(
        matrix,
        cmap=ListedColormap([category_colors[c] for c in category_order]),
        vmin=-0.5,
        vmax=len(category_order) - 0.5,
        interpolation="nearest",
    )
    if p <= 30:
        labels = [str(name)[:24] for name in feature_names]
        ax.set_xticks(np.arange(p))
        ax.set_yticks(np.arange(p))
        ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
        ax.set_yticklabels(labels, fontsize=6.5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{p} features")
        ax.set_ylabel(f"{p} features")
    ax.set_title(title or "Graphical Lasso edge-status map", loc="left", weight="bold")
    handles = [
        Patch(facecolor=category_colors[category], edgecolor="#555555", label=category.replace("_", " "))
        for category in category_order[1:]
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout()
    return fig


def plot_repair_dose_response(results, title=None):
    """Plot held-out AUC and change from baseline for each repair condition."""
    summary = summarize_repair_results(results)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    condition_order = [
        "targeted",
        "random",
        "preserved_matched",
        "marginal_only",
    ]
    labels = {
        "targeted": "Graph-targeted repair",
        "random": "Random-pair control",
        "preserved_matched": "Preserved-edge control",
        "marginal_only": "Marginal-only repair",
    }

    baseline = summary[summary["condition"] == "baseline"]
    axes[0].plot(
        baseline["dose"],
        baseline["mean_auc"],
        color=CONTROL_COLORS["baseline"],
        linewidth=2,
        linestyle="--",
        label="Baseline",
    )
    for condition in condition_order:
        sub = summary[summary["condition"] == condition]
        if sub.empty:
            continue
        axes[0].errorbar(
            sub["dose"],
            sub["mean_auc"],
            yerr=sub["sd_auc"],
            marker="o",
            capsize=3,
            linewidth=2,
            color=CONTROL_COLORS[condition],
            label=labels[condition],
        )
        axes[1].errorbar(
            sub["dose"],
            sub["mean_auc_reduction"],
            yerr=sub["sd_auc_reduction"],
            marker="o",
            capsize=3,
            linewidth=2,
            color=CONTROL_COLORS[condition],
            label=labels[condition],
        )

    axes[0].axhline(0.5, color="#888888", linestyle=":", linewidth=1.2)
    axes[0].set_ylabel("Held-out RF origin AUC")
    axes[1].axhline(0, color="#888888", linestyle=":", linewidth=1.2)
    axes[1].set_ylabel("AUC reduction from baseline")
    for ax in axes:
        ax.set_xlabel("Number of repaired feature relationships")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8.5)
    if title:
        fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout()
    return fig, summary


def generator_color(method):
    return METHOD_COLORS.get(method, NEUTRAL)
