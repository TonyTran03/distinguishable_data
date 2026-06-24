"""Figure 6: reverse-ablation and feature-removal sensitivity."""

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.revision.common import *
from src.revision.stats import ablation_grid, one_run_origin_auc, rank_discriminating_features
from src.revision.cache import _read_cache, _write_cache
from src.revision.figure4_graphical_lasso import FIGURE4_ALPHAS
from src.revision.figure4_graphical_lasso_plots import (
    EDGE_COLORS,
    STATUS_COLORS,
    build_edge_status_matrix,
    compute_edge_recovery,
    compute_frobenius_deviation,
    compute_synthetic_only_rate,
    fit_glasso_precision,
    get_edge_set,
    get_real_structure_order,
    precision_to_partial_corr,
    _fit_profile_tsne,
    _short_label,
)

def compute_reverse_ablation(datasets, seed=SEED, repeats=ABLATION_REPEATS, cvae_epochs=CVAE_EPOCHS):
    rows = []
    for ds in DATASET_ORDER:
        data = datasets[ds]
        X_real = np.asarray(data["X"], dtype=np.float32)
        y_real = np.asarray(data["y"], dtype=int)
        grid = ablation_grid(X_real.shape[1])
        for method in METHOD_ORDER:
            print(f"[ablation] {ds} - {method}")
            X_syn, y_syn = sample_synthetic(ds, data, method, seed=seed, cvae_epochs=cvae_epochs)
            ranking = rank_discriminating_features(X_real, X_syn, seed=seed)
            for n_removed in grid:
                keep = ranking[int(n_removed):]
                vals = [one_run_origin_auc(X_real[:, keep], y_real, np.asarray(X_syn)[:, keep], y_syn,
                                           seed=seed + 1009*r + 13*METHOD_ORDER.index(method))
                        for r in range(repeats)]
                rows.append({"dataset": ds, "method": method, "n_features_removed": int(n_removed),
                             "n_features_retained": int(len(keep)), "auc_mean": float(np.mean(vals)),
                             "auc_sd": float(np.std(vals)), "auc_values": vals})
    return pd.DataFrame(rows)

def get_reverse_ablation(force=False):
    cached = None if force else _read_cache("ablation_df")
    if cached is not None:
        return cached
    return _write_cache("ablation_df", compute_reverse_ablation(require_datasets()))


def discriminator_feature_importance(X_real, X_syn, seed=SEED):
    """Return RF-origin feature importances and ranks."""
    Xr, Xs = standardize_pair(X_real, X_syn)
    X = np.vstack([Xr, Xs])
    origin = np.r_[np.zeros(len(Xr), dtype=int), np.ones(len(Xs), dtype=int)]
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X, origin)
    importance = np.asarray(rf.feature_importances_, dtype=float)
    ranking = np.argsort(importance)[::-1]
    ranks = np.empty_like(ranking)
    ranks[ranking] = np.arange(1, len(ranking) + 1)
    return importance, ranks, ranking


def compute_discriminator_importance_table(
    datasets,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
):
    """Compute quantitative RF-origin feature importances for each dataset/method."""
    rows = []
    for ds in DATASET_ORDER:
        data = datasets[ds]
        X_real = np.asarray(data["X"], dtype=np.float32)
        feature_names = list(data.get("feature_names", [f"feature_{i + 1}" for i in range(X_real.shape[1])]))
        for method in METHOD_ORDER:
            print(f"[RF importance] {ds} - {method}")
            X_syn, _ = sample_synthetic(ds, data, method, seed=seed, cvae_epochs=cvae_epochs)
            importance, ranks, _ = discriminator_feature_importance(X_real, X_syn, seed=seed)
            total = float(np.sum(importance))
            normalized = importance / total if total > 0 else np.zeros_like(importance)
            for feature_index, value in enumerate(normalized):
                rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "feature_index": int(feature_index),
                        "feature_name": feature_names[feature_index],
                        "rf_importance": float(value),
                        "rf_rank": int(ranks[feature_index]),
                    }
                )
    df = pd.DataFrame(rows)
    df["cumulative_rf_importance"] = (
        df.sort_values(["dataset", "method", "rf_rank"])
        .groupby(["dataset", "method"])["rf_importance"]
        .cumsum()
    )
    return df


def get_discriminator_importance_table(force=False):
    cached = None if force else _read_cache("discriminator_importance")
    if cached is not None:
        return cached
    return _write_cache(
        "discriminator_importance",
        compute_discriminator_importance_table(require_datasets()),
    )


def summarize_discriminator_importance(importance_df):
    rows = []
    for (dataset, method), sub in importance_df.groupby(["dataset", "method"]):
        ranked = sub.sort_values("rf_rank")
        values = ranked["rf_importance"].to_numpy(dtype=float)
        cumulative = np.cumsum(values)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "top1_importance": float(cumulative[min(0, len(cumulative) - 1)]),
                "top3_importance": float(cumulative[min(2, len(cumulative) - 1)]),
                "top5_importance": float(cumulative[min(4, len(cumulative) - 1)]),
                "top10_importance": float(cumulative[min(9, len(cumulative) - 1)]),
                "n_features_for_50pct_importance": int(np.searchsorted(cumulative, 0.50) + 1),
                "n_features_for_80pct_importance": int(np.searchsorted(cumulative, 0.80) + 1),
                "n_features": int(len(values)),
                "importance_entropy": float(
                    -np.sum(values[values > 0] * np.log(values[values > 0]))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "method"])


def plot_discriminator_importance_concentration(importance_df):
    """Plot cumulative RF-origin importance by feature rank."""
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(13.4, 3.55), sharey=True)
    for ax, ds, panel in zip(np.ravel(axes), DATASET_ORDER, ["A", "B", "C"]):
        sub = importance_df[importance_df["dataset"] == ds]
        for method in METHOD_ORDER:
            m = sub[sub["method"] == method].sort_values("rf_rank")
            if m.empty:
                continue
            ax.plot(
                m["rf_rank"],
                m["cumulative_rf_importance"],
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=2.0,
                markersize=3.8,
                label=method,
            )
        ax.axhline(0.5, color="#A0A0A0", linestyle="--", linewidth=1.0)
        ax.axhline(0.8, color="#A0A0A0", linestyle=":", linewidth=1.0)
        ax.set_title(ds, color=DATASET_COLORS[ds], weight="semibold", fontsize=12.0, pad=8)
        ax.set_xlabel("RF feature rank")
        ax.set_ylim(-0.03, 1.03)
        clean_axis(ax, grid_axis="y")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
        ax.tick_params(labelsize=8.8, width=1.2, length=4)
        ax.text(
            -0.13,
            1.08,
            panel,
            transform=ax.transAxes,
            fontsize=15,
            weight="bold",
            va="top",
            ha="left",
        )
    axes[0].set_ylabel("Cumulative RF-origin importance")
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], marker="o", linewidth=2.0, markersize=4.8, label=method)
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles,
        METHOD_ORDER,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(METHOD_ORDER),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0,
        borderpad=0.55,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.26, wspace=0.18)
    return fig


def _incident_edge_categories(real_edges, synthetic_edges):
    return {
        "preserved": real_edges & synthetic_edges,
        "real_only": real_edges - synthetic_edges,
        "synthetic_only": synthetic_edges - real_edges,
    }


def _draw_rf_importance_tsne_panel(
    ax,
    coords,
    real_partial,
    real_edges,
    synthetic_partial,
    synthetic_edges,
    feature_names,
    importance,
    method,
    top_features=5,
    max_edges_per_feature=4,
):
    importance = np.asarray(importance, dtype=float)
    max_importance = float(np.max(importance)) if np.max(importance) > 0 else 1.0
    top = np.argsort(importance)[::-1][: int(top_features)]
    top_lookup = {int(feature): rank + 1 for rank, feature in enumerate(top)}
    categories = _incident_edge_categories(real_edges, synthetic_edges)
    edge_partials = {
        "preserved": real_partial,
        "real_only": real_partial,
        "synthetic_only": synthetic_partial,
    }

    drawn_edges = set()
    for feature in top:
        candidates = []
        for category, edges in categories.items():
            partial = edge_partials[category]
            for edge in edges:
                if int(feature) not in edge:
                    continue
                weight = abs(float(partial[edge[0], edge[1]]))
                candidates.append((weight, category, edge))
        candidates = sorted(candidates, reverse=True)[: int(max_edges_per_feature)]
        for weight, category, edge in candidates:
            edge = tuple(edge)
            if (category, edge) in drawn_edges:
                continue
            drawn_edges.add((category, edge))
            i, j = edge
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color=EDGE_COLORS[category],
                linewidth=1.15 + 3.0 * min(weight, 0.8),
                alpha=0.72 if category == "preserved" else 0.82,
                zorder=1,
            )

    sizes = 26 + 520 * np.sqrt(np.clip(importance / max_importance, 0, 1))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=sizes,
        facecolor="#F8F8F8",
        edgecolor="#6A6A6A",
        linewidth=0.55,
        alpha=0.95,
        zorder=2,
    )
    ax.scatter(
        coords[top, 0],
        coords[top, 1],
        s=sizes[top] + 45,
        facecolor=METHOD_PASTELS[method],
        edgecolor=METHOD_COLORS[method],
        linewidth=1.55,
        alpha=0.98,
        zorder=4,
    )
    for feature in top:
        ax.text(
            coords[feature, 0],
            coords[feature, 1],
            f"{top_lookup[int(feature)]}",
            ha="center",
            va="center",
            fontsize=8.4,
            weight="bold",
            color="#111111",
            zorder=5,
        )
        ax.annotate(
            _short_label(feature_names[feature], 16),
            xy=(coords[feature, 0], coords[feature, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=6.8,
            color=METHOD_COLORS[method],
            weight="bold",
            zorder=6,
        )

    ax.set_title(method, color=METHOD_COLORS[method], weight="semibold", fontsize=11.2, pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_edgecolor("#333333")


def plot_rf_importance_tsne_edge_overlay(
    datasets,
    dataset="HIV",
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
    top_features=5,
    max_edges_per_feature=4,
    edge_threshold=1e-7,
):
    """Show RF-important features on the real Graphical-Lasso t-SNE map."""
    data = datasets[dataset]
    X_real = np.asarray(data["X"], dtype=np.float32)
    y_real = np.asarray(data["y"], dtype=int)
    feature_names = list(data.get("feature_names", [f"feature_{i + 1}" for i in range(X_real.shape[1])]))
    alpha = FIGURE4_ALPHAS[dataset]

    theta_real = fit_glasso_precision(X_real, alpha)
    real_partial = precision_to_partial_corr(theta_real)
    real_edges = get_edge_set(real_partial, edge_threshold)
    coords, _, perplexity = _fit_profile_tsne(real_partial, seed=seed)

    fig = plt.figure(figsize=(11.8, 9.7), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.12)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    rows = []

    for ax, method, panel in zip(axes, METHOD_ORDER, ["A", "B", "C", "D"]):
        X_syn, _ = sample_synthetic(dataset, data, method, seed=seed, cvae_epochs=cvae_epochs)
        X_syn = np.asarray(X_syn, dtype=np.float32)
        importance, ranks, ranking = discriminator_feature_importance(X_real, X_syn, seed=seed)
        theta_syn = fit_glasso_precision(X_syn, alpha)
        synthetic_partial = precision_to_partial_corr(theta_syn)
        synthetic_edges = get_edge_set(synthetic_partial, edge_threshold)
        _draw_rf_importance_tsne_panel(
            ax,
            coords,
            real_partial,
            real_edges,
            synthetic_partial,
            synthetic_edges,
            feature_names,
            importance,
            method,
            top_features=top_features,
            max_edges_per_feature=max_edges_per_feature,
        )
        ax.text(
            -0.08,
            1.06,
            panel,
            transform=ax.transAxes,
            fontsize=15,
            weight="bold",
            va="top",
            ha="left",
        )
        for rank_position, feature in enumerate(ranking[: int(top_features)], start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "rf_rank": rank_position,
                    "feature_index": int(feature),
                    "feature_name": feature_names[int(feature)],
                    "rf_importance": float(importance[int(feature)]),
                    "real_degree": int(sum(int(feature) in edge for edge in real_edges)),
                    "preserved_incident_edges": int(sum(int(feature) in edge for edge in (real_edges & synthetic_edges))),
                    "lost_incident_edges": int(sum(int(feature) in edge for edge in (real_edges - synthetic_edges))),
                    "synthetic_only_incident_edges": int(sum(int(feature) in edge for edge in (synthetic_edges - real_edges))),
                }
            )

    handles = [
        Line2D([0], [0], color=EDGE_COLORS["preserved"], linewidth=3.0, label="Preserved edge"),
        Line2D([0], [0], color=EDGE_COLORS["real_only"], linewidth=3.0, label="Real-only / lost"),
        Line2D([0], [0], color=EDGE_COLORS["synthetic_only"], linewidth=3.0, label="Synthetic-only"),
        Line2D([0], [0], marker="o", color="#6A6A6A", markerfacecolor="#F8F8F8", linewidth=0, markersize=7, label="Feature"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0,
        borderpad=0.55,
        fontsize=8.7,
    )
    fig.suptitle(
        f"{dataset}: RF-important features on Graphical Lasso t-SNE (perplexity={perplexity:.0f})",
        y=0.985,
        fontsize=13.6,
        weight="semibold",
    )
    fig.subplots_adjust(left=0.045, right=0.99, top=0.93, bottom=0.095)
    return fig, pd.DataFrame(rows)


def _edge_jaccard_mismatch(real_edges, synthetic_edges):
    union = real_edges | synthetic_edges
    if not union:
        return np.nan
    return float(1.0 - len(real_edges & synthetic_edges) / len(union))


def _structural_metrics_for_subset(X_real, X_syn, alpha, edge_threshold=1e-7):
    theta_real = fit_glasso_precision(X_real, alpha)
    theta_syn = fit_glasso_precision(X_syn, alpha)
    real_partial = precision_to_partial_corr(theta_real)
    synthetic_partial = precision_to_partial_corr(theta_syn)
    real_edges = get_edge_set(real_partial, edge_threshold)
    synthetic_edges = get_edge_set(synthetic_partial, edge_threshold)
    lost_edges = real_edges - synthetic_edges
    new_edges = synthetic_edges - real_edges
    edge_union = real_edges | synthetic_edges
    n_lost_plus_new = len(lost_edges) + len(new_edges)
    return {
        "frobenius_deviation": compute_frobenius_deviation(theta_real, theta_syn),
        "edge_recovery": compute_edge_recovery(real_edges, synthetic_edges),
        "synthetic_only_rate": compute_synthetic_only_rate(real_edges, synthetic_edges),
        "edge_mismatch": _edge_jaccard_mismatch(real_edges, synthetic_edges),
        "lost_plus_new_edge_fraction": n_lost_plus_new / len(edge_union) if edge_union else 0.0,
        "n_lost_plus_new_edges": n_lost_plus_new,
        "n_real_edges": len(real_edges),
        "n_synthetic_edges": len(synthetic_edges),
        "n_preserved_edges": len(real_edges & synthetic_edges),
        "n_real_only_edges": len(lost_edges),
        "n_synthetic_only_edges": len(new_edges),
    }


def structural_refit_removal_grid(n_features, fractions=(0.0, 0.25, 0.50, 0.75)):
    """Sparse removal grid for re-fitting structural diagnostics."""
    max_removed = max(0, int(n_features) - 2)
    points = [int(round(float(frac) * max_removed)) for frac in fractions]
    points.append(max_removed)
    return np.asarray(sorted(set(max(0, min(max_removed, point)) for point in points)), dtype=int)


def compute_refit_structural_ablation(
    datasets,
    seed=SEED,
    repeats=ABLATION_REPEATS,
    cvae_epochs=CVAE_EPOCHS,
    edge_threshold=1e-7,
    removal_fractions=(0.0, 0.25, 0.50, 0.75),
):
    """Recalculate Graphical Lasso edge mismatch after RF-ranked feature removal."""
    rows = []
    for ds in DATASET_ORDER:
        data = datasets[ds]
        X_real = np.asarray(data["X"], dtype=np.float32)
        y_real = np.asarray(data["y"], dtype=int)
        alpha = FIGURE4_ALPHAS[ds]
        grid = structural_refit_removal_grid(
            X_real.shape[1],
            fractions=removal_fractions,
        )
        for method in METHOD_ORDER:
            print(f"[refit structural ablation] {ds} - {method}")
            X_syn, y_syn = sample_synthetic(ds, data, method, seed=seed, cvae_epochs=cvae_epochs)
            X_syn = np.asarray(X_syn, dtype=np.float32)
            ranking = rank_discriminating_features(X_real, X_syn, seed=seed)
            for n_removed in grid:
                n_removed = int(n_removed)
                keep = ranking[n_removed:]
                Xr_keep = X_real[:, keep]
                Xs_keep = X_syn[:, keep]
                vals = [
                    one_run_origin_auc(
                        Xr_keep,
                        y_real,
                        Xs_keep,
                        y_syn,
                        seed=seed + 1009 * r + 13 * METHOD_ORDER.index(method),
                    )
                    for r in range(repeats)
                ]
                structural = _structural_metrics_for_subset(
                    Xr_keep,
                    Xs_keep,
                    alpha=alpha,
                    edge_threshold=edge_threshold,
                )
                rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "n_features_removed": n_removed,
                        "n_features_retained": int(len(keep)),
                        "auc_mean": float(np.mean(vals)),
                        "auc_sd": float(np.std(vals)),
                        "auc_values": vals,
                        **structural,
                    }
                )
    return pd.DataFrame(rows)


def get_refit_structural_ablation(force=False):
    cached = None if force else _read_cache("refit_lost_new_edges_quintile")
    if cached is not None:
        return cached
    return _write_cache(
        "refit_lost_new_edges_quintile",
        compute_refit_structural_ablation(require_datasets()),
    )


def feature_lost_new_edge_scores(X_real, X_syn, alpha, edge_threshold=1e-7):
    """Score features by incident lost and synthetic-only Graphical Lasso edges."""
    real_partial = precision_to_partial_corr(fit_glasso_precision(X_real, alpha))
    synthetic_partial = precision_to_partial_corr(fit_glasso_precision(X_syn, alpha))
    real_edges = get_edge_set(real_partial, edge_threshold)
    synthetic_edges = get_edge_set(synthetic_partial, edge_threshold)
    real_only = real_edges - synthetic_edges
    synthetic_only = synthetic_edges - real_edges
    preserved = real_edges & synthetic_edges
    n_features = real_partial.shape[0]

    rows = []
    for feature in range(n_features):
        incident_real_only = [edge for edge in real_only if feature in edge]
        incident_synthetic_only = [edge for edge in synthetic_only if feature in edge]
        incident_preserved = [edge for edge in preserved if feature in edge]
        incident_union = [
            edge for edge in (real_edges | synthetic_edges) if feature in edge
        ]
        changed_count = len(incident_real_only) + len(incident_synthetic_only)
        union_count = len(incident_union)
        rows.append(
            {
                "feature_index": feature,
                "real_only_edges": len(incident_real_only),
                "synthetic_only_edges": len(incident_synthetic_only),
                "preserved_edges": len(incident_preserved),
                "edge_union": union_count,
                "lost_new_edge_count": changed_count,
                "lost_new_edge_rate": (
                    changed_count / union_count if union_count else 0.0
                ),
                "real_only_weight": sum(
                    abs(float(real_partial[edge[0], edge[1]]))
                    for edge in incident_real_only
                ),
                "synthetic_only_weight": sum(
                    abs(float(synthetic_partial[edge[0], edge[1]]))
                    for edge in incident_synthetic_only
                ),
            }
        )
    scores = pd.DataFrame(rows)
    scores["lost_new_edge_weight"] = (
        scores["real_only_weight"] + scores["synthetic_only_weight"]
    )
    return scores.sort_values(
        ["lost_new_edge_count", "lost_new_edge_weight", "lost_new_edge_rate"],
        ascending=[False, False, False],
        ignore_index=True,
    )


def compute_rf_structure_rank_overlap(
    datasets,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
    edge_threshold=1e-7,
):
    """Compare RF discriminator ranking with lost/new-edge ranking."""
    rows = []
    for ds in DATASET_ORDER:
        data = datasets[ds]
        X_real = np.asarray(data["X"], dtype=np.float32)
        y_real = np.asarray(data["y"], dtype=int)
        feature_names = list(data.get("feature_names", [f"feature_{i + 1}" for i in range(X_real.shape[1])]))
        alpha = FIGURE4_ALPHAS[ds]
        grid = ablation_grid(X_real.shape[1])
        for method in METHOD_ORDER:
            print(f"[rank overlap] {ds} - {method}")
            X_syn, _ = sample_synthetic(ds, data, method, seed=seed, cvae_epochs=cvae_epochs)
            X_syn = np.asarray(X_syn, dtype=np.float32)
            rf_rank = rank_discriminating_features(X_real, X_syn, seed=seed)
            structure_scores = feature_lost_new_edge_scores(
                X_real,
                X_syn,
                alpha=alpha,
                edge_threshold=edge_threshold,
            )
            structure_rank = structure_scores["feature_index"].to_numpy(dtype=int)
            structure_lookup = structure_scores.set_index("feature_index")

            for n_removed in grid:
                n_removed = int(n_removed)
                if n_removed <= 0:
                    overlap = np.nan
                    mean_lost_new_count = 0.0
                    mean_lost_new_weight = 0.0
                else:
                    rf_top = set(rf_rank[:n_removed])
                    structure_top = set(structure_rank[:n_removed])
                    overlap = len(rf_top & structure_top) / n_removed
                    removed = list(rf_rank[:n_removed])
                    mean_lost_new_count = float(
                        structure_lookup.loc[removed, "lost_new_edge_count"].mean()
                    )
                    mean_lost_new_weight = float(
                        structure_lookup.loc[removed, "lost_new_edge_weight"].mean()
                    )
                rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "n_features_removed": n_removed,
                        "rf_structural_topk_overlap": overlap,
                        "mean_removed_lost_new_edge_count": mean_lost_new_count,
                        "mean_removed_lost_new_edge_weight": mean_lost_new_weight,
                    }
                )

            for rank_position, feature in enumerate(rf_rank, start=1):
                row = structure_lookup.loc[int(feature)]
                rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "n_features_removed": -rank_position,
                        "rf_structural_topk_overlap": np.nan,
                        "mean_removed_lost_new_edge_count": np.nan,
                        "mean_removed_lost_new_edge_weight": np.nan,
                        "feature_index": int(feature),
                        "feature_name": feature_names[int(feature)],
                        "rf_rank": rank_position,
                        "lost_new_edge_count": float(row["lost_new_edge_count"]),
                        "lost_new_edge_rate": float(row["lost_new_edge_rate"]),
                        "lost_new_edge_weight": float(row["lost_new_edge_weight"]),
                    }
                )
    return pd.DataFrame(rows)


def get_rf_structure_rank_overlap(force=False):
    cached = None if force else _read_cache("rf_structure_rank_overlap")
    if cached is not None:
        return cached
    return _write_cache(
        "rf_structure_rank_overlap",
        compute_rf_structure_rank_overlap(require_datasets()),
    )


def plot_rf_structure_rank_overlap(overlap_df):
    """Show whether RF-removed features are also high lost/new-edge features."""
    curve_df = overlap_df[overlap_df["n_features_removed"] >= 0].copy()
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(13.4, 3.55), sharey=True)
    for ax, ds, panel in zip(np.ravel(axes), DATASET_ORDER, ["A", "B", "C"]):
        sub = curve_df[curve_df["dataset"] == ds]
        for method in METHOD_ORDER:
            m = sub[sub["method"] == method].sort_values("n_features_removed")
            m = m[m["n_features_removed"] > 0]
            if m.empty:
                continue
            ax.plot(
                m["n_features_removed"],
                m["rf_structural_topk_overlap"],
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=2.0,
                markersize=4.8,
                label=method,
            )
        ax.set_title(ds, color=DATASET_COLORS[ds], weight="semibold", fontsize=12.0, pad=8)
        ax.set_xlabel("Top RF-ranked features removed")
        ax.set_ylim(-0.03, 1.03)
        clean_axis(ax, grid_axis="y")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
        ax.tick_params(labelsize=8.8, width=1.2, length=4)
        ax.text(
            -0.13,
            1.08,
            panel,
            transform=ax.transAxes,
            fontsize=15,
            weight="bold",
            va="top",
            ha="left",
        )
    axes[0].set_ylabel("Overlap with top lost/new-edge features")
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], marker="o", linewidth=2.0, markersize=4.8, label=method)
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles,
        METHOD_ORDER,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(METHOD_ORDER),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0,
        borderpad=0.55,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.26, wspace=0.18)
    return fig


def plot_refit_structural_ablation(refit_df):
    """Plot AUC and recalculated lost/new edges after RF-ranked feature removal."""
    refit_df = refit_df.copy()
    if "lost_plus_new_edge_fraction" not in refit_df.columns:
        refit_df["lost_plus_new_edge_fraction"] = refit_df["edge_mismatch"]
    fig = plt.figure(figsize=(13.4, 7.2), constrained_layout=False)
    gs = fig.add_gridspec(2, len(DATASET_ORDER), height_ratios=[1.0, 1.0])
    auc_axes = [fig.add_subplot(gs[0, i]) for i in range(len(DATASET_ORDER))]
    edge_axes = [fig.add_subplot(gs[1, i], sharex=auc_axes[i]) for i in range(len(DATASET_ORDER))]
    legend_handles = []

    for panel_idx, (ax_auc, ax_edge, ds) in enumerate(zip(auc_axes, edge_axes, DATASET_ORDER)):
        sub = refit_df[refit_df["dataset"] == ds]
        for method in METHOD_ORDER:
            m = sub[sub["method"] == method].sort_values("n_features_removed")
            if m.empty:
                continue
            line, = ax_auc.plot(
                m["n_features_removed"],
                m["auc_mean"],
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=2.15,
                markersize=4.8,
                label=method,
            )
            ax_auc.fill_between(
                m["n_features_removed"],
                m["auc_mean"] - m["auc_sd"],
                m["auc_mean"] + m["auc_sd"],
                color=METHOD_COLORS[method],
                alpha=0.12,
                linewidth=0,
            )
            ax_edge.plot(
                m["n_features_removed"],
                m["lost_plus_new_edge_fraction"],
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=2.15,
                markersize=4.8,
            )
            if panel_idx == 0:
                legend_handles.append(line)

        ax_auc.axhline(0.5, color="#777777", linestyle="--", linewidth=1.15)
        ax_auc.set_title(ds, color=DATASET_COLORS[ds], weight="semibold", fontsize=12.5, pad=8)
        ax_auc.text(
            -0.13,
            1.08,
            f"{chr(ord('A') + panel_idx)}1",
            transform=ax_auc.transAxes,
            fontsize=15,
            weight="bold",
            va="top",
            ha="left",
        )
        ax_edge.text(
            -0.13,
            1.08,
            f"{chr(ord('A') + panel_idx)}2",
            transform=ax_edge.transAxes,
            fontsize=15,
            weight="bold",
            va="top",
            ha="left",
        )
        ax_edge.set_xlabel("Number of RF-ranked features removed")
        for ax in (ax_auc, ax_edge):
            clean_axis(ax, grid_axis="y")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
            ax.tick_params(labelsize=8.8, width=1.2, length=4)

    auc_axes[0].set_ylabel(r"$\langle \mathrm{AUC} \rangle$")
    edge_axes[0].set_ylabel("Lost + new edge fraction")
    for ax in auc_axes[1:] + edge_axes[1:]:
        ax.set_ylabel("")

    y_values = []
    for _, row in refit_df.iterrows():
        y_values.extend([row["auc_mean"] - row["auc_sd"], row["auc_mean"] + row["auc_sd"]])
        y_values.extend(row.get("auc_values", []))
    y_min = max(0.45, np.nanmin(y_values) - 0.03)
    y_max = min(1.03, np.nanmax(y_values) + 0.03)
    for ax in auc_axes:
        ax.set_ylim(y_min, y_max)
    for ax in edge_axes:
        ax.set_ylim(-0.03, 1.03)

    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(legend_handles),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0,
        borderpad=0.55,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.15, wspace=0.18, hspace=0.36)
    return fig


def _snapshot_removal_grid(n_features, fractions=(0.0, 0.20, 0.50)):
    max_removed = max(0, int(n_features) - 2)
    points = [int(round(float(frac) * max_removed)) for frac in fractions]
    points.append(max_removed)
    return np.asarray(sorted(set(max(0, min(max_removed, point)) for point in points)), dtype=int)


def plot_hiv_cvae_refit_edge_status_snapshots(
    datasets,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
    edge_threshold=1e-7,
):
    """2x2 HIV/CVAE edge-status snapshots after RF-ranked feature removal."""
    dataset = "HIV"
    method = "CVAE"
    data = datasets[dataset]
    X_real = np.asarray(data["X"], dtype=np.float32)
    y_real = np.asarray(data["y"], dtype=int)
    feature_names = list(data.get("feature_names", [f"feature_{i + 1}" for i in range(X_real.shape[1])]))
    alpha = FIGURE4_ALPHAS[dataset]

    X_syn, _ = sample_synthetic(dataset, data, method, seed=seed, cvae_epochs=cvae_epochs)
    X_syn = np.asarray(X_syn, dtype=np.float32)
    _, _, ranking = discriminator_feature_importance(X_real, X_syn, seed=seed)
    grid = _snapshot_removal_grid(X_real.shape[1])

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.4), constrained_layout=False)
    axes = list(np.ravel(axes))
    categories = ["absent", "preserved", "real_only", "synthetic_only"]
    cmap = mpl.colors.ListedColormap([STATUS_COLORS[category] for category in categories])
    rows = []

    for ax, n_removed, panel in zip(axes, grid, ["A", "B", "C", "D"]):
        keep = np.asarray(ranking[int(n_removed):], dtype=int)
        Xr_keep = X_real[:, keep]
        Xs_keep = X_syn[:, keep]
        theta_real = fit_glasso_precision(Xr_keep, alpha)
        theta_syn = fit_glasso_precision(Xs_keep, alpha)
        real_partial = precision_to_partial_corr(theta_real)
        synthetic_partial = precision_to_partial_corr(theta_syn)
        real_edges = get_edge_set(real_partial, edge_threshold)
        synthetic_edges = get_edge_set(synthetic_partial, edge_threshold)
        order = get_real_structure_order(real_partial)
        status = build_edge_status_matrix(real_edges, synthetic_edges, len(keep))[np.ix_(order, order)]
        ordered_original_indices = keep[order] + 1

        ax.imshow(
            status,
            cmap=cmap,
            vmin=-0.5,
            vmax=3.5,
            interpolation="nearest",
            aspect="equal",
        )
        n_remaining = len(keep)
        tick_step = 1 if n_remaining <= 12 else 5 if n_remaining <= 35 else 10
        ticks = np.arange(0, n_remaining, tick_step)
        labels = [str(int(ordered_original_indices[tick])) for tick in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, fontsize=7.4)
        ax.set_yticklabels(labels, fontsize=7.4)
        ax.tick_params(axis="both", length=2.2, width=0.8, pad=1.5)
        top_ax = ax.secondary_xaxis("top")
        top_ax.set_xticks(ticks)
        top_ax.set_xticklabels(labels, fontsize=7.4)
        top_ax.tick_params(length=2.2, width=0.8, pad=1.5)
        ax.text(
            0.03,
            0.06,
            f"{panel}. remove {int(n_removed)}\nkeep {n_remaining}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.6,
            weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.5),
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.1)

        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_features_removed": int(n_removed),
                "n_features_retained": int(n_remaining),
                "edge_recovery": compute_edge_recovery(real_edges, synthetic_edges),
                "synthetic_only_rate": compute_synthetic_only_rate(real_edges, synthetic_edges),
                "edge_mismatch": _edge_jaccard_mismatch(real_edges, synthetic_edges),
                "n_real_edges": len(real_edges),
                "n_synthetic_edges": len(synthetic_edges),
                "n_preserved_edges": len(real_edges & synthetic_edges),
                "n_real_only_edges": len(real_edges - synthetic_edges),
                "n_synthetic_only_edges": len(synthetic_edges - real_edges),
                "retained_original_feature_indices": ordered_original_indices.astype(int).tolist(),
            }
        )

    handles = [
        Patch(facecolor=STATUS_COLORS["preserved"], edgecolor="#333333", label="Preserved edge"),
        Patch(facecolor=STATUS_COLORS["real_only"], edgecolor="#333333", label="Real-only / lost"),
        Patch(facecolor=STATUS_COLORS["synthetic_only"], edgecolor="#333333", label="Synthetic-only"),
        Patch(facecolor=STATUS_COLORS["absent"], edgecolor="#C9CDD2", label="Absent in both"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0,
        borderpad=0.55,
        fontsize=8.4,
    )
    fig.suptitle(
        "HIV CVAE: re-fit edge-status maps after RF-ranked feature removal",
        y=0.985,
        fontsize=13.5,
        weight="semibold",
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.925, bottom=0.10, wspace=0.02, hspace=0.10)
    return fig, pd.DataFrame(rows)


def _rf_removal_sweep_grid(n_features, fractions=(0.0, 0.25, 0.50, 0.75)):
    max_removed = max(0, int(n_features) - 2)
    points = [int(round(float(frac) * max_removed)) for frac in fractions]
    points.append(max_removed)
    return np.asarray(sorted(set(max(0, min(max_removed, point)) for point in points)), dtype=int)


def plot_hiv_refit_edge_status_removal_sweep(
    datasets,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
    removal_fractions=(0.0, 0.25, 0.50, 0.75),
    edge_threshold=1e-7,
):
    """HIV Figure-4-style 63x63 edge-status maps after RF-ranked feature removal."""
    dataset = "HIV"
    data = datasets[dataset]
    X_real = np.asarray(data["X"], dtype=np.float32)
    alpha = FIGURE4_ALPHAS[dataset]
    removal_grid = _rf_removal_sweep_grid(X_real.shape[1], fractions=removal_fractions)
    full_real_partial = precision_to_partial_corr(fit_glasso_precision(X_real, alpha))
    full_order = get_real_structure_order(full_real_partial)

    removed_code = 4
    removed_color = "#B8BDC6"
    categories = ["absent", "preserved", "real_only", "synthetic_only", "removed"]
    color_lookup = {**STATUS_COLORS, "removed": removed_color}
    cmap = mpl.colors.ListedColormap([color_lookup[category] for category in categories])
    fig = plt.figure(figsize=(20.5, 7.4), constrained_layout=False)
    outer = fig.add_gridspec(1, len(removal_grid), wspace=0.055)
    rows = []
    method_data = {}
    for method in METHOD_ORDER:
        X_syn, _ = sample_synthetic(dataset, data, method, seed=seed, cvae_epochs=cvae_epochs)
        X_syn = np.asarray(X_syn, dtype=np.float32)
        _, _, ranking = discriminator_feature_importance(X_real, X_syn, seed=seed)
        method_data[method] = {"X_syn": X_syn, "ranking": np.asarray(ranking, dtype=int)}

    method_panels = {
        "Bootstrap": ("A", 0, 0),
        "Column-wise": ("B", 0, 1),
        "GMM": ("C", 1, 0),
        "CVAE": ("D", 1, 1),
    }
    group_label_specs = []

    for group_idx, n_removed in enumerate(removal_grid):
        sub = outer[group_idx].subgridspec(2, 2, wspace=0.015, hspace=0.015)
        axes_by_method = {}
        for method in METHOD_ORDER:
            panel, row_idx, col_idx = method_panels[method]
            ax = fig.add_subplot(sub[row_idx, col_idx])
            axes_by_method[method] = ax

        for method in METHOD_ORDER:
            ax = axes_by_method[method]
            panel, _, _ = method_panels[method]
            X_syn = method_data[method]["X_syn"]
            ranking = method_data[method]["ranking"]
            keep = np.asarray(ranking[int(n_removed):], dtype=int)

            theta_real = fit_glasso_precision(X_real[:, keep], alpha)
            theta_syn = fit_glasso_precision(X_syn[:, keep], alpha)
            real_partial = precision_to_partial_corr(theta_real)
            synthetic_partial = precision_to_partial_corr(theta_syn)
            real_edges = get_edge_set(real_partial, edge_threshold)
            synthetic_edges = get_edge_set(synthetic_partial, edge_threshold)
            status_keep = build_edge_status_matrix(real_edges, synthetic_edges, len(keep))
            status = np.full((X_real.shape[1], X_real.shape[1]), removed_code, dtype=int)
            status[np.ix_(keep, keep)] = status_keep
            status = status[np.ix_(full_order, full_order)]
            ordered_original_indices = keep + 1

            ax.imshow(status, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(length=0)
            ax.text(
                0.035,
                0.055,
                f"{panel}. {method}\nkeep {len(keep)}/63",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8.6,
                weight="bold",
                color=METHOD_COLORS.get(method, "#222222"),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.9),
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.95)

            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n_features_removed": int(n_removed),
                    "removal_label": "all but 2"
                    if int(n_removed) == int(X_real.shape[1] - 2)
                    else f"{int(round(100 * int(n_removed) / max(1, X_real.shape[1] - 2)))}%",
                    "n_features_retained": int(len(keep)),
                    "edge_recovery": compute_edge_recovery(real_edges, synthetic_edges),
                    "synthetic_only_rate": compute_synthetic_only_rate(real_edges, synthetic_edges),
                    "edge_mismatch": _edge_jaccard_mismatch(real_edges, synthetic_edges),
                    "n_real_edges": len(real_edges),
                    "n_synthetic_edges": len(synthetic_edges),
                    "n_preserved_edges": len(real_edges & synthetic_edges),
                    "n_real_only_edges": len(real_edges - synthetic_edges),
                    "n_synthetic_only_edges": len(synthetic_edges - real_edges),
                    "retained_original_feature_indices": ordered_original_indices.astype(int).tolist(),
                }
            )

        label = "all but 2" if int(n_removed) == int(X_real.shape[1] - 2) else f"{int(round(100 * n_removed / max(1, X_real.shape[1] - 2)))}%"
        group_label_specs.append((axes_by_method[METHOD_ORDER[0]], axes_by_method[METHOD_ORDER[-1]], f"Remove {int(n_removed)} ({label})"))

    handles = [
        Patch(facecolor=STATUS_COLORS["preserved"], edgecolor="#333333", label="Preserved edge"),
        Patch(facecolor=STATUS_COLORS["real_only"], edgecolor="#333333", label="Real-only / lost"),
        Patch(facecolor=STATUS_COLORS["synthetic_only"], edgecolor="#333333", label="Synthetic-only"),
        Patch(facecolor=STATUS_COLORS["absent"], edgecolor="#C9CDD2", label="Absent in both"),
        Patch(facecolor=removed_color, edgecolor="#777777", label="Removed feature"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        fontsize=8.8,
        columnspacing=1.05,
        handlelength=1.45,
        handletextpad=0.45,
    )
    fig.suptitle(
        "HIV: fixed 63x63 edge-status maps after RF-ranked feature removal",
        y=0.99,
        fontsize=14.0,
        weight="semibold",
    )
    fig.subplots_adjust(left=0.012, right=0.992, top=0.875, bottom=0.095)
    for first_ax, last_ax, label in group_label_specs:
        first_pos = first_ax.get_position()
        last_pos = last_ax.get_position()
        fig.text(
            (first_pos.x0 + last_pos.x1) / 2,
            0.905,
            label,
            ha="center",
            va="bottom",
            fontsize=10.2,
            weight="bold",
        )
    return fig, pd.DataFrame(rows)


def plot_figure6_ablation_ac(ablation_df):
    """Compact Figure 6 panel: A-C show reverse ablation curves."""
    fig, curve_axes = plt.subplots(1, len(DATASET_ORDER), figsize=(13.4, 3.8), sharey=True)
    curve_axes = list(np.ravel(curve_axes))

    legend_handles = []
    for panel_idx, (ax_curve, ds) in enumerate(zip(curve_axes, DATASET_ORDER)):
        sub = ablation_df[ablation_df["dataset"] == ds]
        if sub.empty:
            ax_curve.set_visible(False)
            continue

        for method in METHOD_ORDER:
            m = sub[sub["method"] == method].sort_values("n_features_removed")
            if m.empty:
                continue
            n_removed = m["n_features_removed"]
            line, = ax_curve.plot(n_removed, m["auc_mean"], color=METHOD_COLORS[method], marker="o",
                                  linewidth=2.35, markersize=5.2, label=method)
            ax_curve.fill_between(n_removed, m["auc_mean"] - m["auc_sd"], m["auc_mean"] + m["auc_sd"],
                                  color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
            if panel_idx == 0:
                legend_handles.append(line)

        ax_curve.axhline(0.5, color="#777777", linestyle="--", linewidth=1.25)
        ax_curve.set_title("")
        max_removed = int(sub["n_features_removed"].max())
        ax_curve.set_xlim(-1, max_removed + 1)
        ax_curve.set_xlabel("Number of features removed", labelpad=6)
        ax_curve.text(
            0.5,
            -0.34,
            ds,
            transform=ax_curve.transAxes,
            ha="center",
            va="top",
            color=DATASET_COLORS[ds],
            weight="semibold",
            fontsize=13,
        )
        clean_axis(ax_curve, grid_axis="y")
        ax_curve.spines["left"].set_linewidth(1.8)
        ax_curve.spines["bottom"].set_linewidth(1.8)
        ax_curve.tick_params(width=1.4, length=5)

    curve_axes[0].set_ylabel(r"$\langle \mathrm{AUC} \rangle$")
    for ax in curve_axes[1:]:
        ax.tick_params(labelleft=False)

    y_values = []
    for _, row in ablation_df.iterrows():
        y_values.extend([row["auc_mean"] - row["auc_sd"], row["auc_mean"] + row["auc_sd"]])
        y_values.extend(row.get("auc_values", []))
    y_min = max(0.45, np.nanmin(y_values) - 0.03)
    y_max = min(1.02, np.nanmax(y_values) + 0.03)
    for ax in curve_axes:
        if ax.get_visible():
            ax.set_ylim(y_min, y_max)

    handles = legend_handles[:len(METHOD_ORDER)]
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=len(handles), frameon=False,
               borderpad=0.25, columnspacing=1.05, handlelength=1.55, handletextpad=0.45)
    fig.suptitle("Reverse feature ablation", y=0.98, fontsize=15, weight="semibold")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.31, wspace=0.18)
    return fig

def _flat_values(series):
    vals = []
    for item in series:
        vals.extend(item if isinstance(item, (list, tuple, np.ndarray)) else [item])
    return np.asarray(vals, dtype=float)

def plot_figure6_ablation_all_datasets(ablation_df):
    fig, curve_axes = plt.subplots(1, len(DATASET_ORDER), figsize=(13.4, 3.85), sharey=True)
    curve_axes = list(np.ravel(curve_axes))

    legend_handles = []
    for panel_idx, (ax_curve, ds) in enumerate(zip(curve_axes, DATASET_ORDER)):
        sub = ablation_df[ablation_df["dataset"] == ds]
        if sub.empty:
            ax_curve.set_visible(False)
            continue

        for method in METHOD_ORDER:
            m = sub[sub["method"] == method].sort_values("n_features_removed")
            if m.empty:
                continue
            n_removed = m["n_features_removed"]
            line, = ax_curve.plot(n_removed, m["auc_mean"], color=METHOD_COLORS[method], marker="o",  linewidth=2.35, markersize=5.2, label=method)
            ax_curve.fill_between(n_removed, m["auc_mean"] - m["auc_sd"], m["auc_mean"] + m["auc_sd"], color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
            if panel_idx == 0:
                legend_handles.append(line)

        ax_curve.axhline(0.5, color="#777777", linestyle="--", linewidth=1.25)
        ax_curve.set_title("")

        max_removed = int(sub["n_features_removed"].max())
        min_removed = int(sub["n_features_removed"].min())

        x_range = max_removed - min_removed
        x_pad = max(0.35, 0.06 * x_range)

        ax_curve.set_xlim(min_removed - x_pad, max_removed + x_pad)

        ax_curve.set_xlabel("Number of features removed", labelpad=6)
        ax_curve.text(
            0.5,
            -0.34,
            ds,
            transform=ax_curve.transAxes,
            ha="center",
            va="top",
            color=DATASET_COLORS[ds],
            weight="semibold",
            fontsize=13,
        )
        clean_axis(ax_curve, grid_axis="y")

        for spine in ax_curve.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)

        ax_curve.tick_params(labelsize=9.0, width=1.2, length=4)

    curve_axes[0].set_ylabel(r"$\langle \mathrm{AUC} \rangle$")
    for ax in curve_axes[1:]:
        ax.set_ylabel("") 
        ax.tick_params(labelleft=True) 

    y_values = []
    for _, row in ablation_df.iterrows():
        y_values.extend([row["auc_mean"] - row["auc_sd"], row["auc_mean"] + row["auc_sd"]])
        y_values.extend(row.get("auc_values", []))
    y_min = max(0.45, np.nanmin(y_values) - 0.03)
    y_max = min(1.02, np.nanmax(y_values) + 0.03)
    for ax in curve_axes:
        if ax.get_visible():
            ax.set_ylim(y_min, y_max)

    handles = legend_handles[:len(METHOD_ORDER)]
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=len(handles), frameon=False,
               borderpad=0.25, columnspacing=1.05, handlelength=1.55, handletextpad=0.45)
    # fig.suptitle("Reverse feature ablation", y=0.98, fontsize=15, weight="semibold")

    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.31, wspace=0.18)
    return fig
