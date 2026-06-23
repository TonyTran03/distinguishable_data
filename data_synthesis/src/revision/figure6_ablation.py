"""Figure 6: reverse-ablation and feature-removal sensitivity."""

from matplotlib.lines import Line2D

from src.revision.common import *
from src.revision.stats import ablation_grid, one_run_origin_auc, rank_discriminating_features
from src.revision.cache import _read_cache, _write_cache
from src.revision.figure4_graphical_lasso import FIGURE4_ALPHAS
from src.revision.figure4_graphical_lasso_plots import (
    EDGE_COLORS,
    compute_edge_recovery,
    compute_frobenius_deviation,
    compute_synthetic_only_rate,
    fit_glasso_precision,
    get_edge_set,
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


def _edge_jaccard_discordance(real_edges, synthetic_edges):
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
    return {
        "frobenius_deviation": compute_frobenius_deviation(theta_real, theta_syn),
        "edge_recovery": compute_edge_recovery(real_edges, synthetic_edges),
        "synthetic_only_rate": compute_synthetic_only_rate(real_edges, synthetic_edges),
        "edge_discordance": _edge_jaccard_discordance(real_edges, synthetic_edges),
        "n_real_edges": len(real_edges),
        "n_synthetic_edges": len(synthetic_edges),
        "n_preserved_edges": len(real_edges & synthetic_edges),
        "n_real_only_edges": len(real_edges - synthetic_edges),
        "n_synthetic_only_edges": len(synthetic_edges - real_edges),
    }


def structural_refit_removal_grid(n_features, fractions=(0.0, 0.20, 0.50, 0.75)):
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
    removal_fractions=(0.0, 0.20, 0.50, 0.75),
):
    """Re-fit Graphical Lasso after RF-ranked feature removal.

    This asks whether the AUC drop under reverse ablation is accompanied by a
    drop in re-estimated real-vs-synthetic structural discordance.
    """
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
    cached = None if force else _read_cache("refit_structural_ablation_sparse")
    if cached is not None:
        return cached
    return _write_cache(
        "refit_structural_ablation_sparse",
        compute_refit_structural_ablation(require_datasets()),
    )


def feature_edge_discordance_scores(X_real, X_syn, alpha, edge_threshold=1e-7):
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
        discordant_count = len(incident_real_only) + len(incident_synthetic_only)
        union_count = len(incident_union)
        rows.append(
            {
                "feature_index": feature,
                "real_only_edges": len(incident_real_only),
                "synthetic_only_edges": len(incident_synthetic_only),
                "preserved_edges": len(incident_preserved),
                "edge_union": union_count,
                "edge_discordance_count": discordant_count,
                "edge_discordance_rate": (
                    discordant_count / union_count if union_count else 0.0
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
    scores["edge_discordance_weight"] = (
        scores["real_only_weight"] + scores["synthetic_only_weight"]
    )
    return scores.sort_values(
        ["edge_discordance_count", "edge_discordance_weight", "edge_discordance_rate"],
        ascending=[False, False, False],
        ignore_index=True,
    )


def compute_rf_structure_rank_overlap(
    datasets,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
    edge_threshold=1e-7,
):
    """Compare RF discriminator ranking with edge-discordance ranking."""
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
            structure_scores = feature_edge_discordance_scores(
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
                    mean_discordance_count = 0.0
                    mean_discordance_weight = 0.0
                else:
                    rf_top = set(rf_rank[:n_removed])
                    structure_top = set(structure_rank[:n_removed])
                    overlap = len(rf_top & structure_top) / n_removed
                    removed = list(rf_rank[:n_removed])
                    mean_discordance_count = float(
                        structure_lookup.loc[removed, "edge_discordance_count"].mean()
                    )
                    mean_discordance_weight = float(
                        structure_lookup.loc[removed, "edge_discordance_weight"].mean()
                    )
                rows.append(
                    {
                        "dataset": ds,
                        "method": method,
                        "n_features_removed": n_removed,
                        "rf_structural_topk_overlap": overlap,
                        "mean_removed_edge_discordance_count": mean_discordance_count,
                        "mean_removed_edge_discordance_weight": mean_discordance_weight,
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
                        "mean_removed_edge_discordance_count": np.nan,
                        "mean_removed_edge_discordance_weight": np.nan,
                        "feature_index": int(feature),
                        "feature_name": feature_names[int(feature)],
                        "rf_rank": rank_position,
                        "edge_discordance_count": float(row["edge_discordance_count"]),
                        "edge_discordance_rate": float(row["edge_discordance_rate"]),
                        "edge_discordance_weight": float(row["edge_discordance_weight"]),
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
    """Show whether RF-removed features are also structurally discordant."""
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
    axes[0].set_ylabel("Overlap with top structural-discordance features")
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
    """Plot AUC and re-fit edge discordance along RF-ranked feature removal."""
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
                m["edge_discordance"],
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
    edge_axes[0].set_ylabel("Re-fit edge discordance")
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
        ax_curve.set_title(ds, color=DATASET_COLORS[ds], weight="semibold", pad=8, fontsize=13)
        ax_curve.text(-0.13, 1.08, chr(ord("A") + panel_idx), transform=ax_curve.transAxes,
                      fontsize=15, weight="bold", va="top", ha="left")
        max_removed = int(sub["n_features_removed"].max())
        ax_curve.set_xlim(-1, max_removed + 1)
        ax_curve.set_xlabel("Features removed")
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
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=len(handles), frameon=True,
               facecolor="white", edgecolor="black", framealpha=1.0, borderpad=0.55)
    fig.suptitle("Reverse feature ablation", y=0.98, fontsize=15, weight="semibold")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.78, bottom=0.27, wspace=0.18)
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
        ax_curve.set_title(ds, color=DATASET_COLORS[ds], weight="semibold", pad=8, fontsize=13)

        ax_curve.text(
            -0.13, 1.08, chr(ord("A") + panel_idx),
            transform=ax_curve.transAxes,
            fontsize=15,
            weight="bold",
            va="top",
            ha="left"
        )

        max_removed = int(sub["n_features_removed"].max())
        min_removed = int(sub["n_features_removed"].min())

        x_range = max_removed - min_removed
        x_pad = max(0.35, 0.06 * x_range)

        ax_curve.set_xlim(min_removed - x_pad, max_removed + x_pad)

        ax_curve.set_xlabel("Number of features removed")
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
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=len(handles), frameon=True,
               facecolor="white", edgecolor="black", framealpha=0, borderpad=0.55)
    # fig.suptitle("Reverse feature ablation", y=0.98, fontsize=15, weight="semibold")

    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.27, wspace=0.18)
    return fig
