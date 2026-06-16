"""Figure 4: structural preservation, edges, clusters, and t-SNE analyses."""

from src.revision.common import *
from src.revision.stats import (
    corr_matrix,
    correlation_frobenius_discrepancy,
    target_rf_importance,
)
from src.revision.figure4_graphical_lasso_plots import (
    plot_figure4_metric_summary,
    plot_figure4_edge_status_matrices,
    plot_figure4_cluster_summary_grid,
    plot_figure4_tsne_edge_supplement,
    plot_edge_status_examples,
)

def feature_group(name):
    lower = str(name).lower()
    if "igg" in lower and "saliva" in lower:
        return "saliva IgG"
    if "iga" in lower:
        return "saliva IgA"
    if "igg" in lower and ("blood" in lower or "serum" in lower):
        return "serum IgG"
    if "ifng" in lower or "il2" in lower or "dual" in lower:
        return "cytokines"
    if "ace2" in lower or "neut" in lower or "neutral" in lower or "rbd" in lower:
        return "neutralization/ACE2"
    return "other"

FEATURE_GROUP_COLORS = {
    "serum IgG": "#8C564B",
    "saliva IgG": "#D62728",
    "saliva IgA": "#4C78A8",
    "cytokines": "#F2B447",
    "neutralization/ACE2": "#8DBF56",
    "other": "#9E9E9E",
}

PASTEL_CORR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "pastel_corr", ["#8FB7D6", "#F7F7F2", "#D9A067"], N=256
)

ABS_CORR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "pastel_abs_corr", ["#FBFAF7", "#CFE7DF", "#71B6A4", "#2F7F75"], N=256
)

def color_feature_ticklabels(ax, top_features):
    top_features = set(top_features)
    for label in ax.get_yticklabels():
        txt = label.get_text()
        if txt in top_features:
            label.set_color(FEATURE_GROUP_COLORS.get(feature_group(txt), NEUTRAL))
            label.set_fontweight("bold")
            label.set_alpha(1.0)
        else:
            label.set_color("#777777")
            label.set_alpha(0.38)

def short_feature_name(name, max_len=18):
    name = str(name).replace("_", " ")
    return name if len(name) <= max_len else name[:max_len - 1] + "..."

def draw_corr_heatmap(ax, fig, matrix, ordered_features, title, show_y=True, top_features=None):
    im = ax.imshow(matrix, aspect="auto", cmap=PASTEL_CORR_CMAP, vmin=-1, vmax=1)
    ax.set_title(title, loc="left", weight="bold", fontsize=12.8)
    ax.set_xticks([])
    if show_y:
        ax.set_yticks(np.arange(len(ordered_features)))
        ax.set_yticklabels(ordered_features, fontsize=5.8)
        color_feature_ticklabels(ax, top_features or [])
    else:
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("#444444")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.ax.tick_params(labelsize=8.5, width=1.2, length=3)
    cbar.outline.set_linewidth(1.0)
    return im

def draw_abs_corr_heatmap(ax, fig, matrix, ordered_features, title, show_y=False):
    vmax = max(0.15, float(np.nanpercentile(matrix, 98)))
    im = ax.imshow(matrix, aspect="auto", cmap=ABS_CORR_CMAP, vmin=0, vmax=min(vmax, 1.0))
    ax.set_title(title, loc="left", weight="bold", fontsize=12.8)
    ax.set_xticks([])
    if show_y:
        ax.set_yticks(np.arange(len(ordered_features)))
        ax.set_yticklabels(ordered_features, fontsize=5.8)
    else:
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("#444444")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.ax.tick_params(labelsize=8.5, width=1.2, length=3)
    cbar.outline.set_linewidth(1.0)
    cbar.set_label("absolute difference", fontsize=9.0)
    return im

def plot_marginal_violin_strip(ax, X_real, X_syn, feature_names, order_idx, representative_method, top_n=10):
    top_idx = list(order_idx[:min(top_n, len(order_idx))])
    Xr_z, Xs_z = standardize_pair(X_real, X_syn)
    positions = np.arange(len(top_idx), dtype=float)
    real_color = "#D5D7DA"
    syn_color = METHOD_COLORS.get(representative_method, "#009E73")

    for pos, feat_idx in zip(positions, top_idx):
        group_color = FEATURE_GROUP_COLORS.get(feature_group(feature_names[feat_idx]), "#9E9E9E")
        vp_real = ax.violinplot([Xr_z[:, feat_idx]], positions=[pos - 0.17], widths=0.28,
                                showmeans=False, showmedians=True, showextrema=False)
        vp_syn = ax.violinplot([Xs_z[:, feat_idx]], positions=[pos + 0.17], widths=0.28,
                               showmeans=False, showmedians=True, showextrema=False)
        for body in vp_real["bodies"]:
            body.set_facecolor(real_color)
            body.set_edgecolor("#5E5E5E")
            body.set_alpha(0.92)
            body.set_linewidth(1.0)
        for body in vp_syn["bodies"]:
            body.set_facecolor(syn_color)
            body.set_edgecolor(group_color)
            body.set_alpha(0.72)
            body.set_linewidth(1.2)
        for key in ["cmedians"]:
            vp_real[key].set_color("#333333")
            vp_real[key].set_linewidth(1.2)
            vp_syn[key].set_color("#333333")
            vp_syn[key].set_linewidth(1.2)

    labels = [short_feature_name(feature_names[i], 16) for i in top_idx]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8.2)
    for tick, feat_idx in zip(ax.get_xticklabels(), top_idx):
        tick.set_color(FEATURE_GROUP_COLORS.get(feature_group(feature_names[feat_idx]), NEUTRAL))
        tick.set_fontweight("bold")
    ax.axhline(0, color="#777777", linewidth=1.0, alpha=0.55)
    ax.set_ylabel("z-scored feature value")
    ax.set_title("Marginal feature distributions", loc="left", weight="bold")
    clean_axis(ax, grid_axis="y")
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.tick_params(width=1.6, length=5)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=real_color, edgecolor="#5E5E5E", label="Real"),
        Rectangle((0, 0), 1, 1, facecolor=syn_color, edgecolor="#333333", alpha=0.72, label=representative_method),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="white", edgecolor="black",
              framealpha=0.96, fontsize=9.0, borderpad=0.35, handlelength=1.2)

def ordered_real_correlation(dataset, data, seed=SEED):
    X_real = np.asarray(data["X"], dtype=np.float32)
    y_real = np.asarray(data["y"], dtype=int)
    feature_names = list(data.get("feature_names", [f"F{i+1}" for i in range(X_real.shape[1])]))
    importances = target_rf_importance(X_real, y_real, seed=seed)
    order_idx = np.argsort(importances)[::-1]
    ordered_features = [feature_names[i] for i in order_idx]
    Xr_scaled, _ = standardize_pair(X_real, X_real)
    real_corr = corr_matrix(Xr_scaled)[:, order_idx][order_idx, :]
    return real_corr, ordered_features, order_idx

def compute_figure4_frobenius_summary(datasets, seed=SEED, cvae_epochs=CVAE_EPOCHS):
    rows = []
    for ds in DATASET_ORDER:
        data = datasets[ds]
        X_real = np.asarray(data["X"], dtype=np.float32)
        real_corr = corr_matrix(X_real)
        for method in METHOD_ORDER:
            print(f"[figure4] {ds} - {method}")
            X_syn, _ = sample_synthetic(ds, data, method, seed=seed, cvae_epochs=cvae_epochs)
            syn_corr = corr_matrix(X_syn)
            rows.append({
                "dataset": ds,
                "method": method,
                "frobenius_corr_discrepancy": correlation_frobenius_discrepancy(real_corr, syn_corr),
            })
    return pd.DataFrame(rows)

FIGURE4_ALPHAS = {"HIV": 0.504, "Breast Cancer": 0.502, "Diabetes": 0.0159}

def _build_notebook_precision_inputs(seed=SEED, cvae_epochs=CVAE_EPOCHS):
    real_data = {}
    synthetic_data = {}
    feature_name_map = {}
    for ds in DATASET_ORDER:
        data = require_datasets()[ds]
        real_data[ds] = np.asarray(data["X"], dtype=np.float64)
        feature_name_map[ds] = list(data.get("feature_names", [f"f{i}" for i in range(real_data[ds].shape[1])]))
        synthetic_data[ds] = {}
        for method in METHOD_ORDER:
            X_syn, _ = sample_synthetic(ds, data, method, seed=seed, cvae_epochs=cvae_epochs)
            synthetic_data[ds][method] = np.asarray(X_syn, dtype=np.float64)
    return real_data, synthetic_data, feature_name_map

def _get_figure4_precision_inputs(seed=SEED, cvae_epochs=CVAE_EPOCHS):
    global figure4_supp_real_data, figure4_supp_synthetic_data, figure4_supp_feature_names
    if not all(name in globals() for name in [
        "figure4_supp_real_data",
        "figure4_supp_synthetic_data",
        "figure4_supp_feature_names",
    ]):
        figure4_supp_real_data, figure4_supp_synthetic_data, figure4_supp_feature_names = _build_notebook_precision_inputs(
            seed=seed, cvae_epochs=cvae_epochs
        )
    return figure4_supp_real_data, figure4_supp_synthetic_data, figure4_supp_feature_names

def plot_figure4_edge_status(dataset_name="HIV", threshold=1e-7, save_path=None):
    real_data, synthetic_data, feature_name_map = _get_figure4_precision_inputs(
        seed=SEED, cvae_epochs=CVAE_EPOCHS
    )
    result = plot_figure4_edge_status_matrices(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_name_map,
        alphas=FIGURE4_ALPHAS,
        dataset_order=DATASET_ORDER,
        method_order=METHOD_ORDER,
        exemplar_ds=dataset_name,
        threshold=threshold,
        save_path=save_path,
    )
    return result

def plot_figure4_tsne_analysis_supplement(
    dataset_name="HIV",
    threshold=1e-7,
    cluster_metric="euclidean",
    cluster_linkage="average",
    save_path=None,
):
    real_data, synthetic_data, feature_name_map = _get_figure4_precision_inputs(
        seed=SEED, cvae_epochs=CVAE_EPOCHS
    )
    result = plot_figure4_tsne_edge_supplement(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_name_map,
        alphas=FIGURE4_ALPHAS,
        dataset_order=DATASET_ORDER,
        method_order=METHOD_ORDER,
        exemplar_ds=dataset_name,
        threshold=threshold,
        seed=SEED,
        cluster_metric=cluster_metric,
        cluster_linkage=cluster_linkage,
        save_path=save_path,
    )
    return result

def plot_figure4_cluster_summary_all_datasets(
    threshold=1e-7,
    cluster_feature_label_top=0,
    cluster_metric="euclidean",
    cluster_linkage="average",
    cluster_blob_pad=0.52,
    cluster_boundary_style="blob",
    cluster_fill_alpha=0.16,
    save_path=None,
):
    real_data, synthetic_data, feature_name_map = _get_figure4_precision_inputs(
        seed=SEED, cvae_epochs=CVAE_EPOCHS
    )
    result = plot_figure4_cluster_summary_grid(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_name_map,
        alphas=FIGURE4_ALPHAS,
        dataset_order=["HIV", "Diabetes", "Breast Cancer"],
        method_order=METHOD_ORDER,
        threshold=threshold,
        cluster_feature_label_top=cluster_feature_label_top,
        cluster_metric=cluster_metric,
        cluster_linkage=cluster_linkage,
        cluster_blob_pad=cluster_blob_pad,
        cluster_boundary_style=cluster_boundary_style,
        cluster_fill_alpha=cluster_fill_alpha,
        seed=SEED,
        save_path=save_path,
    )
    return result

def build_figure4_supplemental_figures_inline(
    real_data,
    synthetic_data,
    feature_names,
    seed=SEED,
):
    """Build the inline HIV edge-status reading example without writing exports."""
    examples_result = plot_edge_status_examples(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_names,
        alphas=FIGURE4_ALPHAS,
        dataset_order=DATASET_ORDER,
        method_order=METHOD_ORDER,
        exemplar_ds="HIV",
        save_path=None,
    )

    results = {"HIV examples": examples_result}
    metrics = examples_result.metrics.assign(figure_dataset="HIV examples")
    return results, metrics


def build_precision_inputs(
    datasets,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
    dataset_order=None,
    method_order=None,
):
    """Build real/synthetic dictionaries consumed by Figure 4 plotting helpers."""
    dataset_order = list(dataset_order or datasets.keys())
    method_order = list(method_order or METHOD_ORDER)
    real_data = {}
    synthetic_data = {}
    feature_names = {}
    for dataset in dataset_order:
        data = datasets[dataset]
        real_data[dataset] = np.asarray(data["X"], dtype=np.float64)
        feature_names[dataset] = list(data.get("feature_names", [f"f{i}" for i in range(real_data[dataset].shape[1])]))
        synthetic_data[dataset] = {}
        for method in method_order:
            print(f"[figure4 generate] {dataset} - {method} seed={seed}")
            X_syn, _ = sample_synthetic(dataset, data, method, seed=seed, cvae_epochs=cvae_epochs)
            synthetic_data[dataset][method] = np.asarray(X_syn, dtype=np.float64)
    return real_data, synthetic_data, feature_names


def verify_generator_seeds(
    datasets,
    seed=SEED,
    cvae_epochs=3,
    dataset_order=None,
    method_order=None,
):
    """Check same-seed reproducibility for each configured generator."""
    dataset_order = list(dataset_order or datasets.keys())
    method_order = list(method_order or METHOD_ORDER)
    rows = []
    for dataset in dataset_order:
        data = datasets[dataset]
        for method in method_order:
            X1, y1 = sample_synthetic(dataset, data, method, seed=seed, cvae_epochs=cvae_epochs)
            X2, y2 = sample_synthetic(dataset, data, method, seed=seed, cvae_epochs=cvae_epochs)
            X3, _ = sample_synthetic(dataset, data, method, seed=seed + 1, cvae_epochs=cvae_epochs)
            rows.append({
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "same_seed_equal": bool(np.array_equal(X1, X2) and np.array_equal(y1, y2)),
                "different_seed_changed": bool(not np.array_equal(X1, X3)),
                "shape": tuple(np.asarray(X1).shape),
                "class_counts": class_counts(y1),
            })
    return pd.DataFrame(rows)


def _dataset_slug(dataset):
    return str(dataset).lower().replace(" ", "_").replace("/", "_")


def export_figure4_supplemental_figures(
    real_data,
    synthetic_data,
    feature_names,
    output_dir=None,
    seed=SEED,
    dataset_order=None,
    method_order=None,
    alphas=None,
    exemplar_ds=None,
    cluster_dataset_order=None,
    include_edge_examples=True,
    include_main_matrices=True,
    include_cluster_summary=True,
    include_tsne=True,
    include_cosine_tsne=True,
):
    """Export Figure 4 panels from prebuilt real/synthetic dictionaries."""
    dataset_order = list(dataset_order or real_data.keys())
    method_order = list(method_order or METHOD_ORDER)
    alphas = dict(alphas or FIGURE4_ALPHAS)
    exemplar_ds = exemplar_ds or dataset_order[0]
    cluster_dataset_order = list(cluster_dataset_order or dataset_order)
    output_dir = Path(output_dir or repo_root / "data_synthesis" / "notebooks" / "revision_exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    tsne_output_dir = output_dir / "t-SNE analysis"
    if include_tsne:
        tsne_output_dir.mkdir(parents=True, exist_ok=True)

    exported_figures = []
    all_metrics = []
    results = {}

    if include_edge_examples:
        examples_path = output_dir / f"supplemental_figure_s1_{_dataset_slug(exemplar_ds)}_edge_status_examples.png"
        examples_result = plot_edge_status_examples(
            real_data=real_data,
            synthetic_data=synthetic_data,
            feature_names=feature_names,
            alphas=alphas,
            dataset_order=dataset_order,
            method_order=method_order,
            exemplar_ds=exemplar_ds,
            save_path=examples_path,
        )
        exported_figures.append({
            "section": "Supplement",
            "dataset": exemplar_ds,
            "figure": "Edge-status examples",
            "path": str(examples_path),
        })
        all_metrics.append(examples_result.metrics.assign(figure_dataset=f"{exemplar_ds} examples"))
        results[f"{exemplar_ds} examples"] = examples_result
        plt.close(examples_result.fig)

    if include_main_matrices:
        matrices_path = output_dir / f"figure4_{_dataset_slug(exemplar_ds)}_edge_status_matrices.png"
        matrices_result = plot_figure4_edge_status_matrices(
            real_data=real_data,
            synthetic_data=synthetic_data,
            feature_names=feature_names,
            alphas=alphas,
            dataset_order=dataset_order,
            method_order=method_order,
            exemplar_ds=exemplar_ds,
            save_path=matrices_path,
        )
        exported_figures.append({
            "section": "Main text",
            "dataset": exemplar_ds,
            "figure": "Graphical Lasso edge-status matrices",
            "path": str(matrices_path),
        })
        all_metrics.append(matrices_result.metrics.assign(figure_dataset=f"{exemplar_ds} matrices"))
        results[f"{exemplar_ds} matrices"] = matrices_result
        plt.close(matrices_result.fig)

    if include_cluster_summary:
        cluster_summary_path = output_dir / "figure4_cluster_summary_all_datasets.png"
        cluster_summary_result = plot_figure4_cluster_summary_grid(
            real_data=real_data,
            synthetic_data=synthetic_data,
            feature_names=feature_names,
            alphas=alphas,
            dataset_order=cluster_dataset_order,
            method_order=method_order,
            seed=seed,
            save_path=cluster_summary_path,
        )
        exported_figures.append({
            "section": "Main text",
            "dataset": "All datasets",
            "figure": "Cluster-level t-SNE summaries",
            "path": str(cluster_summary_path),
        })
        all_metrics.append(cluster_summary_result.metrics.assign(figure_dataset="all dataset cluster summaries"))
        results["all dataset cluster summaries"] = cluster_summary_result
        plt.close(cluster_summary_result.fig)

    if include_tsne:
        for dataset in dataset_order:
            tsne_path = tsne_output_dir / f"supplemental_tsne_analysis_{_dataset_slug(dataset)}.png"
            result = plot_figure4_tsne_edge_supplement(
                real_data=real_data,
                synthetic_data=synthetic_data,
                feature_names=feature_names,
                alphas=alphas,
                dataset_order=dataset_order,
                method_order=method_order,
                exemplar_ds=dataset,
                seed=seed,
                save_path=tsne_path,
            )
            exported_figures.append({
                "section": "Supplement: t-SNE analysis",
                "dataset": dataset,
                "figure": "Graphical Lasso t-SNE edge overlays",
                "path": str(tsne_path),
            })
            all_metrics.append(result.metrics.assign(figure_dataset=f"{dataset} t-SNE analysis"))
            results[f"{dataset} t-SNE analysis"] = result
            plt.close(result.fig)

            if include_cosine_tsne:
                tsne_cosine_path = tsne_output_dir / f"supplemental_tsne_analysis_{_dataset_slug(dataset)}_cosine_clusters.png"
                cosine_result = plot_figure4_tsne_edge_supplement(
                    real_data=real_data,
                    synthetic_data=synthetic_data,
                    feature_names=feature_names,
                    alphas=alphas,
                    dataset_order=dataset_order,
                    method_order=method_order,
                    exemplar_ds=dataset,
                    seed=seed,
                    cluster_metric="cosine",
                    cluster_linkage="average",
                    save_path=tsne_cosine_path,
                )
                exported_figures.append({
                    "section": "Supplement: t-SNE analysis",
                    "dataset": dataset,
                    "figure": "Graphical Lasso t-SNE edge overlays with cosine cluster distance",
                    "path": str(tsne_cosine_path),
                })
                all_metrics.append(cosine_result.metrics.assign(figure_dataset=f"{dataset} t-SNE analysis cosine clusters"))
                results[f"{dataset} t-SNE analysis cosine clusters"] = cosine_result
                plt.close(cosine_result.fig)

    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    metrics.to_csv(output_dir / "figure4_structural_metrics.csv", index=False)
    pd.DataFrame(exported_figures).to_csv(output_dir / "figure4_exported_figures.csv", index=False)
    return results, metrics


def export_figure4_supplementals(
    datasets=None,
    output_dir=None,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
    verify_epochs=3,
    dataset_order=None,
    method_order=None,
    alphas=None,
    exemplar_ds=None,
    cluster_dataset_order=None,
    verify=True,
    **export_options,
):
    """Hot-swappable Figure 4 export pipeline.

    Pass any dataset dictionary with entries containing ``X``, ``y``, and
    optional ``feature_names``. Override ``dataset_order`` and ``alphas`` when
    swapping in new datasets.
    """
    datasets = load_datasets() if datasets is None else datasets
    dataset_order = list(dataset_order or datasets.keys())
    method_order = list(method_order or METHOD_ORDER)
    alphas = dict(alphas or FIGURE4_ALPHAS)
    output_dir = Path(output_dir or repo_root / "data_synthesis" / "notebooks" / "revision_exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_checks = pd.DataFrame()
    if verify:
        seed_checks = verify_generator_seeds(
            datasets,
            seed=seed,
            cvae_epochs=verify_epochs,
            dataset_order=dataset_order,
            method_order=method_order,
        )
        print(seed_checks.to_string(index=False))
        if not seed_checks["same_seed_equal"].all():
            raise RuntimeError("At least one generator failed same-seed determinism.")
        seed_checks.to_csv(output_dir / "figure4_seed_checks.csv", index=False)

    real_data, synthetic_data, names = build_precision_inputs(
        datasets,
        seed=seed,
        cvae_epochs=cvae_epochs,
        dataset_order=dataset_order,
        method_order=method_order,
    )

    results, metrics = export_figure4_supplemental_figures(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=names,
        output_dir=output_dir,
        seed=seed,
        dataset_order=dataset_order,
        method_order=method_order,
        alphas=alphas,
        exemplar_ds=exemplar_ds,
        cluster_dataset_order=cluster_dataset_order,
        **export_options,
    )
    print(f"Saved Figure 4 supplementals to {output_dir}")
    return results, metrics, seed_checks
