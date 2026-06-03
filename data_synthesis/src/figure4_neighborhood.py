from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn import covariance
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


EDGE_COLORS = {
    "preserved": "#2F6DB3",
    "real_only": "#C43C39",
    "synthetic_only": "#E88925",
}

METRIC_LABELS = {
    "frobenius_deviation": r"Frobenius deviation, $||\Theta_R-\Theta_S||_F$",
    "edge_recovery": r"Edge recovery, $|E_R \cap E_S| / |E_R|$",
    "synthetic_only_rate": r"Synthetic-only edge rate, $|E_S \setminus E_R| / |E_S|$",
}

METHOD_PRESERVATION_COLORS = {
    "Bootstrap": "#6A5ACD",
    "Column-wise": "#CC79A7",
    "GMM": "#009E73",
    "CVAE": "#D55E00",
}

METHOD_PRESERVATION_PASTELS = {
    "Bootstrap": "#C7C2F4",
    "Column-wise": "#E8B4D2",
    "GMM": "#A8DEC9",
    "CVAE": "#F2B49B",
}


@dataclass
class Figure4Result:
    fig: plt.Figure
    metrics: pd.DataFrame
    anchor: int
    anchor_feature: str
    structures: dict
    edge_recovery: pd.DataFrame | None = None
    feature_index: pd.DataFrame | None = None


def _prepare_glasso_input(X):
    X = np.asarray(X, dtype=np.float64)
    X = np.where(np.isfinite(X), X, np.nan)
    col_medians = np.nanmedian(X, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
    missing = np.where(~np.isfinite(X))
    if len(missing[0]):
        X = X.copy()
        X[missing] = np.take(col_medians, missing[1])
    Xs = StandardScaler().fit_transform(X)
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)


def fit_glasso_precision(X, alpha):
    """Fit Graphical Lasso on standardized features and return a precision matrix."""
    Xs = _prepare_glasso_input(X)
    if alpha is None:
        cv = min(5, max(2, Xs.shape[0] // 3))
        try:
            model = covariance.GraphicalLassoCV(
                alphas=8, cv=cv, max_iter=1000, n_refinements=3
            ).fit(Xs)
            alpha = float(model.alpha_)
        except Exception:
            alpha = 0.08
    try:
        model = covariance.GraphicalLasso(alpha=float(alpha), max_iter=1000).fit(Xs)
        return np.asarray(model.precision_, dtype=float)
    except Exception:
        empirical = covariance.EmpiricalCovariance().fit(Xs)
        return np.linalg.pinv(empirical.covariance_)


def _estimate_glasso_alpha(X, cv_folds=5, max_iter=1000):
    Xs = _prepare_glasso_input(X)
    cv = min(cv_folds, max(2, Xs.shape[0] // 3))
    model = covariance.GraphicalLassoCV(
        alphas=8, cv=cv, max_iter=max_iter, n_refinements=3
    ).fit(Xs)
    return float(model.alpha_)


def precision_to_partial_corr(theta):
    """Convert a precision matrix to a partial-correlation matrix."""
    theta = np.asarray(theta, dtype=np.float64)
    diag = np.clip(np.diag(theta), 1e-12, None)
    denom = np.sqrt(np.outer(diag, diag))
    partial = -theta / denom
    np.fill_diagonal(partial, 0.0)
    partial[np.abs(partial) < 1e-12] = 0.0
    return np.clip(partial, -1.0, 1.0)


def get_edge_set(partial_corr, threshold=1e-7):
    """Return undirected off-diagonal edges with absolute partial correlation above threshold."""
    partial_corr = np.asarray(partial_corr, dtype=float)
    edges = set()
    for i in range(partial_corr.shape[0]):
        for j in range(i + 1, partial_corr.shape[1]):
            if abs(float(partial_corr[i, j])) > threshold:
                edges.add((i, j))
    return edges


def choose_anchor_feature(partial_corr, feature_names):
    """Choose the highest-degree real-network node, breaking ties by total absolute weight."""
    partial_corr = np.asarray(partial_corr, dtype=float)
    edges = get_edge_set(partial_corr)
    degree = np.zeros(partial_corr.shape[0], dtype=int)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    strength = np.sum(np.abs(partial_corr), axis=1)
    order = np.lexsort((-strength, -degree))
    return int(order[0])


def _normalize_edge(edge):
    i, j = edge
    return (i, j) if i < j else (j, i)


def get_anchor_neighborhood_edges(real_edges, synthetic_edges, anchor):
    """Return edge categories incident to anchor in either the real or synthetic network."""
    real_anchor = {_normalize_edge(e) for e in real_edges if anchor in e}
    syn_anchor = {_normalize_edge(e) for e in synthetic_edges if anchor in e}
    return {
        "preserved": real_anchor & syn_anchor,
        "real_only": real_anchor - syn_anchor,
        "synthetic_only": syn_anchor - real_anchor,
        "all": real_anchor | syn_anchor,
    }


def _filter_anchor_neighborhood(categories, real_partial, synthetic_partial, max_neighbors=None, prefer="preserved"):
    if max_neighbors is None or len(categories["all"]) <= max_neighbors:
        return categories
    priority = {
        "preserved": {"preserved": 0, "real_only": 1, "synthetic_only": 2},
        "lost": {"real_only": 0, "preserved": 1, "synthetic_only": 2},
    }.get(prefer, {"preserved": 0, "real_only": 1, "synthetic_only": 2})
    rows = []
    for category in ("preserved", "real_only", "synthetic_only"):
        for edge in categories[category]:
            rows.append((
                priority[category],
                -_edge_weight(edge, real_partial, synthetic_partial, category),
                category,
                edge,
            ))
    kept = sorted(rows)[:max_neighbors]
    out = {"preserved": set(), "real_only": set(), "synthetic_only": set()}
    for _, _, category, edge in kept:
        out[category].add(edge)
    out["all"] = out["preserved"] | out["real_only"] | out["synthetic_only"]
    return out


def compute_edge_recovery(real_edges, synthetic_edges):
    return float(len(real_edges & synthetic_edges) / len(real_edges)) if real_edges else np.nan


def compute_synthetic_only_rate(real_edges, synthetic_edges):
    return float(len(synthetic_edges - real_edges) / len(synthetic_edges)) if synthetic_edges else np.nan


def compute_frobenius_deviation(theta_real, theta_syn):
    theta_real = np.asarray(theta_real, dtype=float).copy()
    theta_syn = np.asarray(theta_syn, dtype=float).copy()
    np.fill_diagonal(theta_real, 0.0)
    np.fill_diagonal(theta_syn, 0.0)
    return float(np.linalg.norm(theta_real - theta_syn, ord="fro"))


def _edge_weight(edge, real_partial, synthetic_partial, category):
    i, j = edge
    if category == "preserved":
        return max(abs(float(real_partial[i, j])), abs(float(synthetic_partial[i, j])))
    if category == "real_only":
        return abs(float(real_partial[i, j]))
    return abs(float(synthetic_partial[i, j]))


def _edge_sign(edge, real_partial, synthetic_partial, category):
    i, j = edge
    value = real_partial[i, j] if category in {"preserved", "real_only"} else synthetic_partial[i, j]
    return 1 if float(value) >= 0 else -1


def _short_label(label, max_chars=18):
    label = str(label)
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1] + "."




def _anchor_layout(nodes, anchor):
    neighbors = sorted([n for n in nodes if n != anchor])
    pos = {anchor: np.array([0.0, 0.0])}
    if not neighbors:
        return pos
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(neighbors), endpoint=False)
    for node, angle in zip(neighbors, angles):
        pos[node] = np.array([np.cos(angle), np.sin(angle)])
    return pos




def _fit_structures(real_data, synthetic_data, alphas=None, threshold=1e-7, dataset_order=None, method_order=None):
    dataset_order = list(dataset_order or real_data.keys())
    method_order = list(method_order or synthetic_data[dataset_order[0]].keys())
    structures = {}
    rows = []

    for dataset in dataset_order:
        alpha = None if alphas is None else alphas.get(dataset)
        if alpha is None:
            alpha = _estimate_glasso_alpha(real_data[dataset])
        theta_real = fit_glasso_precision(real_data[dataset], alpha)
        real_partial = precision_to_partial_corr(theta_real)
        real_edges = get_edge_set(real_partial, threshold)
        structures[dataset] = {
            "real": {"theta": theta_real, "partial": real_partial, "edges": real_edges},
            "synthetic": {},
            "alpha": alpha,
        }
        for method in method_order:
            theta_syn = fit_glasso_precision(synthetic_data[dataset][method], alpha)
            syn_partial = precision_to_partial_corr(theta_syn)
            syn_edges = get_edge_set(syn_partial, threshold)
            structures[dataset]["synthetic"][method] = {
                "theta": theta_syn, "partial": syn_partial, "edges": syn_edges,
            }
            rows.append({
                "dataset": dataset,
                "method": method,
                "frobenius_deviation": compute_frobenius_deviation(theta_real, theta_syn),
                "edge_recovery": compute_edge_recovery(real_edges, syn_edges),
                "synthetic_only_rate": compute_synthetic_only_rate(real_edges, syn_edges),
                "n_real_edges": len(real_edges),
                "n_synthetic_edges": len(syn_edges),
            })

    return structures, pd.DataFrame(rows)


def plot_summary_metrics(axs, metrics, dataset_order=None, method_order=None, metric_names=None, palette=None):
    """Plot grouped metric bars for Panel E."""
    if not isinstance(axs, Iterable) or hasattr(axs, "bar"):
        axs = [axs]
    axs = list(axs)
    metric_names = list(metric_names or ["frobenius_deviation", "edge_recovery", "synthetic_only_rate"])
    dataset_order = list(dataset_order or metrics["dataset"].drop_duplicates())
    method_order = list(method_order or metrics["method"].drop_duplicates())
    palette = palette or METHOD_PRESERVATION_PASTELS

    width = 0.18
    x = np.arange(len(dataset_order))
    for ax, metric in zip(axs, metric_names):
        table = metrics.pivot(index="dataset", columns="method", values=metric).reindex(dataset_order)
        for i, method in enumerate(method_order):
            vals = table[method].to_numpy(dtype=float)
            offset = (i - (len(method_order) - 1) / 2) * width
            ax.bar(
                x + offset, vals, width=width, label=method, color=palette.get(method, "#BBBBBB"),
                edgecolor="#333333", linewidth=0.75
            )
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9.2, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_order, fontsize=8.0)
        ax.tick_params(axis="y", labelsize=8.0)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return axs


def plot_figure4_metric_summary(
    metrics,
    dataset_order=None,
    method_order=None,
    metric_names=None,
    title="E. Global structural deviation across datasets",
):
    """Plot Figure 4 metric summary as a standalone panel."""
    dataset_order = list(dataset_order or metrics["dataset"].drop_duplicates())
    method_order = list(method_order or metrics["method"].drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.8), constrained_layout=False)
    plot_summary_metrics(
        axes,
        metrics,
        dataset_order=dataset_order,
        method_order=method_order,
        metric_names=metric_names,
    )
    axes[1].text(
        0.5,
        1.24,
        title,
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        weight="semibold",
    )
    axes[-1].legend(loc="upper right", fontsize=7.5, frameon=True, edgecolor="#BBBBBB")
    fig.subplots_adjust(left=0.070, right=0.985, top=0.70, bottom=0.25, wspace=0.30)
    return fig


def _anchor_score(real_edges, synthetic_edges, anchor, mode):
    categories = get_anchor_neighborhood_edges(real_edges, synthetic_edges, anchor)
    if mode == "preserved":
        return (len(categories["preserved"]), -len(categories["real_only"]), len(categories["all"]))
    if mode == "lost":
        return (len(categories["real_only"]), -len(categories["preserved"]), len(categories["all"]))
    return (len(categories["all"]), 0, 0)


def choose_example_anchor(real_edges, synthetic_edges, n_features, mode="preserved"):
    scores = [
        (*_anchor_score(real_edges, synthetic_edges, anchor, mode), -anchor, anchor)
        for anchor in range(n_features)
    ]
    return int(max(scores)[-1])


def _edge_label(edge, feature_names, max_chars=14):
    left = _short_label(feature_names[edge[0]], max_chars)
    right = _short_label(feature_names[edge[1]], max_chars)
    return f"{left} | {right}"


def build_edge_recovery_matrix(real_partial, real_edges, synthetic_edge_map, feature_names, top_n=35):
    ranked_edges = sorted(
        real_edges,
        key=lambda edge: abs(float(real_partial[edge[0], edge[1]])),
        reverse=True,
    )[:top_n]
    rows = []
    for rank, edge in enumerate(ranked_edges, start=1):
        row = {
            "rank": rank,
            "edge": edge,
            "edge_label": f"{rank}. {_edge_label(edge, feature_names)}",
            "real_abs_partial_corr": abs(float(real_partial[edge[0], edge[1]])),
            "real_partial_corr": float(real_partial[edge[0], edge[1]]),
        }
        for method, synthetic_edges in synthetic_edge_map.items():
            row[method] = int(edge in synthetic_edges)
        rows.append(row)
    return pd.DataFrame(rows)





STATUS_CODES = {
    "absent": 0,
    "preserved": 1,
    "real_only": 2,
    "synthetic_only": 3,
}

STATUS_COLORS = {
    "absent": "#F3F5F7",
    "preserved": "#1F5A93",
    "real_only": "#C83F3F",
    "synthetic_only": "#E98A2A",
}


def get_real_structure_order(real_partial):
    """Order features by hierarchical clustering on the real partial-correlation structure."""
    structure = np.abs(np.asarray(real_partial, dtype=float))
    np.fill_diagonal(structure, 1.0)
    distance = 1.0 - np.clip(structure, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    if np.allclose(condensed, condensed[0] if len(condensed) else 0.0):
        return np.arange(structure.shape[0])
    linkage = hierarchy.linkage(condensed, method="average")
    return hierarchy.leaves_list(linkage)


def build_edge_status_matrix(real_edges, synthetic_edges, n_features):
    """Build a symmetric categorical matrix comparing real and synthetic edge sets."""
    status = np.zeros((n_features, n_features), dtype=int)
    all_edges = real_edges | synthetic_edges
    for edge in all_edges:
        i, j = edge
        if edge in real_edges and edge in synthetic_edges:
            code = STATUS_CODES["preserved"]
        elif edge in real_edges:
            code = STATUS_CODES["real_only"]
        else:
            code = STATUS_CODES["synthetic_only"]
        status[i, j] = code
        status[j, i] = code
    np.fill_diagonal(status, STATUS_CODES["absent"])
    return status


def make_feature_index_table(feature_names, order):
    return pd.DataFrame({
        "matrix_index": np.arange(1, len(order) + 1),
        "feature_original_index": np.asarray(order, dtype=int) + 1,
        "feature_name": [feature_names[i] for i in order],
    })










def plot_edge_status_examples(
    real_data,
    synthetic_data,
    feature_names,
    alphas=None,
    dataset_order=None,
    method_order=None,
    exemplar_ds="HIV",
    threshold=1e-7,
    save_path=None,
):
    """Export the explanatory edge-status zoom and graph panels separately."""
    method_order = list(method_order or synthetic_data[exemplar_ds].keys())
    dataset_order = list(dataset_order or real_data.keys())
    structures, metrics = _fit_structures(
        real_data,
        synthetic_data,
        alphas=alphas,
        threshold=threshold,
        dataset_order=dataset_order,
        method_order=method_order,
    )

    names = list(feature_names[exemplar_ds] if isinstance(feature_names, Mapping) else feature_names)
    real = structures[exemplar_ds]["real"]
    real_edges = real["edges"]
    real_partial = real["partial"]
    order = get_real_structure_order(real_partial)
    feature_index = make_feature_index_table(names, order)

    example_method = "Bootstrap" if "Bootstrap" in method_order else method_order[0]
    example_status = build_edge_status_matrix(
        real_edges,
        structures[exemplar_ds]["synthetic"][example_method]["edges"],
        real_partial.shape[0],
    )

    fig = plt.figure(figsize=(12.0, 4.9), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.18], wspace=0.20)
    example_anchor, _ = plot_real_status_zoom(
        fig.add_subplot(gs[0, 0]),
        example_status,
        order,
        "A. Edge-status cells",
    )
    plot_feature_row_graph(
        fig.add_subplot(gs[0, 1]),
        real_edges,
        structures[exemplar_ds]["synthetic"][example_method]["edges"],
        real_partial,
        names,
        "B. Selected row as a graph",
        anchor=example_anchor,
        order=order,
    )

    legend_handles = [
        Patch(facecolor=STATUS_COLORS["preserved"], edgecolor="#333333", label="Preserved edge"),
        Patch(facecolor=STATUS_COLORS["real_only"], edgecolor="#333333", label="Real-only / lost"),
        Patch(facecolor=STATUS_COLORS["synthetic_only"], edgecolor="#333333", label="Synthetic-only"),
        Patch(facecolor=STATUS_COLORS["absent"], edgecolor="#C9CDD2", label="Absent in both"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.35,
    )
    fig.suptitle(
        f"Reading edge-status matrices for {exemplar_ds}",
        y=0.985,
        fontsize=14.5,
        weight="semibold",
    )
    fig.subplots_adjust(left=0.060, right=0.985, top=0.760, bottom=0.135)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return Figure4Result(
        fig=fig,
        metrics=metrics,
        anchor=example_anchor,
        anchor_feature=names[example_anchor],
        structures=structures,
        feature_index=feature_index,
    )


def plot_edge_status_example(ax, title, example="preserved_lost"):
    status = np.zeros((5, 5), dtype=int)
    if example == "preserved_lost":
        pairs = {
            (0, 2): STATUS_CODES["preserved"],
            (1, 4): STATUS_CODES["real_only"],
        }
        notes = [
            ("Preserved real edge", 0, 2, STATUS_COLORS["preserved"]),
            ("Lost real edge", 1, 4, STATUS_COLORS["real_only"]),
        ]
    else:
        pairs = {
            (0, 3): STATUS_CODES["synthetic_only"],
            (2, 4): STATUS_CODES["absent"],
        }
        notes = [
            ("Synthetic-only edge", 0, 3, STATUS_COLORS["synthetic_only"]),
            ("Blank means absent", 2, 4, "#5F666D"),
        ]

    for (i, j), code in pairs.items():
        status[i, j] = code
        status[j, i] = code

    cmap = ListedColormap([
        STATUS_COLORS["absent"],
        STATUS_COLORS["preserved"],
        STATUS_COLORS["real_only"],
        STATUS_COLORS["synthetic_only"],
    ])
    ax.imshow(status, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=10.5, weight="semibold", pad=7)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(range(1, 6), fontsize=7.5)
    ax.set_yticklabels(range(1, 6), fontsize=7.5)
    ax.set_xlabel("Feature index", fontsize=7.8)
    ax.set_ylabel("Feature index", fontsize=7.8)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(length=2.0, width=0.8)
    for idx, (label, i, j, color) in enumerate(notes):
        if status[i, j] != STATUS_CODES["absent"]:
            ax.text(j, i, "x", ha="center", va="center", fontsize=9.0, weight="bold", color="white")
        ax.text(
            1.02,
            0.82 - 0.17 * idx,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.4,
            color=color,
        )
    ax.text(
        0.5,
        -0.20,
        "One off-diagonal cell pair represents one conditional edge.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color="#4B4B4B",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")
    return ax


def _status_cmap():
    return ListedColormap([
        STATUS_COLORS["absent"],
        STATUS_COLORS["preserved"],
        STATUS_COLORS["real_only"],
        STATUS_COLORS["synthetic_only"],
    ])


def plot_real_status_zoom(ax, status_matrix, order, title, window=12):
    ordered = status_matrix[np.ix_(order, order)]
    non_absent = (ordered != STATUS_CODES["absent"]).astype(float)
    best_start = 0
    best_score = -1
    max_start = max(0, len(order) - window)
    for start in range(max_start + 1):
        block = non_absent[start:start + window, start:start + window]
        score = float(block.sum() - np.trace(block))
        if score > best_score:
            best_score = score
            best_start = start

    zoom = ordered[best_start:best_start + window, best_start:best_start + window]
    matrix_indices = np.arange(best_start + 1, best_start + len(zoom) + 1)
    focus_local = int(np.argmax(non_absent[best_start:best_start + len(zoom), best_start:best_start + len(zoom)].sum(axis=1)))
    focus_matrix_index = int(matrix_indices[focus_local])
    focus_feature = int(order[best_start + focus_local])

    rgba = _status_cmap()(np.clip(zoom, 0, 3))
    alpha = np.full(zoom.shape, 0.22, dtype=float)
    alpha[focus_local, :] = 1.0
    alpha[:, focus_local] = 1.0
    np.fill_diagonal(alpha, 0.48)
    rgba[..., 3] = alpha

    ax.imshow(rgba, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=10.5, weight="semibold", pad=7)
    ax.set_xticks(np.arange(len(zoom)))
    ax.set_yticks(np.arange(len(zoom)))
    ax.set_xticklabels(matrix_indices, fontsize=6.5, rotation=90)
    ax.set_yticklabels(matrix_indices, fontsize=6.5)
    ax.set_xlabel("Feature index", fontsize=7.6)
    ax.set_ylabel("Feature index", fontsize=7.6)
    ax.set_xticks(np.arange(-0.5, len(zoom), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(zoom), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(length=1.8, width=0.7)
    ax.axhline(focus_local - 0.5, color="#111111", linewidth=1.1)
    ax.axhline(focus_local + 0.5, color="#111111", linewidth=1.1)
    ax.axvline(focus_local - 0.5, color="#111111", linewidth=1.1)
    ax.axvline(focus_local + 0.5, color="#111111", linewidth=1.1)
    ax.text(
        0.5,
        -0.20,
        f"Focus on feature #{focus_matrix_index}: its row/column is converted to the graph at right.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color="#4B4B4B",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")
    return focus_feature, focus_matrix_index


def plot_feature_row_graph(
    ax,
    real_edges,
    synthetic_edges,
    partial_corr,
    feature_names,
    title,
    anchor=None,
    order=None,
    max_neighbors=None,
):
    if anchor is None:
        degrees = np.zeros(len(feature_names), dtype=int)
        for i, j in real_edges:
            degrees[i] += 1
            degrees[j] += 1
        anchor = int(np.argmax(degrees))
    incident = [edge for edge in real_edges | synthetic_edges if anchor in edge]
    order_lookup = {int(feature): idx + 1 for idx, feature in enumerate(order)} if order is not None else None

    def sort_key(edge):
        i, j = edge
        other = j if i == anchor else i
        code_rank = 0 if edge in real_edges and edge in synthetic_edges else 1 if edge in real_edges else 2
        return (code_rank, -abs(float(partial_corr[anchor, other])))

    incident = sorted(incident, key=sort_key)
    if max_neighbors is not None:
        incident = incident[:max_neighbors]
    nodes = {anchor}
    for edge in incident:
        nodes.update(edge)
    pos = _anchor_layout(nodes, anchor)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(incident)

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=list(graph.nodes),
        node_size=[620 if n == anchor else 260 for n in graph.nodes],
        node_color=["#F7C948" if n == anchor else "#F7F7F7" for n in graph.nodes],
        edgecolors="#303030",
        linewidths=1.0,
        ax=ax,
    )
    for edge in incident:
        if edge in real_edges and edge in synthetic_edges:
            color = STATUS_COLORS["preserved"]
        elif edge in real_edges:
            color = STATUS_COLORS["real_only"]
        else:
            color = STATUS_COLORS["synthetic_only"]
        i, j = edge
        other = j if i == anchor else i
        width = 1.0 + 6.0 * abs(float(partial_corr[anchor, other]))
        style = "solid" if float(partial_corr[anchor, other]) >= 0 else "dashed"
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], width=width, edge_color=color, style=style, ax=ax)

    for node in graph.nodes:
        x, y = pos[node]
        label = str(order_lookup[node] if order_lookup is not None else node + 1)
        ax.text(
            x,
            y,
            label,
            fontsize=7.0 if node == anchor else 6.2,
            ha="center",
            va="center",
            weight="bold",
            color="#111111",
            zorder=5,
        )
    ax.set_title(title, fontsize=10.5, weight="semibold", pad=7)
    anchor_index = order_lookup[anchor] if order_lookup is not None else anchor + 1
    ax.text(
        0.5,
        -0.08,
        f"Graph view of feature #{anchor_index}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.4,
        color="#4B4B4B",
    )
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def plot_figure4_edge_status_matrices(
    real_data,
    synthetic_data,
    feature_names,
    alphas=None,
    dataset_order=None,
    method_order=None,
    exemplar_ds="HIV",
    comparison_methods=None,
    threshold=1e-7,
    save_path=None,
):
    method_order = list(method_order or synthetic_data[exemplar_ds].keys())
    dataset_order = list(dataset_order or real_data.keys())
    comparison_methods = list(comparison_methods or ["Bootstrap", "Column-wise", "CVAE", "GMM"])
    comparison_methods = [m for m in comparison_methods if m in method_order][:4]
    if len(comparison_methods) < 4:
        for method in method_order:
            if method not in comparison_methods:
                comparison_methods.append(method)
            if len(comparison_methods) == 4:
                break

    structures, metrics = _fit_structures(
        real_data, synthetic_data, alphas=alphas, threshold=threshold,
        dataset_order=dataset_order, method_order=method_order
    )

    names = list(feature_names[exemplar_ds] if isinstance(feature_names, Mapping) else feature_names)
    real = structures[exemplar_ds]["real"]
    real_edges = real["edges"]
    real_partial = real["partial"]
    order = get_real_structure_order(real_partial)
    feature_index = make_feature_index_table(names, order)

    panels = ["A", "B", "C", "D"]
    n_features = real_partial.shape[0]
    block_status = []
    for method in comparison_methods:
        syn_edges = structures[exemplar_ds]["synthetic"][method]["edges"]
        status = build_edge_status_matrix(real_edges, syn_edges, real_partial.shape[0])
        block_status.append(status[np.ix_(order, order)])

    composite = np.block([
        [block_status[0], block_status[1]],
        [block_status[2], block_status[3]],
    ])
    cmap = ListedColormap([
        STATUS_COLORS["absent"],
        STATUS_COLORS["preserved"],
        STATUS_COLORS["real_only"],
        STATUS_COLORS["synthetic_only"],
    ])

    fig, ax = plt.subplots(figsize=(11.2, 10.6), constrained_layout=False)
    ax.imshow(composite, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest", aspect="equal")
    ax.axvline(n_features - 0.5, color="#111111", linewidth=1.25)
    ax.axhline(n_features - 0.5, color="#111111", linewidth=1.25)

    tick_step = 1 if n_features <= 12 else 5 if n_features <= 35 else 10
    base_ticks = np.arange(0, n_features, tick_step)
    ticks = np.r_[base_ticks, n_features + base_ticks]
    tick_labels = [str(i + 1) for i in base_ticks] * 2
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=7.8)
    ax.set_yticklabels(tick_labels, fontsize=7.8)
    ax.tick_params(axis="both", which="major", length=2.2, width=0.8, pad=1.5)
    ax.set_xlabel("")
    ax.set_ylabel("")

    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(ticks)
    top_ax.set_xticklabels(tick_labels, fontsize=7.8)
    top_ax.tick_params(length=2.2, width=0.8, pad=1.5)

    panel_positions = [
        (0.018, 0.518),
        (0.518, 0.518),
        (0.018, 0.018),
        (0.518, 0.018),
    ]
    for panel, method, (x, y) in zip(panels, comparison_methods, panel_positions):
        ax.text(
            x,
            y,
            f"{panel}. {method}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12.0,
            weight="semibold",
            color="#111111",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.8),
            zorder=5,
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
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
        bbox_to_anchor=(0.5, 0.060),
        ncol=4,
        frameon=False,
        fontsize=9.2,
        handlelength=1.55,
        handletextpad=0.48,
        columnspacing=1.18,
        borderaxespad=0.3,
    )

    # fig.suptitle(
    #     "Figure 4. Real conditional-dependency preservation and synthetic structural deviation",
    #     y=0.978,
    #     fontsize=15.5,
    #     weight="semibold",
    # )
    fig.subplots_adjust(left=0.066, right=0.968, top=0.925, bottom=0.140)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    result = Figure4Result(
        fig=fig,
        metrics=metrics,
        anchor=-1,
        anchor_feature="",
        structures=structures,
        feature_index=feature_index,
    )
    empty_group_summary = pd.DataFrame(columns=[
        "cluster_id",
        "method",
        "n_features",
        "n_features_matching_method",
        "prominent_features",
        "feature_indices",
        "center_x",
        "center_y",
    ])
    result.preserve_group_summary = empty_group_summary.copy()
    result.lost_group_summary = empty_group_summary.copy()
    result.synthetic_only_group_summary = empty_group_summary.copy()
    result.neighborhood_summary = result.preserve_group_summary
    return result



def _fit_profile_tsne(partial_corr, seed=123, perplexity=None):
    """Embed features from their Graphical Lasso partial-correlation profiles."""
    profiles = np.asarray(partial_corr, dtype=float).copy()
    np.fill_diagonal(profiles, 0.0)
    profiles = StandardScaler().fit_transform(profiles)
    n_features = profiles.shape[0]
    if perplexity is None:
        perplexity = min(30, max(2, (n_features - 1) // 3))
    perplexity = float(min(perplexity, max(1, n_features - 1)))
    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
    )
    try:
        coords = TSNE(max_iter=1500, **kwargs).fit_transform(profiles)
    except TypeError:
        coords = TSNE(n_iter=1500, **kwargs).fit_transform(profiles)
    return coords, profiles, perplexity


def build_feature_preservation_scores(real_edges, synthetic_edge_map, n_features, epsilon=1e-12):
    """Score per-feature preservation of incident Graphical Lasso edges by method."""
    rows = []
    real_incident = [set() for _ in range(n_features)]
    for edge in real_edges:
        i, j = edge
        real_incident[i].add(edge)
        real_incident[j].add(edge)

    for method, synthetic_edges in synthetic_edge_map.items():
        syn_incident = [set() for _ in range(n_features)]
        for edge in synthetic_edges:
            i, j = edge
            syn_incident[i].add(edge)
            syn_incident[j].add(edge)

        for feature_idx in range(n_features):
            real_here = real_incident[feature_idx]
            syn_here = syn_incident[feature_idx]
            preserved = len(real_here & syn_here)
            lost = len(real_here - syn_here)
            new = len(syn_here - real_here)
            denominator = preserved + lost + new + epsilon
            rows.append({
                "feature_index": feature_idx,
                "method": method,
                "preserved_edges": preserved,
                "lost_edges": lost,
                "synthetic_only_edges": new,
                "rewiring_score": lost + new,
                "preservation_score": preserved / denominator,
                "real_degree": len(real_here),
                "synthetic_degree": len(syn_here),
            })
    return pd.DataFrame(rows)


def summarize_feature_preservation(feature_scores, method_order=None):
    """Return winner annotations per feature and a compact method-level summary."""
    method_order = list(method_order or feature_scores["method"].drop_duplicates())
    rows = []
    for feature_idx, group in feature_scores.groupby("feature_index", sort=True):
        ranked = group.copy()
        ranked["method_rank"] = ranked["method"].map({method: i for i, method in enumerate(method_order)})
        ranked = ranked.sort_values(
            ["preservation_score", "rewiring_score", "method_rank"],
            ascending=[False, True, True],
        )
        best = ranked.iloc[0]
        second_score = float(ranked.iloc[1]["preservation_score"]) if len(ranked) > 1 else float(best["preservation_score"])
        rows.append({
            "feature_index": int(feature_idx),
            "best_method": best["method"],
            "best_preservation_score": float(best["preservation_score"]),
            "second_best_preservation_score": second_score,
            "confidence_margin": float(best["preservation_score"] - second_score),
            "best_rewiring_score": float(best["rewiring_score"]),
            "best_lost_edges": float(best["lost_edges"]),
            "best_synthetic_only_edges": float(best["synthetic_only_edges"]),
            "real_degree": float(best["real_degree"]),
        })
    winners = pd.DataFrame(rows)

    counts = winners["best_method"].value_counts().reindex(method_order, fill_value=0)
    means = (
        feature_scores.groupby("method", as_index=True)
        .agg(
            mean_preservation_score=("preservation_score", "mean"),
            mean_rewiring_score=("rewiring_score", "mean"),
        )
        .reindex(method_order)
    )
    summary = means.assign(n_features_best=counts).reset_index().rename(columns={"index": "method"})
    summary = summary[["method", "n_features_best", "mean_preservation_score", "mean_rewiring_score"]]
    return winners, summary


def summarize_feature_loss(feature_scores, method_order=None):
    """Return the method with the largest lost-real-edge burden for each feature."""
    method_order = list(method_order or feature_scores["method"].drop_duplicates())
    rows = []
    for feature_idx, group in feature_scores.groupby("feature_index", sort=True):
        ranked = group.copy()
        ranked["method_rank"] = ranked["method"].map({method: i for i, method in enumerate(method_order)})
        ranked = ranked.sort_values(
            ["lost_edges", "rewiring_score", "preservation_score", "method_rank"],
            ascending=[False, False, True, True],
        )
        worst = ranked.iloc[0]
        rows.append({
            "feature_index": int(feature_idx),
            "lost_method": worst["method"],
            "lost_edges": float(worst["lost_edges"]),
            "rewiring_score": float(worst["rewiring_score"]),
            "preservation_score": float(worst["preservation_score"]),
            "real_degree": float(worst["real_degree"]),
        })
    lost = pd.DataFrame(rows)
    counts = lost["lost_method"].value_counts().reindex(method_order, fill_value=0)
    means = (
        feature_scores.groupby("method", as_index=True)
        .agg(
            mean_lost_edges=("lost_edges", "mean"),
            mean_rewiring_score=("rewiring_score", "mean"),
        )
        .reindex(method_order)
    )
    summary = means.assign(n_features_most_lost=counts).reset_index().rename(columns={"index": "method"})
    return lost, summary[["method", "n_features_most_lost", "mean_lost_edges", "mean_rewiring_score"]]


def summarize_feature_synthetic_only(feature_scores, method_order=None):
    """Return the method with the largest synthetic-only edge burden for each feature."""
    method_order = list(method_order or feature_scores["method"].drop_duplicates())
    rows = []
    for feature_idx, group in feature_scores.groupby("feature_index", sort=True):
        ranked = group.copy()
        ranked["method_rank"] = ranked["method"].map({method: i for i, method in enumerate(method_order)})
        ranked = ranked.sort_values(
            ["synthetic_only_edges", "rewiring_score", "preservation_score", "method_rank"],
            ascending=[False, False, True, True],
        )
        worst = ranked.iloc[0]
        rows.append({
            "feature_index": int(feature_idx),
            "synthetic_only_method": worst["method"],
            "synthetic_only_edges": float(worst["synthetic_only_edges"]),
            "rewiring_score": float(worst["rewiring_score"]),
            "preservation_score": float(worst["preservation_score"]),
            "real_degree": float(worst["real_degree"]),
        })
    synthetic_only = pd.DataFrame(rows)
    counts = synthetic_only["synthetic_only_method"].value_counts().reindex(method_order, fill_value=0)
    means = (
        feature_scores.groupby("method", as_index=True)
        .agg(
            mean_synthetic_only_edges=("synthetic_only_edges", "mean"),
            mean_rewiring_score=("rewiring_score", "mean"),
        )
        .reindex(method_order)
    )
    summary = means.assign(n_features_most_synthetic_only=counts).reset_index().rename(columns={"index": "method"})
    return synthetic_only, summary[["method", "n_features_most_synthetic_only", "mean_synthetic_only_edges", "mean_rewiring_score"]]


def _ellipse_params(points, pad=1.35, min_radius=0.85):
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    if points.shape[0] == 1:
        return center, min_radius * 2.0, min_radius * 2.0, 0.0
    if points.shape[0] == 2:
        diff = points[1] - points[0]
        norm = np.linalg.norm(diff)
        if norm < 1e-9:
            return center, min_radius * 2.0, min_radius * 2.0, 0.0
        angle = np.degrees(np.arctan2(diff[1], diff[0]))
        return center, max(norm * pad, min_radius * 2.0), min_radius * 2.0, float(angle)
    cov = np.cov(points, rowvar=False)
    if not np.all(np.isfinite(cov)):
        spread = np.ptp(points, axis=0)
        return center, max(float(spread[0]) * pad, min_radius * 2.0), max(float(spread[1]) * pad, min_radius * 2.0), 0.0
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, 0.0, None)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 4.0 * np.sqrt(values) * pad
    return center, max(float(width), min_radius * 2.0), max(float(height), min_radius * 2.0), float(angle)


def _local_tsne_neighborhoods(coords, distance_scale=1.85, min_groups=3, max_groups=12):
    coords = np.asarray(coords, dtype=float)
    if len(coords) <= 1:
        return [np.arange(len(coords), dtype=int)] if len(coords) else []
    distances = squareform(pdist(coords, metric="euclidean"))
    nearest = distances.copy()
    np.fill_diagonal(nearest, np.inf)
    nearest_dist = np.min(nearest, axis=1)
    threshold = float(np.nanmedian(nearest_dist[np.isfinite(nearest_dist)]) * distance_scale)
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = float(np.nanmedian(distances[distances > 0])) if np.any(distances > 0) else 1.0

    while True:
        pending = set(range(len(coords)))
        groups = []
        while pending:
            start = pending.pop()
            component = {start}
            queue = [start]
            while queue:
                current = queue.pop()
                neighbors = [other for other in list(pending) if distances[current, other] <= threshold]
                for other in neighbors:
                    pending.remove(other)
                    component.add(other)
                    queue.append(other)
            groups.append(np.asarray(sorted(component), dtype=int))
        if len(groups) >= min_groups or threshold <= 0:
            break
        threshold *= 0.72

    if len(groups) > max_groups:
        try:
            labels = KMeans(n_clusters=max_groups, random_state=123, n_init=30).fit_predict(coords)
            groups = [np.where(labels == label)[0] for label in range(max_groups)]
        except Exception:
            groups = sorted(groups, key=len, reverse=True)[:max_groups]
    return sorted(groups, key=lambda idx: (float(coords[idx, 0].mean()), float(coords[idx, 1].mean())))


def _cluster_method_summary(feature_scores, clusters, method_order, mode, feature_names=None, top_features=2):
    method_order = list(method_order)
    rank = {method: i for i, method in enumerate(method_order)}
    feature_names = list(feature_names) if feature_names is not None else None
    if mode == "preserve":
        assignment_sort = ["preservation_score", "rewiring_score", "method_rank"]
        assignment_ascending = [False, True, True]
    elif mode == "lost":
        assignment_sort = ["lost_edges", "rewiring_score", "preservation_score", "method_rank"]
        assignment_ascending = [False, False, True, True]
    else:
        assignment_sort = ["synthetic_only_edges", "rewiring_score", "preservation_score", "method_rank"]
        assignment_ascending = [False, False, True, True]
    rows = []
    for cluster_id, idx in enumerate(clusters, start=1):
        cluster_scores = feature_scores[feature_scores["feature_index"].isin(idx)].copy()
        cluster_features = (
            cluster_scores.drop_duplicates("feature_index")
            .sort_values(["real_degree", "feature_index"], ascending=[False, True])
            .head(max(0, int(top_features)))
        )
        top_feature_indices = cluster_features["feature_index"].to_numpy(dtype=int)
        if feature_names is None:
            top_feature_names = [str(i) for i in top_feature_indices]
        else:
            top_feature_names = [str(feature_names[i]) for i in top_feature_indices]
        feature_assignments = cluster_scores.copy()
        feature_assignments["method_rank"] = feature_assignments["method"].map(rank)
        feature_assignments = (
            feature_assignments
            .sort_values(assignment_sort, ascending=assignment_ascending)
            .drop_duplicates("feature_index", keep="first")
        )
        feature_assignment_counts = (
            feature_assignments["method"]
            .value_counts()
            .reindex(method_order, fill_value=0)
            .rename_axis("method")
            .reset_index(name="n_features_matching_method")
        )
        if mode == "preserve":
            method_scores = (
                cluster_scores.groupby("method", as_index=False)
                .agg(
                    preserved_edges=("preserved_edges", "sum"),
                    rewiring_score=("rewiring_score", "sum"),
                    preservation_score=("preservation_score", "mean"),
                )
            )
            method_scores["method_rank"] = method_scores["method"].map(rank)
            method_scores = method_scores.sort_values(
                ["preserved_edges", "preservation_score", "rewiring_score", "method_rank"],
                ascending=[False, False, True, True],
            )
            method_scores = method_scores.merge(feature_assignment_counts, on="method", how="left")
            method_scores = method_scores.sort_values(
                ["n_features_matching_method", "preserved_edges", "preservation_score", "rewiring_score", "method_rank"],
                ascending=[False, False, False, True, True],
            )
            winner = method_scores.iloc[0]
            score_columns = {
                "preserved_edges": float(winner["preserved_edges"]),
                "mean_preservation_score": float(winner["preservation_score"]),
                "rewiring_score": float(winner["rewiring_score"]),
            }
        elif mode == "lost":
            method_scores = (
                cluster_scores.groupby("method", as_index=False)
                .agg(
                    lost_edges=("lost_edges", "sum"),
                    rewiring_score=("rewiring_score", "sum"),
                    preservation_score=("preservation_score", "mean"),
                )
            )
            method_scores["method_rank"] = method_scores["method"].map(rank)
            method_scores = method_scores.sort_values(
                ["lost_edges", "rewiring_score", "preservation_score", "method_rank"],
                ascending=[False, False, True, True],
            )
            method_scores = method_scores.merge(feature_assignment_counts, on="method", how="left")
            method_scores = method_scores.sort_values(
                ["n_features_matching_method", "lost_edges", "rewiring_score", "preservation_score", "method_rank"],
                ascending=[False, False, False, True, True],
            )
            winner = method_scores.iloc[0]
            score_columns = {
                "lost_edges": float(winner["lost_edges"]),
                "rewiring_score": float(winner["rewiring_score"]),
                "mean_preservation_score": float(winner["preservation_score"]),
            }
        else:
            method_scores = (
                cluster_scores.groupby("method", as_index=False)
                .agg(
                    synthetic_only_edges=("synthetic_only_edges", "sum"),
                    rewiring_score=("rewiring_score", "sum"),
                    preservation_score=("preservation_score", "mean"),
                )
            )
            method_scores["method_rank"] = method_scores["method"].map(rank)
            method_scores = method_scores.sort_values(
                ["synthetic_only_edges", "rewiring_score", "preservation_score", "method_rank"],
                ascending=[False, False, True, True],
            )
            method_scores = method_scores.merge(feature_assignment_counts, on="method", how="left")
            method_scores = method_scores.sort_values(
                ["n_features_matching_method", "synthetic_only_edges", "rewiring_score", "preservation_score", "method_rank"],
                ascending=[False, False, False, True, True],
            )
            winner = method_scores.iloc[0]
            score_columns = {
                "synthetic_only_edges": float(winner["synthetic_only_edges"]),
                "rewiring_score": float(winner["rewiring_score"]),
                "mean_preservation_score": float(winner["preservation_score"]),
            }
        rows.append({
            "cluster_id": int(cluster_id),
            "method": winner["method"],
            "n_features": int(len(idx)),
            "n_features_matching_method": int(winner["n_features_matching_method"]),
            "prominent_feature_indices": top_feature_indices,
            "prominent_features": ", ".join(top_feature_names),
            "feature_indices": np.asarray(idx, dtype=int),
            **score_columns,
        })
    return pd.DataFrame(rows)


def _blob_polygon(points, pad=0.52, n_angles=96):
    points = np.asarray(points, dtype=float)
    center = points.mean(axis=0)
    if len(points) <= 2:
        ellipse_center, width, height, angle = _ellipse_params(points, pad=1.55, min_radius=0.72)
        theta = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        unit = np.column_stack([np.cos(theta) * width / 2.0, np.sin(theta) * height / 2.0])
        rotation = np.deg2rad(angle)
        rot = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
        return ellipse_center + unit @ rot.T

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    centered = points - center
    radii = np.linalg.norm(centered, axis=1)
    base_pad = max(float(np.median(radii)) * 0.33, pad)
    vertices = []
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        perpendicular = np.array([-np.sin(angle), np.cos(angle)])
        forward = centered @ direction
        side = np.abs(centered @ perpendicular)
        nearby = side <= max(base_pad * 1.35, np.percentile(side, 55))
        radius = float(np.max(forward[nearby])) if np.any(nearby) else float(np.max(forward))
        vertices.append(center + direction * (max(radius, 0.0) + base_pad))
    return np.asarray(vertices, dtype=float)


def _cluster_blob_geometry(coords, clusters):
    geometry = []
    for cluster_id, idx in enumerate(clusters, start=1):
        points = coords[idx]
        geometry.append({
            "cluster_id": int(cluster_id),
            "feature_indices": np.asarray(idx, dtype=int),
            "center": points.mean(axis=0),
            "blob": _blob_polygon(points),
        })
    return geometry


def _draw_neighborhood_blobs(ax, blob_geometry, summary, palette, show_feature_names=False):
    group_rows = []
    summary_by_cluster = summary.set_index("cluster_id")
    all_centers = np.asarray([geom["center"] for geom in blob_geometry], dtype=float)
    layout_center = all_centers.mean(axis=0) if len(all_centers) else np.zeros(2)
    layout_span = np.ptp(all_centers, axis=0).max() if len(all_centers) > 1 else 1.0
    label_offset = max(float(layout_span) * 0.035, 0.18)
    placed_label_boxes = []

    def _estimate_label_box(label, xy):
        lines = [line for line in str(label).split("\n") if line]
        max_chars = max((len(line) for line in lines), default=1)
        width = max(float(layout_span) * 0.0105 * max_chars, 0.42)
        height = max(float(layout_span) * 0.040 * max(len(lines), 1), 0.28)
        x, y = float(xy[0]), float(xy[1])
        return (x - width / 2.0, x + width / 2.0, y - height / 2.0, y + height / 2.0)

    def _overlap_area(box_a, box_b):
        x_overlap = max(0.0, min(box_a[1], box_b[1]) - max(box_a[0], box_b[0]))
        y_overlap = max(0.0, min(box_a[3], box_b[3]) - max(box_a[2], box_b[2]))
        return x_overlap * y_overlap

    def _label_position(center, blob, cluster_id, label):
        center = np.asarray(center, dtype=float)
        radius = float(np.max(np.linalg.norm(np.asarray(blob) - center, axis=1))) if len(blob) else 0.0
        outward = center - layout_center
        norm = float(np.linalg.norm(outward))
        if norm <= 1e-9:
            angle = (cluster_id - 1) * (2.0 * np.pi / max(len(blob_geometry), 1))
            outward = np.asarray([np.cos(angle), np.sin(angle)])
        else:
            outward = outward / norm

        base_directions = [
            outward,
            np.asarray([1.0, 0.0]),
            np.asarray([-1.0, 0.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, -1.0]),
            np.asarray([1.0, 1.0]) / np.sqrt(2.0),
            np.asarray([-1.0, 1.0]) / np.sqrt(2.0),
            np.asarray([1.0, -1.0]) / np.sqrt(2.0),
            np.asarray([-1.0, -1.0]) / np.sqrt(2.0),
        ]
        candidates = []
        for distance_scale in (1.0, 1.25, 1.55):
            distance = radius + label_offset * distance_scale
            for direction_rank, direction in enumerate(base_directions):
                direction = np.asarray(direction, dtype=float)
                direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
                xy = center + direction * distance
                box = _estimate_label_box(label, xy)
                overlap = sum(_overlap_area(box, placed) for placed in placed_label_boxes)
                away_penalty = max(0.0, 1.0 - float(np.dot(direction, outward)))
                cost = overlap * 1000.0 + distance_scale * 0.8 + direction_rank * 0.05 + away_penalty
                candidates.append((cost, xy, box, direction))
        return min(candidates, key=lambda item: item[0])

    for geom in blob_geometry:
        cluster_id = int(geom["cluster_id"])
        idx = geom["feature_indices"]
        row = summary_by_cluster.loc[cluster_id]
        method = row["method"]
        color = palette.get(method, "#888888")
        center = geom["center"]
        blob = geom["blob"]
        ax.add_patch(Polygon(
            blob,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            alpha=0.16,
            zorder=0,
        ))
        ax.add_patch(Polygon(
            blob,
            closed=True,
            facecolor="none",
            edgecolor=color,
            linewidth=1.2,
            alpha=0.82,
            zorder=2,
        ))
        prominent_features = str(row.get("prominent_features", "")).strip()
        if show_feature_names and prominent_features:
            label = prominent_features.replace(", ", "\n")
            _, label_xy, label_box, direction = _label_position(center, blob, cluster_id, label)
            placed_label_boxes.append(label_box)
            ax.update_datalim(label_xy.reshape(1, 2))
            ax.autoscale_view()
            ax.annotate(
                label,
                xy=center,
                xytext=label_xy,
                ha="center",
                va="center",
                fontsize=7.1,
                weight="semibold",
                color=color,
                zorder=6,
                linespacing=1.03,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.05),
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    linewidth=1.0,
                    alpha=0.78,
                    shrinkA=1.0,
                    shrinkB=3.0,
                ),
            )
        else:
            support = int(row.get("n_features_matching_method", len(idx)))
            label = f"{method}\n{support}/{len(idx)} features"
            ax.text(
                center[0],
                center[1],
                label,
                ha="center",
                va="center",
                fontsize=7.1,
                weight="semibold",
                color=color,
                zorder=5,
                linespacing=1.03,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=1.25),
            )
        extra = {
            key: value
            for key, value in row.to_dict().items()
            if key not in {"method", "n_features", "feature_indices"}
        }
        group_rows.append({
            "cluster_id": int(cluster_id),
            "method": method,
            "n_features": int(len(idx)),
            "feature_indices": idx,
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            **extra,
        })
    return pd.DataFrame(group_rows)


def _clusters_from_labels(labels):
    labels = np.asarray(labels, dtype=int)
    return [np.where(labels == cluster_id)[0] for cluster_id in sorted(set(labels))]


def _cluster_color_map(cluster_ids):
    cluster_ids = sorted(set(np.asarray(cluster_ids, dtype=int)))
    cluster_palette = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(cluster_ids), 1)))
    return {cluster: cluster_palette[i] for i, cluster in enumerate(cluster_ids)}


def _cluster_dot_colors(clusters, cluster_labels):
    cluster_color = _cluster_color_map(cluster_labels)
    colors = np.empty(sum(len(idx) for idx in clusters), dtype=object)
    for idx in clusters:
        label = int(cluster_labels[idx[0]]) if len(idx) else 0
        colors[idx] = [cluster_color[label]] * len(idx)
    return colors


def _draw_cluster_dots(ax, coords, clusters, cluster_labels):
    colors = _cluster_dot_colors(clusters, cluster_labels)
    for idx, (x, y) in enumerate(coords):
        ax.scatter(
            x,
            y,
            s=42,
            color=colors[idx],
            edgecolor="#252525",
            linewidth=0.52,
            alpha=0.94,
            zorder=3,
        )


def _draw_feature_preservation_tsne_panel(
    ax,
    coords,
    real_edges,
    real_partial,
    feature_names,
    winners,
    feature_scores,
    clusters,
    blob_geometry,
    cluster_labels,
    method_order,
    palette,
    draw_backbone=False,
    cluster_feature_label_top=0,
):
    winners = winners.sort_values("feature_index")
    cluster_summary = _cluster_method_summary(
        feature_scores,
        clusters,
        method_order,
        mode="preserve",
        feature_names=feature_names,
        top_features=max(2, cluster_feature_label_top),
    )
    group_summary = _draw_neighborhood_blobs(
        ax,
        blob_geometry,
        cluster_summary,
        palette,
        show_feature_names=cluster_feature_label_top > 0,
    )

    if draw_backbone and real_edges:
        ranked_edges = sorted(
            real_edges,
            key=lambda edge: abs(float(real_partial[edge[0], edge[1]])),
            reverse=True,
        )
        for i, j in ranked_edges[: max(12, min(80, len(ranked_edges) // 5))]:
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color="#1B1B1B",
                linewidth=0.35,
                alpha=0.10,
                zorder=1,
            )

    _draw_cluster_dots(ax, coords, clusters, cluster_labels)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#333333")
    return group_summary


def _draw_lost_tsne_panel(
    ax,
    coords,
    lost,
    feature_scores,
    clusters,
    blob_geometry,
    cluster_labels,
    method_order,
    palette,
    feature_names=None,
    cluster_feature_label_top=0,
):
    lost = lost.sort_values("feature_index")
    cluster_summary = _cluster_method_summary(
        feature_scores,
        clusters,
        method_order,
        mode="lost",
        feature_names=feature_names,
        top_features=max(2, cluster_feature_label_top),
    )
    group_summary = _draw_neighborhood_blobs(
        ax,
        blob_geometry,
        cluster_summary,
        palette,
        show_feature_names=cluster_feature_label_top > 0,
    )

    _draw_cluster_dots(ax, coords, clusters, cluster_labels)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#333333")
    return group_summary


def _draw_synthetic_only_tsne_panel(
    ax,
    coords,
    feature_scores,
    clusters,
    blob_geometry,
    cluster_labels,
    method_order,
    palette,
    feature_names=None,
    cluster_feature_label_top=0,
):
    cluster_summary = _cluster_method_summary(
        feature_scores,
        clusters,
        method_order,
        mode="synthetic_only",
        feature_names=feature_names,
        top_features=max(2, cluster_feature_label_top),
    )
    group_summary = _draw_neighborhood_blobs(
        ax,
        blob_geometry,
        cluster_summary,
        palette,
        show_feature_names=cluster_feature_label_top > 0,
    )

    _draw_cluster_dots(ax, coords, clusters, cluster_labels)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#333333")
    return group_summary



def _profile_clusters(profiles, max_clusters=7):
    n_features = profiles.shape[0]
    if n_features <= 2:
        return np.ones(n_features, dtype=int)
    n_clusters = int(min(max_clusters, max(3, round(np.sqrt(n_features) * 1.35))))
    n_clusters = min(n_clusters, max(2, n_features // 2))
    profiles = np.asarray(profiles, dtype=float)
    if not np.all(np.isfinite(profiles)) or np.allclose(profiles, profiles[0]):
        return np.ones(n_features, dtype=int)
    distances = pdist(profiles, metric="euclidean")
    if not np.all(np.isfinite(distances)) or np.allclose(distances, 0):
        return np.ones(n_features, dtype=int)
    try:
        linkage = hierarchy.linkage(distances, method="average")
        return hierarchy.fcluster(linkage, t=n_clusters, criterion="maxclust")
    except Exception:
        labels = KMeans(n_clusters=n_clusters, random_state=123, n_init=30).fit_predict(profiles)
        return labels.astype(int) + 1


def _draw_glasso_tsne_panel(
    ax,
    coords,
    clusters,
    real_partial,
    real_edges,
    synthetic_partial,
    synthetic_edges,
    feature_names,
    title,
    label_top=10,
    show_xlabel=True,
):
    clusters = np.asarray(clusters, dtype=int)
    cluster_color = _cluster_color_map(clusters)

    categories = {
        "preserved": real_edges & synthetic_edges,
        "real_only": real_edges - synthetic_edges,
        "synthetic_only": synthetic_edges - real_edges,
    }
    edge_partials = {
        "preserved": real_partial,
        "real_only": real_partial,
        "synthetic_only": synthetic_partial,
    }

    for category in ("preserved", "real_only", "synthetic_only"):
        partial = edge_partials[category]
        for edge in categories[category]:
            i, j = edge
            weight = abs(float(partial[i, j]))
            width = 0.35 + 2.2 * min(weight, 0.8)
            alpha = 0.26 if category == "preserved" else 0.18
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color=EDGE_COLORS[category],
                linewidth=width,
                alpha=alpha,
                zorder=1,
            )

    degree = np.zeros(coords.shape[0], dtype=float)
    for i, j in real_edges:
        degree[i] += 1
        degree[j] += 1
    sizes = 30 + 18 * np.sqrt(degree + 1)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=sizes,
        c=[cluster_color[int(cluster)] for cluster in clusters],
        edgecolor="#1F1F1F",
        linewidth=0.55,
        alpha=0.94,
        zorder=3,
    )

    label_nodes = np.argsort(-(degree + np.sum(np.abs(real_partial), axis=1)))[:label_top]
    for node in label_nodes:
        ax.text(
            coords[node, 0],
            coords[node, 1],
            _short_label(feature_names[node], 14),
            fontsize=6.5,
            ha="center",
            va="center",
            color="#111111",
            zorder=4,
        )

    ax.set_title(title, fontsize=10.2, weight="semibold", pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1" if show_xlabel else "", fontsize=8.2, labelpad=2)
    ax.set_ylabel("t-SNE 2", fontsize=8.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#333333")


def plot_glasso_tsne_layout(
    real_data,
    synthetic_data,
    feature_names,
    alphas=None,
    dataset_order=None,
    method_order=None,
    exemplar_ds="HIV",
    threshold=1e-7,
    seed=123,
    max_clusters=7,
    label_top=0,
    panels=None,
    save_path=None,
):
    """Plot t-SNE of Graphical Lasso feature-dependency profiles with edge overlays."""
    method_order = list(method_order or synthetic_data[exemplar_ds].keys())
    dataset_order = list(dataset_order or real_data.keys())
    panels = list(panels or ["A", "B", "C", "D"])
    structures, metrics = _fit_structures(
        real_data,
        synthetic_data,
        alphas=alphas,
        threshold=threshold,
        dataset_order=dataset_order,
        method_order=method_order,
    )

    names = list(feature_names[exemplar_ds] if isinstance(feature_names, Mapping) else feature_names)
    real = structures[exemplar_ds]["real"]
    real_partial = real["partial"]
    real_edges = real["edges"]
    coords, profiles, perplexity = _fit_profile_tsne(real_partial, seed=seed)
    clusters = _profile_clusters(profiles, max_clusters=max_clusters)

    fig = plt.figure(figsize=(11.4, 11.4), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.16)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    for ax, method, panel in zip(axes, method_order, panels):
        syn = structures[exemplar_ds]["synthetic"][method]
        _draw_glasso_tsne_panel(
            ax,
            coords,
            clusters,
            real_partial,
            real_edges,
            syn["partial"],
            syn["edges"],
            names,
            f"{panel}. {method} vs Real",
            label_top=label_top,
        )

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="#1F1F1F", markersize=7, label="Feature"),
        Line2D([0], [0], color=EDGE_COLORS["preserved"], lw=3, label="Preserved edge"),
        Line2D([0], [0], color=EDGE_COLORS["real_only"], lw=3, label="Real-only / lost"),
        Line2D([0], [0], color=EDGE_COLORS["synthetic_only"], lw=3, label="Synthetic-only edge"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.950),
        ncol=4,
        frameon=False,
        fontsize=8.1,
    )
    fig.text(
        0.5,
        0.965,
        f"{exemplar_ds}: t-SNE layout of Graphical Lasso partial-correlation profiles (perplexity={perplexity:.0f})",
        ha="center",
        va="top",
        fontsize=12.2,
        weight="semibold",
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.900, bottom=0.050)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return Figure4Result(
        fig=fig,
        metrics=metrics,
        anchor=-1,
        anchor_feature="",
        structures=structures,
    )


def plot_figure4_tsne_edge_supplement(
    real_data,
    synthetic_data,
    feature_names,
    alphas=None,
    dataset_order=None,
    method_order=None,
    exemplar_ds="HIV",
    threshold=1e-7,
    seed=123,
    max_clusters=7,
    label_top=0,
    save_path=None,
):
    """Supplemental E-H t-SNE edge-overlay figure for one dataset."""
    return plot_glasso_tsne_layout(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_names,
        alphas=alphas,
        dataset_order=dataset_order,
        method_order=method_order,
        exemplar_ds=exemplar_ds,
        threshold=threshold,
        seed=seed,
        max_clusters=max_clusters,
        label_top=label_top,
        panels=["E", "F", "G", "H"],
        save_path=save_path,
    )


def plot_figure4_cluster_summary_grid(
    real_data,
    synthetic_data,
    feature_names,
    alphas=None,
    dataset_order=None,
    method_order=None,
    threshold=1e-7,
    seed=123,
    max_clusters=7,
    palette=None,
    cluster_feature_label_top=0,
    save_path=None,
):
    """Main-text 3x3 cluster-summary grid across datasets."""
    method_order = list(method_order or synthetic_data[next(iter(synthetic_data))].keys())
    dataset_order = list(dataset_order or real_data.keys())
    palette = dict(METHOD_PRESERVATION_COLORS if palette is None else palette)
    structures, metrics = _fit_structures(
        real_data,
        synthetic_data,
        alphas=alphas,
        threshold=threshold,
        dataset_order=dataset_order,
        method_order=method_order,
    )

    fig = plt.figure(
        figsize=(14.4, 4.35 * len(dataset_order) + 1.15),
        constrained_layout=False,
    )
    gs = fig.add_gridspec(
        len(dataset_order) + 1,
        4,
        width_ratios=[0.34, 1.0, 1.0, 1.0],
        height_ratios=[0.18] + [1.0] * len(dataset_order),
        wspace=0.08,
        hspace=0.20,
    )

    corner_ax = fig.add_subplot(gs[0, 0])
    corner_ax.axis("off")

    column_headers = ["Preserved", "Lost", "Synthetic-only"]
    for col, header in enumerate(column_headers, start=1):
        header_ax = fig.add_subplot(gs[0, col])
        header_ax.axis("off")
        header_ax.text(
            0.5,
            0.5,
            header,
            ha="center",
            va="center",
            fontsize=13.2,
            weight="heavy",
            color="#222222",
            transform=header_ax.transAxes,
        )

    plot_axes = np.empty((len(dataset_order), 3), dtype=object)
    for row, dataset in enumerate(dataset_order, start=1):
        label_ax = fig.add_subplot(gs[row, 0])
        label_ax.axis("off")
        label_ax.text(
            0.5,
            0.5,
            dataset,
            ha="center",
            va="center",
            rotation=90,
            fontsize=13.8,
            weight="heavy",
            color="#222222",
            transform=label_ax.transAxes,
        )
        for col in range(3):
            plot_axes[row - 1, col] = fig.add_subplot(gs[row, col + 1])

    panel_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    panel_idx = 0
    group_summaries = []

    for row, dataset in enumerate(dataset_order):
        names = list(feature_names[dataset] if isinstance(feature_names, Mapping) else feature_names)
        real = structures[dataset]["real"]
        real_partial = real["partial"]
        real_edges = real["edges"]
        synthetic_edge_map = {
            method: structures[dataset]["synthetic"][method]["edges"]
            for method in method_order
        }
        feature_scores = build_feature_preservation_scores(
            real_edges,
            synthetic_edge_map,
            real_partial.shape[0],
        )
        winners, preservation_summary = summarize_feature_preservation(feature_scores, method_order=method_order)
        lost, lost_summary = summarize_feature_loss(feature_scores, method_order=method_order)
        synthetic_only, synthetic_only_summary = summarize_feature_synthetic_only(feature_scores, method_order=method_order)
        coords, profiles, perplexity = _fit_profile_tsne(real_partial, seed=seed)
        cluster_labels = _profile_clusters(profiles, max_clusters=max_clusters)
        clusters = _clusters_from_labels(cluster_labels)
        blob_geometry = _cluster_blob_geometry(coords, clusters)

        preserve_group_summary = _draw_feature_preservation_tsne_panel(
            plot_axes[row, 0],
            coords,
            real_edges,
            real_partial,
            names,
            winners,
            feature_scores,
            clusters,
            blob_geometry,
            cluster_labels,
            method_order,
            palette,
            draw_backbone=False,
            cluster_feature_label_top=cluster_feature_label_top,
        )
        lost_group_summary = _draw_lost_tsne_panel(
            plot_axes[row, 1],
            coords,
            lost,
            feature_scores,
            clusters,
            blob_geometry,
            cluster_labels,
            method_order,
            palette,
            feature_names=names,
            cluster_feature_label_top=cluster_feature_label_top,
        )
        synthetic_only_group_summary = _draw_synthetic_only_tsne_panel(
            plot_axes[row, 2],
            coords,
            feature_scores,
            clusters,
            blob_geometry,
            cluster_labels,
            method_order,
            palette,
            feature_names=names,
            cluster_feature_label_top=cluster_feature_label_top,
        )
        summaries = [
            (preserve_group_summary, "preserve"),
            (lost_group_summary, "lost"),
            (synthetic_only_group_summary, "synthetic-only"),
        ]
        for col, (summary, mode) in enumerate(summaries):
            panel = panel_letters[panel_idx]
            plot_axes[row, col].set_title(
                panel,
                loc="left",
                fontsize=10.6,
                weight="semibold",
                pad=7,
            )
            group_summaries.append(summary.assign(dataset=dataset, panel=panel, mode=mode, perplexity=perplexity))
            panel_idx += 1

    handles = [
        Patch(facecolor=palette.get(method, "#888888"), edgecolor=palette.get(method, "#888888"), alpha=0.28, label=method)
        for method in method_order
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.86),
        ncol=len(method_order),
        frameon=False,
        fontsize=8.8,
        handlelength=1.5,
        handletextpad=0.45,
        columnspacing=1.15,
    )
    # fig.text(
    #     0.5,
    #     0.988,
    #     "Cluster-level feature summaries on Graphical Lasso t-SNE layouts",
    #     ha="center",
    #     va="top",
    #     fontsize=13.2,
    #     weight="semibold",
    # )
    fig.subplots_adjust(left=0.045, right=0.992, top=0.895, bottom=0.035)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    result = Figure4Result(
        fig=fig,
        metrics=metrics,
        anchor=-1,
        anchor_feature="",
        structures=structures,
    )
    result.group_summary = pd.concat(group_summaries, ignore_index=True)
    result.neighborhood_summary = result.group_summary
    result.preservation_summary = preservation_summary
    result.lost_summary = lost_summary
    result.synthetic_only_summary = synthetic_only_summary
    return result


