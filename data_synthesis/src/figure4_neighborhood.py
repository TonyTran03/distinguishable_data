from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Polygon
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


def _draw_radial_labels(ax, graph, pos, feature_names, anchor):
    ax.text(
        pos[anchor][0],
        pos[anchor][1],
        _short_label(feature_names[anchor], 13),
        fontsize=6.9,
        ha="center",
        va="center",
        weight="semibold",
        zorder=5,
    )
    for node in graph.nodes:
        if node == anchor:
            continue
        x, y = pos[node]
        angle = np.degrees(np.arctan2(y, x))
        rotation = angle
        ha = "left"
        if angle > 90 or angle < -90:
            rotation = angle + 180
            ha = "right"
        lx, ly = np.array([x, y]) * 1.17
        ax.text(
            lx,
            ly,
            _short_label(feature_names[node], 17),
            fontsize=6.2,
            rotation=rotation,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            color="#202020",
        )


def _anchor_layout(nodes, anchor):
    neighbors = sorted([n for n in nodes if n != anchor])
    pos = {anchor: np.array([0.0, 0.0])}
    if not neighbors:
        return pos
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(neighbors), endpoint=False)
    for node, angle in zip(neighbors, angles):
        pos[node] = np.array([np.cos(angle), np.sin(angle)])
    return pos


def plot_overlap_neighborhood(
    ax,
    real_partial,
    synthetic_partial,
    real_edges,
    synthetic_edges,
    anchor,
    feature_names,
    title,
    fixed_nodes=None,
    fixed_pos=None,
    max_neighbors=None,
    prefer="preserved",
):
    """Plot preserved, lost, and invented conditional-dependency edges around one anchor."""
    categories = get_anchor_neighborhood_edges(real_edges, synthetic_edges, anchor)
    categories = _filter_anchor_neighborhood(
        categories, real_partial, synthetic_partial, max_neighbors=max_neighbors, prefer=prefer
    )
    nodes = set(fixed_nodes or {anchor})
    for edge in categories["all"]:
        nodes.update(edge)
    nodes.add(anchor)
    pos = fixed_pos or _anchor_layout(nodes, anchor)

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(categories["all"])

    node_sizes = [610 if n == anchor else 185 for n in graph.nodes]
    node_colors = ["#F7C948" if n == anchor else "#F7F7F7" for n in graph.nodes]
    nx.draw_networkx_nodes(
        graph, pos, nodelist=list(graph.nodes), node_size=node_sizes, node_color=node_colors,
        edgecolors="#303030", linewidths=1.0, ax=ax
    )

    for category in ("preserved", "real_only", "synthetic_only"):
        for edge in sorted(categories[category]):
            width = 1.0 + 6.0 * _edge_weight(edge, real_partial, synthetic_partial, category)
            style = "solid" if _edge_sign(edge, real_partial, synthetic_partial, category) >= 0 else "dashed"
            nx.draw_networkx_edges(
                graph, pos, edgelist=[edge], width=width, edge_color=EDGE_COLORS[category],
                style=style, alpha=0.92, ax=ax
            )

    _draw_radial_labels(ax, graph, pos, feature_names, anchor)

    ax.set_title(title, fontsize=10.5, weight="semibold", pad=8)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    return categories


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


def plot_edge_overlap_matrix(ax, edge_recovery, method_order, title="C. Recovery of real conditional edges"):
    matrix = edge_recovery[method_order].to_numpy(dtype=float)
    cmap = ListedColormap(["#F4F6F8", "#174A7C"])
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11.5, weight="semibold", pad=24)
    ax.set_xticks(np.arange(len(method_order)))
    ax.set_xticklabels(method_order, fontsize=8.8)
    ax.set_yticks(np.arange(len(edge_recovery)))
    ax.set_yticklabels(edge_recovery["rank"], fontsize=6.4)
    ax.set_ylabel("Real HIV edge rank", fontsize=8.2)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(method_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(edge_recovery), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.0,
        1.015,
        "Top real conditional edges by |partial correlation|; edge names are listed in the returned table",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        color="#444444",
    )

    strength_ax = ax.inset_axes([-0.065, 0.0, 0.018, 1.0], transform=ax.transAxes)
    strength = edge_recovery["real_abs_partial_corr"].to_numpy(dtype=float)[:, None]
    strength_ax.imshow(strength, aspect="auto", interpolation="nearest", cmap="Blues")
    strength_ax.set_xticks([])
    strength_ax.set_yticks([])
    strength_ax.set_title(r"$|\rho_{partial}|$", fontsize=6.8, pad=4)
    for spine in strength_ax.spines.values():
        spine.set_visible(False)

    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=8, markerfacecolor="#174A7C",
               markeredgecolor="#174A7C", label="Recovered"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=8, markerfacecolor="#F4F6F8",
               markeredgecolor="#C8CDD2", label="Lost"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.08), frameon=False, fontsize=8.0, ncol=2)
    return ax


def plot_figure4_edge_overlap_matrix(
    real_data,
    synthetic_data,
    feature_names,
    alphas=None,
    dataset_order=None,
    method_order=None,
    exemplar_ds="HIV",
    threshold=1e-7,
    top_n_edges=35,
    save_path=None,
):
    method_order = list(method_order or synthetic_data[exemplar_ds].keys())
    dataset_order = list(dataset_order or real_data.keys())
    structures, metrics = _fit_structures(
        real_data, synthetic_data, alphas=alphas, threshold=threshold,
        dataset_order=dataset_order, method_order=method_order
    )

    names = list(feature_names[exemplar_ds] if isinstance(feature_names, Mapping) else feature_names)
    real = structures[exemplar_ds]["real"]
    real_edges = real["edges"]
    real_partial = real["partial"]
    synthetic_edge_map = {
        method: structures[exemplar_ds]["synthetic"][method]["edges"]
        for method in method_order
    }

    hiv_metrics = metrics[metrics["dataset"] == exemplar_ds].set_index("method")
    preserved_method = hiv_metrics["edge_recovery"].idxmax()
    lost_method = "Column-wise" if "Column-wise" in method_order else hiv_metrics["edge_recovery"].idxmin()
    preserved_anchor = choose_example_anchor(
        real_edges, synthetic_edge_map[preserved_method], real_partial.shape[0], mode="preserved"
    )
    lost_anchor = choose_example_anchor(
        real_edges, synthetic_edge_map[lost_method], real_partial.shape[0], mode="lost"
    )
    edge_recovery = build_edge_recovery_matrix(
        real_partial, real_edges, synthetic_edge_map, names, top_n=top_n_edges
    )

    fig = plt.figure(figsize=(13.8, 8.9), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.55], hspace=0.55, wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    syn_preserved = structures[exemplar_ds]["synthetic"][preserved_method]
    plot_overlap_neighborhood(
        ax_a,
        real_partial,
        syn_preserved["partial"],
        real_edges,
        syn_preserved["edges"],
        preserved_anchor,
        names,
        "A. Example of preserved conditional links",
        max_neighbors=14,
        prefer="preserved",
    )
    ax_a.text(
        0.5, -0.08, f"{preserved_method} vs Real; anchor: {names[preserved_anchor]}",
        transform=ax_a.transAxes, ha="center", va="top", fontsize=7.8, color="#444444"
    )

    syn_lost = structures[exemplar_ds]["synthetic"][lost_method]
    plot_overlap_neighborhood(
        ax_b,
        real_partial,
        syn_lost["partial"],
        real_edges,
        syn_lost["edges"],
        lost_anchor,
        names,
        "B. Example of lost conditional links",
        max_neighbors=14,
        prefer="lost",
    )
    ax_b.text(
        0.5, -0.08, f"{lost_method} vs Real; anchor: {names[lost_anchor]}",
        transform=ax_b.transAxes, ha="center", va="top", fontsize=7.8, color="#444444"
    )

    ax_c = fig.add_subplot(gs[1, 0:4])
    plot_edge_overlap_matrix(ax_c, edge_recovery, method_order)

    handles = [
        Line2D([0], [0], color=EDGE_COLORS["preserved"], lw=3, label="Preserved local edge"),
        Line2D([0], [0], color=EDGE_COLORS["real_only"], lw=3, label="Real-only local edge"),
        Line2D([0], [0], color=EDGE_COLORS["synthetic_only"], lw=3, label="Synthetic-only local edge"),
        Line2D([0], [0], color="#333333", lw=2, linestyle="solid", label="Positive partial correlation"),
        Line2D([0], [0], color="#333333", lw=2, linestyle="dashed", label="Negative partial correlation"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=5, frameon=False, fontsize=8.3)
    fig.suptitle(
        "Real conditional-dependency preservation and synthetic structural deviation",
        y=0.985,
        fontsize=15.5,
        weight="semibold",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.835, bottom=0.075)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    anchor = preserved_anchor
    return Figure4Result(
        fig=fig,
        metrics=metrics,
        anchor=anchor,
        anchor_feature=names[anchor],
        structures=structures,
        edge_recovery=edge_recovery,
    )


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


def _add_axes_titles(fig, axes, titles, fontsize=10.6, pad=0.004):
    for ax, title in zip(axes, titles):
        box = ax.get_position()
        fig.text(
            (box.x0 + box.x1) / 2,
            box.y1 + pad,
            title,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            weight="semibold",
        )


def plot_edge_status_matrix(
    ax,
    status_matrix,
    order,
    title,
    subtitle=None,
    triangle=False,
    show_axes=True,
    show_xlabel=True,
):
    ordered = status_matrix[np.ix_(order, order)]
    cmap = ListedColormap([
        STATUS_COLORS["absent"],
        STATUS_COLORS["preserved"],
        STATUS_COLORS["real_only"],
        STATUS_COLORS["synthetic_only"],
    ])
    if triangle:
        cmap = cmap.copy()
        cmap.set_bad((1, 1, 1, 0))
        ordered = np.ma.array(ordered, mask=np.triu(np.ones_like(ordered, dtype=bool), k=0))

    ax.imshow(ordered, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest", aspect="equal")
    if title:
        ax.set_title(title, fontsize=10.6, weight="semibold", pad=4)
    if subtitle:
        ax.text(0.5, 1.006, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.2, color="#4B4B4B")

    n = len(order)
    tick_step = 1 if n <= 12 else 5 if n <= 35 else 10
    ticks = np.arange(0, n, tick_step)
    labels = [str(i + 1) for i in ticks]
    if show_axes:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, fontsize=7.0)
        ax.set_yticklabels(labels, fontsize=7.0)
        ax.set_xlabel("Feature index" if show_xlabel else "", fontsize=7.8, labelpad=1)
        ax.set_ylabel("Feature index", fontsize=7.8, labelpad=1)
        ax.tick_params(length=2.0, width=0.8, pad=1.5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#555555")
    return ax


def plot_supplemental_edge_status_matrices(
    real_data,
    synthetic_data,
    feature_names,
    alphas=None,
    dataset_order=None,
    method_order=None,
    exemplar_ds="Breast Cancer",
    comparison_methods=None,
    threshold=1e-7,
    save_path=None,
):
    """Compact 2x2 version of the Figure 4 edge-status matrices for supplements."""
    method_order = list(method_order or synthetic_data[exemplar_ds].keys())
    dataset_order = list(dataset_order or real_data.keys())
    comparison_methods = list(comparison_methods or ["Bootstrap", "Column-wise", "CVAE", "GMM"])
    comparison_methods = [m for m in comparison_methods if m in method_order][:4]
    if len(comparison_methods) < 4:
        comparison_methods.extend([m for m in method_order if m not in comparison_methods])
    comparison_methods = comparison_methods[:4]

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

    fig = plt.figure(figsize=(8.7, 7.6), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        4,
        hspace=0.24,
        wspace=0.04,
    )
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]
    panels = ["A", "B", "C", "D"]
    panel_titles = []
    for ax, panel, method in zip(axes, panels, comparison_methods):
        syn_edges = structures[exemplar_ds]["synthetic"][method]["edges"]
        status = build_edge_status_matrix(real_edges, syn_edges, real_partial.shape[0])
        panel_titles.append(f"{panel}. {method} vs Real")
        plot_edge_status_matrix(
            ax,
            status,
            order,
            None,
            triangle=False,
            show_axes=True,
            show_xlabel=(panel in {"C", "D"}),
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
        bbox_to_anchor=(0.5, 0.920),
        ncol=4,
        frameon=False,
        fontsize=8.0,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.05,
        borderaxespad=0.3,
    )
    fig.suptitle(
        f"Structural comparison matrices for {exemplar_ds}",
        y=0.975,
        fontsize=13.4,
        weight="semibold",
    )
    fig.subplots_adjust(left=0.045, right=0.990, top=0.850, bottom=0.040, wspace=0.04, hspace=0.24)
    _add_axes_titles(fig, axes, panel_titles)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return Figure4Result(
        fig=fig,
        metrics=metrics,
        anchor=-1,
        anchor_feature="",
        structures=structures,
        feature_index=feature_index,
    )


def plot_combined_edge_status_and_glasso_tsne(
    real_data,
    synthetic_data,
    feature_names,
    alphas=None,
    dataset_order=None,
    method_order=None,
    exemplar_ds="HIV",
    comparison_methods=None,
    threshold=1e-7,
    seed=123,
    max_clusters=7,
    label_top=0,
    save_path=None,
):
    """Combine edge-status matrices with matching Graphical Lasso t-SNE panels."""
    method_order = list(method_order or synthetic_data[exemplar_ds].keys())
    dataset_order = list(dataset_order or real_data.keys())
    comparison_methods = list(comparison_methods or ["Bootstrap", "Column-wise", "CVAE", "GMM"])
    comparison_methods = [m for m in comparison_methods if m in method_order][:4]
    if len(comparison_methods) < 4:
        comparison_methods.extend([m for m in method_order if m not in comparison_methods])
    comparison_methods = comparison_methods[:4]

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

    coords, profiles, perplexity = _fit_profile_tsne(real_partial, seed=seed)
    cluster_labels = _profile_clusters(profiles, max_clusters=max_clusters)
    clusters = _clusters_from_labels(cluster_labels)
    blob_geometry = _cluster_blob_geometry(coords, clusters)
    synthetic_edge_map = {
        method: structures[exemplar_ds]["synthetic"][method]["edges"]
        for method in comparison_methods
    }
    feature_scores = build_feature_preservation_scores(
        real_edges,
        synthetic_edge_map,
        real_partial.shape[0],
    )
    winners, preservation_summary = summarize_feature_preservation(feature_scores, method_order=comparison_methods)
    lost, lost_summary = summarize_feature_loss(feature_scores, method_order=comparison_methods)
    synthetic_only, synthetic_only_summary = summarize_feature_synthetic_only(feature_scores, method_order=comparison_methods)
    winners["feature_name"] = [names[i] for i in winners["feature_index"]]
    lost["feature_name"] = [names[i] for i in lost["feature_index"]]
    synthetic_only["feature_name"] = [names[i] for i in synthetic_only["feature_index"]]
    palette = dict(METHOD_PRESERVATION_COLORS)

    fig = plt.figure(figsize=(13.4, 17.6), constrained_layout=False)
    gs = fig.add_gridspec(
        7,
        6,
        height_ratios=[0.92, 0.92, 0.20, 1.0, 1.0, 0.16, 0.84],
        hspace=0.26,
        wspace=0.10,
    )
    matrix_axes = [
        fig.add_subplot(gs[0, 0:3]),
        fig.add_subplot(gs[0, 3:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]
    tsne_axes = [
        fig.add_subplot(gs[3, 0:3]),
        fig.add_subplot(gs[3, 3:6]),
        fig.add_subplot(gs[4, 0:3]),
        fig.add_subplot(gs[4, 3:6]),
    ]
    summary_axes = [
        fig.add_subplot(gs[6, 0:2]),
        fig.add_subplot(gs[6, 2:4]),
        fig.add_subplot(gs[6, 4:6]),
    ]

    matrix_titles = []
    for ax, panel, method in zip(matrix_axes, ["A", "B", "C", "D"], comparison_methods):
        syn_edges = structures[exemplar_ds]["synthetic"][method]["edges"]
        status = build_edge_status_matrix(real_edges, syn_edges, real_partial.shape[0])
        matrix_titles.append(f"{panel}. {method} vs Real")
        plot_edge_status_matrix(
            ax,
            status,
            order,
            None,
            triangle=False,
            show_axes=True,
            show_xlabel=(panel in {"C", "D"}),
        )

    for idx, (ax, panel, method) in enumerate(zip(tsne_axes, ["E", "F", "G", "H"], comparison_methods)):
        syn = structures[exemplar_ds]["synthetic"][method]
        _draw_glasso_tsne_panel(
            ax,
            coords,
            cluster_labels,
            real_partial,
            real_edges,
            syn["partial"],
            syn["edges"],
            names,
            f"{panel}. {method} vs Real",
            label_top=label_top,
            show_xlabel=idx >= 2,
        )

    preserve_group_summary = _draw_feature_preservation_tsne_panel(
        summary_axes[0],
        coords,
        real_edges,
        real_partial,
        names,
        winners,
        feature_scores,
        clusters,
        blob_geometry,
        cluster_labels,
        comparison_methods,
        palette,
        draw_backbone=False,
    )
    lost_group_summary = _draw_lost_tsne_panel(
        summary_axes[1],
        coords,
        lost,
        feature_scores,
        clusters,
        blob_geometry,
        cluster_labels,
        comparison_methods,
        palette,
    )
    synthetic_only_group_summary = _draw_synthetic_only_tsne_panel(
        summary_axes[2],
        coords,
        feature_scores,
        clusters,
        blob_geometry,
        cluster_labels,
        comparison_methods,
        palette,
    )
    for ax, panel, title in zip(summary_axes, ["I", "J", "K"], ["preserve", "lost", "synthetic-only"]):
        ax.set_title(f"{panel}. {title}", fontsize=11.0, weight="semibold", pad=7)

    status_handles = [
        Patch(facecolor=STATUS_COLORS["preserved"], edgecolor="#333333", label="Preserved edge"),
        Patch(facecolor=STATUS_COLORS["real_only"], edgecolor="#333333", label="Real-only / lost"),
        Patch(facecolor=STATUS_COLORS["synthetic_only"], edgecolor="#333333", label="Synthetic-only"),
        Patch(facecolor=STATUS_COLORS["absent"], edgecolor="#C9CDD2", label="Absent in both"),
    ]
    edge_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="#1F1F1F", markersize=7, label="Feature"),
        Line2D([0], [0], color=EDGE_COLORS["preserved"], lw=3, label="Preserved edge"),
        Line2D([0], [0], color=EDGE_COLORS["real_only"], lw=3, label="Real-only / lost"),
        Line2D([0], [0], color=EDGE_COLORS["synthetic_only"], lw=3, label="Synthetic-only edge"),
    ]
    fig.suptitle(
        f"{exemplar_ds}: real conditional-dependency preservation and t-SNE structural overlays",
        y=0.985,
        fontsize=14.6,
        weight="semibold",
    )
    fig.subplots_adjust(left=0.038, right=0.992, top=0.925, bottom=0.030)
    _add_axes_titles(fig, matrix_axes, matrix_titles)
    status_legend_y = matrix_axes[0].get_position().y1 + 0.66 * (0.985 - matrix_axes[0].get_position().y1)
    fig.legend(
        handles=status_handles,
        loc="center",
        bbox_to_anchor=(0.5, status_legend_y),
        ncol=4,
        frameon=False,
        fontsize=8.0,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.05,
        borderaxespad=0.3,
    )
    graph_gap_top = matrix_axes[2].get_position().y0
    graph_gap_bottom = tsne_axes[0].get_position().y1
    graph_title_y = graph_gap_bottom + 0.68 * (graph_gap_top - graph_gap_bottom)
    graph_legend_y = graph_gap_bottom + 0.28 * (graph_gap_top - graph_gap_bottom)
    summary_title_y = (tsne_axes[2].get_position().y0 + summary_axes[0].get_position().y1) / 2
    fig.text(
        0.5,
        graph_title_y,
        f"Graphical Lasso partial-correlation profile t-SNE (perplexity={perplexity:.0f})",
        ha="center",
        va="center",
        fontsize=11.4,
        weight="semibold",
    )
    fig.legend(
        handles=edge_handles,
        loc="center",
        bbox_to_anchor=(0.5, graph_legend_y),
        ncol=4,
        frameon=False,
        fontsize=7.8,
        handlelength=2.0,
        handletextpad=0.55,
        columnspacing=1.25,
    )
    fig.text(
        0.5,
        summary_title_y,
        "Cluster-level feature summaries on the same t-SNE layout",
        ha="center",
        va="center",
        fontsize=11.4,
        weight="semibold",
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    feature_plot_table = winners.merge(
        feature_scores.pivot(index="feature_index", columns="method", values="preservation_score")
        .add_prefix("preservation_score_")
        .reset_index(),
        on="feature_index",
        how="left",
    )

    result = Figure4Result(
        fig=fig,
        metrics=metrics,
        anchor=-1,
        anchor_feature="",
        structures=structures,
        edge_recovery=feature_scores,
        feature_index=feature_index,
    )
    result.preservation_summary = preservation_summary
    result.lost_summary = lost_summary
    result.synthetic_only_summary = synthetic_only_summary
    result.preserve_group_summary = preserve_group_summary
    result.lost_group_summary = lost_group_summary
    result.synthetic_only_group_summary = synthetic_only_group_summary
    result.neighborhood_summary = preserve_group_summary
    result.feature_preservation_index = feature_plot_table
    result.tsne_coordinates = pd.DataFrame({
        "feature_index": np.arange(real_partial.shape[0], dtype=int),
        "feature_name": names,
        "profile_cluster": cluster_labels,
        "preserve_method": winners["best_method"].to_numpy(dtype=object),
        "lost_method": lost["lost_method"].to_numpy(dtype=object),
        "synthetic_only_method": synthetic_only["synthetic_only_method"].to_numpy(dtype=object),
        "tSNE1": coords[:, 0],
        "tSNE2": coords[:, 1],
    })
    return result


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

    subtitles = {
        "Bootstrap": "many preserved real edges",
        "Column-wise": "substantial loss of real structure",
        "GMM": "mixed preservation and synthetic-only structure",
        "CVAE": "mixed preservation and synthetic-only structure",
    }

    fig = plt.figure(figsize=(8.7, 7.6), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        4,
        hspace=0.24,
        wspace=0.04,
    )

    matrix_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]
    panels = ["A", "B", "C", "D"]
    matrix_titles = []
    for ax, panel, method in zip(matrix_axes, panels, comparison_methods):
        syn_edges = structures[exemplar_ds]["synthetic"][method]["edges"]
        status = build_edge_status_matrix(real_edges, syn_edges, real_partial.shape[0])
        matrix_titles.append(f"{panel}. {method} vs Real")
        plot_edge_status_matrix(
            ax,
            status,
            order,
            None,
            subtitle=None,
            triangle=False,
            show_axes=True,
            show_xlabel=(panel in {"C", "D"}),
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
        bbox_to_anchor=(0.5, 0.920),
        ncol=4,
        frameon=False,
        fontsize=8.0,
        handlelength=1.4,
        handletextpad=0.45,
        columnspacing=1.05,
        borderaxespad=0.3,
    )

    # fig.suptitle(
    #     "Figure 4. Real conditional-dependency preservation and synthetic structural deviation",
    #     y=0.978,
    #     fontsize=15.5,
    #     weight="semibold",
    # )
    fig.subplots_adjust(left=0.045, right=0.990, top=0.850, bottom=0.040, wspace=0.04, hspace=0.24)
    _add_axes_titles(fig, matrix_axes, matrix_titles)

    return Figure4Result(
        fig=fig,
        metrics=metrics,
        anchor=-1,
        anchor_feature="",
        structures=structures,
        feature_index=feature_index,
    )


def plot_figure4_neighborhood_overlap(
    real_data,
    synthetic_data,
    feature_names,
    alphas=None,
    dataset_order=None,
    method_order=None,
    exemplar_ds="HIV",
    anchor_feature=None,
    threshold=1e-7,
    save_path=None,
):
    method_order = list(method_order or synthetic_data[exemplar_ds].keys())
    dataset_order = list(dataset_order or real_data.keys())
    structures, metrics = _fit_structures(
        real_data, synthetic_data, alphas=alphas, threshold=threshold,
        dataset_order=dataset_order, method_order=method_order
    )

    names = list(feature_names[exemplar_ds] if isinstance(feature_names, Mapping) else feature_names)
    real_partial = structures[exemplar_ds]["real"]["partial"]
    if anchor_feature is None:
        anchor = choose_anchor_feature(real_partial, names)
    else:
        anchor = names.index(anchor_feature)

    fixed_nodes = {anchor}
    real_edges = structures[exemplar_ds]["real"]["edges"]
    for method in method_order:
        syn_edges = structures[exemplar_ds]["synthetic"][method]["edges"]
        for edge in get_anchor_neighborhood_edges(real_edges, syn_edges, anchor)["all"]:
            fixed_nodes.update(edge)
    fixed_pos = _anchor_layout(fixed_nodes, anchor)

    fig = plt.figure(figsize=(13.8, 10.2), constrained_layout=False)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.18, 1.18], hspace=0.70, wspace=0.22)
    graph_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]

    panel_labels = ["A", "B", "C", "D"]
    for ax, method, panel in zip(graph_axes, method_order, panel_labels):
        syn = structures[exemplar_ds]["synthetic"][method]
        plot_overlap_neighborhood(
            ax,
            real_partial,
            syn["partial"],
            real_edges,
            syn["edges"],
            anchor,
            names,
            f"{panel}. {method} vs Real",
            fixed_nodes=fixed_nodes,
            fixed_pos=fixed_pos,
        )

    handles = [
        Line2D([0], [0], color=EDGE_COLORS["preserved"], lw=3, label="Preserved"),
        Line2D([0], [0], color=EDGE_COLORS["real_only"], lw=3, label="Real-only"),
        Line2D([0], [0], color=EDGE_COLORS["synthetic_only"], lw=3, label="Synthetic-only"),
        Line2D([0], [0], color="#333333", lw=2, linestyle="solid", label="Positive"),
        Line2D([0], [0], color="#333333", lw=2, linestyle="dashed", label="Negative"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=5, frameon=False, fontsize=8.6)
    fig.suptitle(
        "Figure 4. Local conditional-dependency neighborhoods reveal synthetic structural distortion",
        y=0.985,
        fontsize=15.5,
        weight="semibold",
    )
    fig.text(
        0.5,
        0.94,
        f"Anchor feature: {names[anchor]}",
        ha="center",
        va="center",
        fontsize=9.4,
        color="#333333",
    )
    fig.subplots_adjust(left=0.045, right=0.99, top=0.815, bottom=0.070)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return Figure4Result(fig=fig, metrics=metrics, anchor=anchor, anchor_feature=names[anchor], structures=structures)


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


def _cluster_method_summary(feature_scores, clusters, method_order, mode):
    method_order = list(method_order)
    rank = {method: i for i, method in enumerate(method_order)}
    rows = []
    for cluster_id, idx in enumerate(clusters, start=1):
        cluster_scores = feature_scores[feature_scores["feature_index"].isin(idx)].copy()
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


def _draw_neighborhood_blobs(ax, blob_geometry, summary, palette):
    group_rows = []
    summary_by_cluster = summary.set_index("cluster_id")
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
        ax.text(
            center[0],
            center[1],
            method,
            ha="center",
            va="center",
            fontsize=7.7,
            weight="semibold",
            color=color,
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2),
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
):
    winners = winners.sort_values("feature_index")
    cluster_summary = _cluster_method_summary(feature_scores, clusters, method_order, mode="preserve")
    group_summary = _draw_neighborhood_blobs(ax, blob_geometry, cluster_summary, palette)

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


def _draw_lost_tsne_panel(ax, coords, lost, feature_scores, clusters, blob_geometry, cluster_labels, method_order, palette):
    lost = lost.sort_values("feature_index")
    cluster_summary = _cluster_method_summary(feature_scores, clusters, method_order, mode="lost")
    group_summary = _draw_neighborhood_blobs(ax, blob_geometry, cluster_summary, palette)

    _draw_cluster_dots(ax, coords, clusters, cluster_labels)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#333333")
    return group_summary


def _draw_synthetic_only_tsne_panel(ax, coords, feature_scores, clusters, blob_geometry, cluster_labels, method_order, palette):
    cluster_summary = _cluster_method_summary(feature_scores, clusters, method_order, mode="synthetic_only")
    group_summary = _draw_neighborhood_blobs(ax, blob_geometry, cluster_summary, palette)

    _draw_cluster_dots(ax, coords, clusters, cluster_labels)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor("#333333")
    return group_summary


def plot_feature_preservation_tsne(
    real_data,
    synthetic_data,
    feature_names,
    alphas=None,
    dataset_order=None,
    method_order=None,
    exemplar_ds="HIV",
    threshold=1e-7,
    seed=123,
    palette=None,
    draw_backbone=False,
):
    """Plot feature-level structural preservation on a real Graphical Lasso t-SNE map."""
    method_order = list(method_order or synthetic_data[exemplar_ds].keys())
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

    names = list(feature_names[exemplar_ds] if isinstance(feature_names, Mapping) else feature_names)
    real = structures[exemplar_ds]["real"]
    real_partial = real["partial"]
    real_edges = real["edges"]
    synthetic_edge_map = {
        method: structures[exemplar_ds]["synthetic"][method]["edges"]
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
    winners["feature_name"] = [names[i] for i in winners["feature_index"]]
    lost["feature_name"] = [names[i] for i in lost["feature_index"]]
    synthetic_only["feature_name"] = [names[i] for i in synthetic_only["feature_index"]]

    coords, profiles, perplexity = _fit_profile_tsne(real_partial, seed=seed)
    cluster_labels = _profile_clusters(profiles, max_clusters=7)
    clusters = _clusters_from_labels(cluster_labels)
    blob_geometry = _cluster_blob_geometry(coords, clusters)
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.9), constrained_layout=False)
    preserve_group_summary = _draw_feature_preservation_tsne_panel(
        axes[0],
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
        draw_backbone=draw_backbone,
    )
    lost_group_summary = _draw_lost_tsne_panel(
        axes[1],
        coords,
        lost,
        feature_scores,
        clusters,
        blob_geometry,
        cluster_labels,
        method_order,
        palette,
    )
    synthetic_only_group_summary = _draw_synthetic_only_tsne_panel(
        axes[2],
        coords,
        feature_scores,
        clusters,
        blob_geometry,
        cluster_labels,
        method_order,
        palette,
    )
    axes[0].set_title("preserve", fontsize=11.0, weight="semibold", pad=7)
    axes[1].set_title("lost", fontsize=11.0, weight="semibold", pad=7)
    axes[2].set_title("synthetic-only", fontsize=11.0, weight="semibold", pad=7)
    fig.text(
        0.5,
        0.965,
        f"{exemplar_ds}: Graphical Lasso partial-correlation profile t-SNE clusters (perplexity={perplexity:.0f})",
        ha="center",
        va="top",
        fontsize=12.2,
        weight="semibold",
    )
    axes[0].text(
        0.012,
        0.018,
        f"perplexity={perplexity:.0f}",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#333333",
    )
    fig.subplots_adjust(left=0.045, right=0.990, top=0.875, bottom=0.065, wspace=0.085)

    feature_plot_table = winners.merge(
        feature_scores.pivot(index="feature_index", columns="method", values="preservation_score")
        .add_prefix("preservation_score_")
        .reset_index(),
        on="feature_index",
        how="left",
    )

    caption = (
        "Features are positioned using a t-SNE embedding of real Graphical Lasso "
        "partial-correlation profiles. Blobs use the same Graphical Lasso profile "
        "clusters and node colors as the t-SNE layout panels, with identical boundaries "
        "reused in preserve, lost, and synthetic-only. Blob color summarizes the "
        "winning synthetic method: best cluster preservation on the left, dominant "
        "lost/rewired signal in the middle, and dominant synthetic-only signal on "
        "the right."
    )

    result = Figure4Result(
        fig=fig,
        metrics=metrics,
        anchor=-1,
        anchor_feature="",
        structures=structures,
        edge_recovery=feature_scores,
        feature_index=feature_plot_table,
    )
    result.preservation_summary = preservation_summary
    result.lost_summary = lost_summary
    result.synthetic_only_summary = synthetic_only_summary
    result.preserve_group_summary = preserve_group_summary
    result.lost_group_summary = lost_group_summary
    result.synthetic_only_group_summary = synthetic_only_group_summary
    result.neighborhood_summary = preserve_group_summary
    result.caption = caption
    result.tsne_coordinates = pd.DataFrame({
        "feature_index": np.arange(real_partial.shape[0], dtype=int),
        "feature_name": names,
        "profile_cluster": cluster_labels,
        "preserve_method": winners["best_method"].to_numpy(dtype=object),
        "lost_method": lost["lost_method"].to_numpy(dtype=object),
        "synthetic_only_method": synthetic_only["synthetic_only_method"].to_numpy(dtype=object),
        "tSNE1": coords[:, 0],
        "tSNE2": coords[:, 1],
    })
    return result


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
):
    """Plot t-SNE of Graphical Lasso feature-dependency profiles with edge overlays."""
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
    real_partial = real["partial"]
    real_edges = real["edges"]
    coords, profiles, perplexity = _fit_profile_tsne(real_partial, seed=seed)
    clusters = _profile_clusters(profiles, max_clusters=max_clusters)

    fig = plt.figure(figsize=(11.4, 11.4), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.16)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    for ax, method, panel in zip(axes, method_order, ["A", "B", "C", "D"]):
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
    return Figure4Result(
        fig=fig,
        metrics=metrics,
        anchor=-1,
        anchor_feature="",
        structures=structures,
    )


def plot_glasso_tsne_layout_with_neighborhood_summary(
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
    palette=None,
    draw_backbone=False,
):
    """Plot edge overlays and cluster-level summaries in one dataset-specific figure."""
    method_order = list(method_order or synthetic_data[exemplar_ds].keys())
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

    names = list(feature_names[exemplar_ds] if isinstance(feature_names, Mapping) else feature_names)
    real = structures[exemplar_ds]["real"]
    real_partial = real["partial"]
    real_edges = real["edges"]

    coords, profiles, perplexity = _fit_profile_tsne(real_partial, seed=seed)
    cluster_labels = _profile_clusters(profiles, max_clusters=max_clusters)
    clusters = _clusters_from_labels(cluster_labels)
    blob_geometry = _cluster_blob_geometry(coords, clusters)

    synthetic_edge_map = {
        method: structures[exemplar_ds]["synthetic"][method]["edges"]
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
    winners["feature_name"] = [names[i] for i in winners["feature_index"]]
    lost["feature_name"] = [names[i] for i in lost["feature_index"]]
    synthetic_only["feature_name"] = [names[i] for i in synthetic_only["feature_index"]]

    fig = plt.figure(figsize=(16.2, 15.2), constrained_layout=False)
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 0.92], hspace=0.30, wspace=0.25)
    edge_axes = [
        fig.add_subplot(gs[0, 0:3]),
        fig.add_subplot(gs[0, 3:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]
    summary_axes = [
        fig.add_subplot(gs[2, 0:2]),
        fig.add_subplot(gs[2, 2:4]),
        fig.add_subplot(gs[2, 4:6]),
    ]

    for ax, method, panel in zip(edge_axes, method_order, ["D", "E", "F", "G"]):
        syn = structures[exemplar_ds]["synthetic"][method]
        _draw_glasso_tsne_panel(
            ax,
            coords,
            cluster_labels,
            real_partial,
            real_edges,
            syn["partial"],
            syn["edges"],
            names,
            f"{panel}. {method} vs Real",
            label_top=label_top,
        )

    preserve_group_summary = _draw_feature_preservation_tsne_panel(
        summary_axes[0],
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
        draw_backbone=draw_backbone,
    )
    lost_group_summary = _draw_lost_tsne_panel(
        summary_axes[1],
        coords,
        lost,
        feature_scores,
        clusters,
        blob_geometry,
        cluster_labels,
        method_order,
        palette,
    )
    synthetic_only_group_summary = _draw_synthetic_only_tsne_panel(
        summary_axes[2],
        coords,
        feature_scores,
        clusters,
        blob_geometry,
        cluster_labels,
        method_order,
        palette,
    )
    for ax, title in zip(summary_axes, ["preserve", "lost", "synthetic-only"]):
        ax.set_title(title, fontsize=11.0, weight="semibold", pad=7)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="#1F1F1F", markersize=7, label="Feature cluster"),
        Line2D([0], [0], color=EDGE_COLORS["preserved"], lw=3, label="Preserved edge"),
        Line2D([0], [0], color=EDGE_COLORS["real_only"], lw=3, label="Real-only / lost"),
        Line2D([0], [0], color=EDGE_COLORS["synthetic_only"], lw=3, label="Synthetic-only edge"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=4,
        frameon=False,
        fontsize=8.1,
    )
    fig.text(
        0.5,
        0.982,
        f"{exemplar_ds}: Graphical Lasso partial-correlation profile t-SNE and cluster summaries (perplexity={perplexity:.0f})",
        ha="center",
        va="top",
        fontsize=12.6,
        weight="semibold",
    )
    summary_axes[0].text(
        0.012,
        0.018,
        f"perplexity={perplexity:.0f}",
        transform=summary_axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#333333",
    )
    fig.subplots_adjust(left=0.045, right=0.990, top=0.910, bottom=0.050)

    feature_plot_table = winners.merge(
        feature_scores.pivot(index="feature_index", columns="method", values="preservation_score")
        .add_prefix("preservation_score_")
        .reset_index(),
        on="feature_index",
        how="left",
    )

    caption = (
        "Each dataset figure uses one t-SNE embedding of the real Graphical Lasso "
        "partial-correlation profiles. The upper panels show method-specific edge "
        "status on that layout. The lower panels reuse the exact same coordinates, "
        "hierarchical profile clusters, and blob boundaries; only each blob's "
        "method summary color changes for preserve, lost, and synthetic-only."
    )

    result = Figure4Result(
        fig=fig,
        metrics=metrics,
        anchor=-1,
        anchor_feature="",
        structures=structures,
        edge_recovery=feature_scores,
        feature_index=feature_plot_table,
    )
    result.preservation_summary = preservation_summary
    result.lost_summary = lost_summary
    result.synthetic_only_summary = synthetic_only_summary
    result.preserve_group_summary = preserve_group_summary
    result.lost_group_summary = lost_group_summary
    result.synthetic_only_group_summary = synthetic_only_group_summary
    result.neighborhood_summary = preserve_group_summary
    result.caption = caption
    result.tsne_coordinates = pd.DataFrame({
        "feature_index": np.arange(real_partial.shape[0], dtype=int),
        "feature_name": names,
        "profile_cluster": cluster_labels,
        "preserve_method": winners["best_method"].to_numpy(dtype=object),
        "lost_method": lost["lost_method"].to_numpy(dtype=object),
        "synthetic_only_method": synthetic_only["synthetic_only_method"].to_numpy(dtype=object),
        "tSNE1": coords[:, 0],
        "tSNE2": coords[:, 1],
    })
    return result
