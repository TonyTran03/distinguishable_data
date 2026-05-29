from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn import covariance
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
    palette = palette or {
        "Bootstrap": "#8FB8DE",
        "Column-wise": "#F0A35E",
        "GMM": "#9AC48A",
        "CVAE": "#C79BCB",
    }

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
        ax.set_xticklabels(dataset_order, rotation=20, ha="right", fontsize=8.0)
        ax.tick_params(axis="y", labelsize=8.0)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return axs


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

    fig = plt.figure(figsize=(13.8, 11.4), constrained_layout=False)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.55, 0.82], hspace=0.55, wspace=0.30)

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

    metric_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    plot_summary_metrics(metric_axes, metrics, dataset_order=dataset_order, method_order=method_order)
    metric_axes[0].text(-0.23, 1.18, "D", transform=metric_axes[0].transAxes, fontsize=15, weight="bold")
    metric_axes[1].text(
        0.5,
        1.24,
        "D. Global structural deviation across datasets",
        transform=metric_axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        weight="semibold",
    )
    metric_axes[-1].legend(loc="upper right", fontsize=7.5, frameon=True, edgecolor="#BBBBBB")

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


def plot_edge_status_matrix(ax, status_matrix, order, title, subtitle=None):
    ordered = status_matrix[np.ix_(order, order)]
    cmap = ListedColormap([
        STATUS_COLORS["absent"],
        STATUS_COLORS["preserved"],
        STATUS_COLORS["real_only"],
        STATUS_COLORS["synthetic_only"],
    ])
    ax.imshow(ordered, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=11.5, weight="semibold", pad=8)
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.2, color="#4B4B4B")

    n = len(order)
    tick_step = 1 if n <= 12 else 5 if n <= 35 else 10
    ticks = np.arange(0, n, tick_step)
    labels = [str(i + 1) for i in ticks]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlabel("Feature index", fontsize=8.0)
    ax.set_ylabel("Feature index", fontsize=8.0)
    ax.tick_params(length=2.5, width=0.8)
    for spine in ax.spines.values():
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

    fig = plt.figure(figsize=(10.8, 12.2), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.0, 1.0, 0.48],
        hspace=0.42,
        wspace=0.18,
    )
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]
    panels = ["A", "B", "C", "D"]
    for ax, panel, method in zip(axes, panels, comparison_methods):
        syn_edges = structures[exemplar_ds]["synthetic"][method]["edges"]
        status = build_edge_status_matrix(real_edges, syn_edges, real_partial.shape[0])
        plot_edge_status_matrix(ax, status, order, f"{panel}. {method} vs Real")

    metric_axes = [
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1:3]),
        fig.add_subplot(gs[2, 3]),
    ]
    plot_summary_metrics(metric_axes, metrics, dataset_order=dataset_order, method_order=method_order)
    metric_axes[0].text(-0.30, 1.16, "E", transform=metric_axes[0].transAxes, fontsize=14, weight="bold")
    metric_axes[1].text(
        0.5,
        1.24,
        "E. Global structural deviation across datasets",
        transform=metric_axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=10.8,
        weight="semibold",
    )
    metric_axes[-1].legend(loc="upper right", fontsize=7.2, frameon=True, edgecolor="#BBBBBB")

    legend_handles = [
        Patch(facecolor=STATUS_COLORS["preserved"], edgecolor="#333333", label="Preserved edge"),
        Patch(facecolor=STATUS_COLORS["real_only"], edgecolor="#333333", label="Real-only / lost"),
        Patch(facecolor=STATUS_COLORS["synthetic_only"], edgecolor="#333333", label="Synthetic-only"),
        Patch(facecolor=STATUS_COLORS["absent"], edgecolor="#C9CDD2", label="Absent in both"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=4,
        frameon=False,
        fontsize=8.6,
        handlelength=1.4,
        handletextpad=0.55,
        columnspacing=1.45,
        borderaxespad=0.8,
    )
    fig.suptitle(
        f"Structural comparison matrices for {exemplar_ds}",
        y=0.978,
        fontsize=14.5,
        weight="semibold",
    )
    fig.subplots_adjust(left=0.070, right=0.985, top=0.860, bottom=0.070, wspace=0.14, hspace=0.30)

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

    subtitles = {
        "Bootstrap": "many preserved real edges",
        "Column-wise": "substantial loss of real structure",
        "GMM": "mixed preservation and synthetic-only structure",
        "CVAE": "mixed preservation and synthetic-only structure",
    }

    fig = plt.figure(figsize=(10.8, 12.2), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        4,
        height_ratios=[1.0, 1.0, 0.48],
        hspace=0.55,
        wspace=0.18,
    )

    matrix_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]
    panels = ["A", "B", "C", "D"]
    for ax, panel, method in zip(matrix_axes, panels, comparison_methods):
        syn_edges = structures[exemplar_ds]["synthetic"][method]["edges"]
        status = build_edge_status_matrix(real_edges, syn_edges, real_partial.shape[0])
        plot_edge_status_matrix(
            ax,
            status,
            order,
            f"{panel}. {method} vs Real",
            subtitle=None,
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
        bbox_to_anchor=(0.5, 0.947),
        ncol=4,
        frameon=False,
        fontsize=8.8,
        handlelength=1.4,
        handletextpad=0.55,
        columnspacing=1.45,
        borderaxespad=0.8,
    )

    metric_axes = [
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1:3]),
        fig.add_subplot(gs[2, 3]),
    ]
    plot_summary_metrics(metric_axes, metrics, dataset_order=dataset_order, method_order=method_order)
    metric_axes[0].text(-0.28, 1.18, "E", transform=metric_axes[0].transAxes, fontsize=15, weight="bold")
    metric_axes[1].text(
        0.5,
        1.24,
        "E. Global structural deviation across datasets",
        transform=metric_axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        weight="semibold",
    )
    metric_axes[-1].legend(loc="upper right", fontsize=7.5, frameon=True, edgecolor="#BBBBBB")

    fig.suptitle(
        "Figure 4. Real conditional-dependency preservation and synthetic structural deviation",
        y=0.978,
        fontsize=15.5,
        weight="semibold",
    )
    fig.subplots_adjust(left=0.070, right=0.985, top=0.860, bottom=0.070, wspace=0.14, hspace=0.30)

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

    fig = plt.figure(figsize=(13.8, 13.0), constrained_layout=False)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.18, 1.18, 0.78], hspace=0.70, wspace=0.22)
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

    metric_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    plot_summary_metrics(metric_axes, metrics, dataset_order=dataset_order, method_order=method_order)
    metric_axes[0].text(-0.23, 1.16, "E", transform=metric_axes[0].transAxes, fontsize=15, weight="bold")
    handles = [
        Line2D([0], [0], color=EDGE_COLORS["preserved"], lw=3, label="Preserved"),
        Line2D([0], [0], color=EDGE_COLORS["real_only"], lw=3, label="Real-only"),
        Line2D([0], [0], color=EDGE_COLORS["synthetic_only"], lw=3, label="Synthetic-only"),
        Line2D([0], [0], color="#333333", lw=2, linestyle="solid", label="Positive"),
        Line2D([0], [0], color="#333333", lw=2, linestyle="dashed", label="Negative"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=5, frameon=False, fontsize=8.6)
    metric_axes[-1].legend(loc="upper right", fontsize=7.5, frameon=True, edgecolor="#BBBBBB")
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
