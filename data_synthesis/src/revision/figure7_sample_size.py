"""Sample-size sensitivity for real-vs-synthetic origin AUC."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.revision.common import (
    Config,
    CVAE_EPOCHS,
    DATASET_COLORS,
    DATASET_ORDER,
    METHOD_COLORS,
    METHOD_ORDER,
    METHOD_PASTELS,
    SEED,
    clean_axis,
    sample_bootstrap,
    sample_columnwise,
    sample_gmm,
    sample_trained_cvae,
    train_cvae,
)
from src.revision.cache import _read_cache, _write_cache
from src.revision.stats import one_run_origin_auc, stratified_subsample


DEFAULT_FRACTIONS = (0.2, 0.3, 0.4, 0.5, 0.75, 1.0)


def _fraction_class_counts(y, fraction):
    y = np.asarray(y, dtype=int)
    n0 = max(2, int(np.floor(np.sum(y == 0) * float(fraction))))
    n1 = max(2, int(np.floor(np.sum(y == 1) * float(fraction))))
    return n0, n1


def _sample_from_method(method, X_sub, y_sub, n0, n1, seed, cvae_state=None):
    if method == "Bootstrap":
        return sample_bootstrap(X_sub, y_sub, n0, n1, seed=seed)
    if method == "Column-wise":
        return sample_columnwise(X_sub, y_sub, n0, n1, seed=seed)
    if method == "GMM":
        return sample_gmm(X_sub, y_sub, n0, n1, seed=seed)
    if method == "CVAE":
        if cvae_state is None:
            raise ValueError("CVAE state is required for CVAE sampling.")
        return sample_trained_cvae(cvae_state, n0, n1, seed=seed)
    raise ValueError(f"Unknown method: {method}")


def compute_sample_size_auc_table(
    datasets,
    fractions=DEFAULT_FRACTIONS,
    split_repeats=5,
    auc_repeats=5,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
):
    """Estimate origin AUC as generator training-set size changes.

    For each dataset/fraction/split, generators are trained on the same
    stratified real subset and asked to generate the same class counts. The
    origin probe then compares that real subset with the generated synthetic
    data using the same AUC routine as the main revision figures.
    """
    rows = []
    fractions = [float(frac) for frac in fractions]

    for dataset in DATASET_ORDER:
        data = datasets[dataset]
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=int)
        for fraction in fractions:
            n0, n1 = _fraction_class_counts(y, fraction)
            for split in range(int(split_repeats)):
                split_seed = seed + 10007 * split + int(round(fraction * 1000))
                X_sub, y_sub = stratified_subsample(X, y, n0, n1, seed=split_seed)

                cvae_state = None
                if "CVAE" in METHOD_ORDER:
                    print(f"[sample size] training CVAE: {dataset} fraction={fraction:g} split={split}")
                    with contextlib.redirect_stdout(io.StringIO()):
                        cvae_state = train_cvae(
                            X_sub,
                            y_sub,
                            cfg=Config(seed=split_seed, epochs=cvae_epochs, batch_size=32),
                            verbose=False,
                        )

                for method in METHOD_ORDER:
                    generation_seed = split_seed + 373 * (METHOD_ORDER.index(method) + 1)
                    print(f"[sample size] {dataset} {method} fraction={fraction:g} split={split}")
                    X_syn, y_syn = _sample_from_method(
                        method,
                        X_sub,
                        y_sub,
                        n0,
                        n1,
                        seed=generation_seed,
                        cvae_state=cvae_state,
                    )
                    for auc_repeat in range(int(auc_repeats)):
                        auc_seed = generation_seed + 1009 * auc_repeat
                        auc = one_run_origin_auc(
                            X_sub,
                            y_sub,
                            X_syn,
                            y_syn,
                            seed=auc_seed,
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "method": method,
                                "fraction": fraction,
                                "split": split,
                                "auc_repeat": auc_repeat,
                                "n0": n0,
                                "n1": n1,
                                "n_real": int(len(y_sub)),
                                "separability_auc": auc,
                            }
                        )
    return pd.DataFrame(rows)


def get_sample_size_auc(
    force=False,
    datasets=None,
    fractions=DEFAULT_FRACTIONS,
    split_repeats=5,
    auc_repeats=5,
    seed=SEED,
    cvae_epochs=CVAE_EPOCHS,
):
    cached = None if force else _read_cache("sample_size_auc")
    if cached is not None:
        return cached
    if datasets is None:
        from src.revision.common import require_datasets

        datasets = require_datasets()
    result = compute_sample_size_auc_table(
        datasets,
        fractions=fractions,
        split_repeats=split_repeats,
        auc_repeats=auc_repeats,
        seed=seed,
        cvae_epochs=cvae_epochs,
    )
    return _write_cache("sample_size_auc", result)


def summarize_sample_size_auc(sample_size_auc):
    return (
        sample_size_auc.groupby(["dataset", "method", "fraction"], as_index=False)
        .agg(
            auc_mean=("separability_auc", "mean"),
            auc_sd=("separability_auc", "std"),
            auc_q025=("separability_auc", lambda x: float(np.percentile(x, 2.5))),
            auc_q975=("separability_auc", lambda x: float(np.percentile(x, 97.5))),
            n_real=("n_real", "median"),
            runs=("separability_auc", "size"),
        )
        .sort_values(["dataset", "method", "fraction"])
    )


def _plot_dataset_sample_size_panel(ax, summary, dataset, panel=None):
    sub = summary[summary["dataset"] == dataset]
    for method in METHOD_ORDER:
        m = sub[sub["method"] == method].sort_values("fraction")
        if m.empty:
            continue
        x = m["fraction"].to_numpy(dtype=float) * 100.0
        y = m["auc_mean"].to_numpy(dtype=float)
        lo = m["auc_q025"].to_numpy(dtype=float)
        hi = m["auc_q975"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=METHOD_COLORS[method],
            marker="o",
            linewidth=2.2,
            markersize=5.0,
            label=method,
            zorder=3,
        )
        ax.fill_between(
            x,
            lo,
            hi,
            color=METHOD_COLORS[method],
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )

    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.15)
    ax.set_title(dataset, color=DATASET_COLORS[dataset], weight="bold", fontsize=11.5, pad=8)
    ax.set_xlabel("Real data used to train generator (%)")
    ax.set_ylim(0.45, 1.03)
    ax.set_xlim(100, 2)
    ax.set_xticks([100, 75, 50, 25, 2])
    ax.set_xticklabels(["100%", "75%", "50%", "25%", "2"])
    clean_axis(ax, grid_axis="y")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.tick_params(labelsize=8.5, width=1.2, length=4)
    if panel:
        ax.text(
            -0.12,
            1.08,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            weight="bold",
            color="#333434",
        )


def plot_sample_size_auc(sample_size_auc, dataset=None):
    """Plot sample-size sensitivity using revision figure styling."""
    summary = summarize_sample_size_auc(sample_size_auc)
    datasets = [dataset] if dataset is not None else DATASET_ORDER
    if dataset is not None:
        fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.2))
        _plot_dataset_sample_size_panel(ax, summary, dataset)
        ax.set_ylabel(r"$\langle \mathrm{AUC} \rangle$")
        handles = [
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=2.2,
                markersize=5.0,
                label=method,
            )
            for method in METHOD_ORDER
        ]
        ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="#BDBDBD", fontsize=8.2)
        fig.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.16)
        return fig, summary

    fig, axes = plt.subplots(1, len(datasets), figsize=(14.4, 4.0), sharey=True)
    for ax, ds in zip(np.ravel(axes), datasets):
        _plot_dataset_sample_size_panel(ax, summary, ds)
    np.ravel(axes)[0].set_ylabel(r"$\langle \mathrm{AUC} \rangle$")
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker="o",
            linewidth=2.2,
            markersize=5.0,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(METHOD_ORDER),
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0,
        borderpad=0.55,
        fontsize=8.8,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.84, bottom=0.25, wspace=0.22)
    return fig, summary
