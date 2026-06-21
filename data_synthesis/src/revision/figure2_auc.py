"""Figure 2: PCA examples and real-vs-synthetic separability AUC."""

import contextlib
import io

from src.revision.common import *
from src.revision.stats import one_run_origin_auc

def plot_figure2_cvae_pca(datasets, seed=SEED, cvae_epochs=CVAE_EPOCHS):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.85))
    for ax, ds, panel in zip(axes, DATASET_ORDER, ["A", "B", "C"]):
        data = datasets[ds]
        X_real = np.asarray(data["X"], dtype=np.float32)
        X_syn, _ = sample_synthetic(ds, data, "CVAE", seed=seed, cvae_epochs=cvae_epochs)
        Xr, Xs = standardize_pair(X_real, X_syn)
        pca = PCA(n_components=2, random_state=seed).fit(Xr)
        Zr = pca.transform(Xr)
        Zs = pca.transform(Xs)
        rng = np.random.default_rng(seed)
        Zs_plot = Zs[rng.choice(len(Zs), size=700, replace=False)] if len(Zs) > 700 else Zs

        point_size = 10
        real_color = "#8A8A8A"
        syn_color = DATASET_COLORS[ds]
        ax.scatter(Zr[:, 0], Zr[:, 1], s=point_size, marker="o", facecolors="none", alpha=0.62,
                   edgecolors=real_color, linewidths=0.65, label="Real data")
        ax.scatter(Zs_plot[:, 0], Zs_plot[:, 1], s=point_size, marker="o", color=syn_color, alpha=0.78,
                   edgecolors="none", label="CVAE synthetic data")
        add_confidence_ellipse(ax, Zr, real_color, linestyle="-", linewidth=2.2)
        add_confidence_ellipse(ax, Zs, syn_color, linestyle="-", linewidth=2.4)

        all_z = np.vstack([Zr, Zs_plot])
        x_min, x_max = np.nanmin(all_z[:, 0]), np.nanmax(all_z[:, 0])
        y_min, y_max = np.nanmin(all_z[:, 1]), np.nanmax(all_z[:, 1])
        x_pad = max((x_max - x_min) * 0.24, 1e-6)
        y_pad = max((y_max - y_min) * 0.24, 1e-6)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ev = pca.explained_variance_ratio_
        ax.set_title(ds, color=DATASET_COLORS[ds], weight="bold", pad=8, fontsize=12.0)
        ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)", color="black", fontsize=10.0)
        ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)", color="black", fontsize=10.0, labelpad=6)
        ax.text(0.045, 0.055, f"n = {len(data['y'])}, p = {X_real.shape[1]}", transform=ax.transAxes,
                color=DATASET_COLORS[ds], fontsize=9.0, weight="bold", ha="left", va="bottom")
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#BDBDBD",
                  framealpha=0.92, fontsize=7.8, handlelength=1.2, borderpad=0.35,
                  labelspacing=0.25, handletextpad=0.35)
        ax.tick_params(axis="both", colors="black", direction="out", width=1.2, length=4, labelsize=8.8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.25)
        add_panel_label(ax, panel)

    fig.subplots_adjust(left=0.070, right=0.985, top=0.84, bottom=0.20, wspace=0.25)

    for left_ax, right_ax in zip(axes[:-1], axes[1:]):
        left_box = left_ax.get_position()
        right_box = right_ax.get_position()
        x_sep = (left_box.x1 + right_box.x0) / 2
        fig.add_artist(mpl.lines.Line2D([x_sep, x_sep], [0.20, 0.80], transform=fig.transFigure,
                                        color="#D8D8D8", linewidth=0.8, alpha=0.75))
    plt.show()

def compute_auc_run_table(datasets, seed=SEED, repeats=AUC_REPEATS, cvae_epochs=CVAE_EPOCHS):
    rows = []
    for ds in DATASET_ORDER:
        data = datasets[ds]
        X_real = np.asarray(data["X"], dtype=np.float32)
        y_real = np.asarray(data["y"], dtype=int)
        for method in METHOD_ORDER:
            print(f"[AUC runs] {ds} - {method}")
            X_syn, y_syn = sample_synthetic(ds, data, method, seed=seed, cvae_epochs=cvae_epochs)
            for r in range(repeats):
                auc = one_run_origin_auc(X_real, y_real, X_syn, y_syn, seed=seed + 1000*r + 17*METHOD_ORDER.index(method))
                rows.append({"dataset": ds, "method": method, "run": r, "separability_auc": auc})
    return pd.DataFrame(rows)

def _plot_pca_panel(ax, ds, panel, seed=SEED, cvae_epochs=CVAE_EPOCHS):
    data = require_datasets()[ds]
    X_real = np.asarray(data["X"], dtype=np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        X_syn, _ = sample_synthetic(ds, data, "CVAE", seed=seed, cvae_epochs=cvae_epochs)
    Xr, Xs = standardize_pair(X_real, X_syn)
    pca = PCA(n_components=2, random_state=seed).fit(Xr)
    Zr = pca.transform(Xr)
    Zs = pca.transform(Xs)
    rng = np.random.default_rng(seed)
    Zs_plot = Zs[rng.choice(len(Zs), size=700, replace=False)] if len(Zs) > 700 else Zs

    ax.scatter(Zr[:, 0], Zr[:, 1], s=8, marker="o", facecolors="none", alpha=0.58,
               edgecolors="#8A8A8A", linewidths=0.55, label="Real data")
    ax.scatter(Zs_plot[:, 0], Zs_plot[:, 1], s=8, marker="o", color=DATASET_COLORS[ds], alpha=0.74,
               edgecolors="none", label="CVAE synthetic data")
    add_confidence_ellipse(ax, Zr, "#8A8A8A", linewidth=1.8)
    add_confidence_ellipse(ax, Zs, DATASET_COLORS[ds], linewidth=2.0)

    all_z = np.vstack([Zr, Zs_plot])
    x_min, x_max = np.nanmin(all_z[:, 0]), np.nanmax(all_z[:, 0])
    y_min, y_max = np.nanmin(all_z[:, 1]), np.nanmax(all_z[:, 1])
    ax.set_xlim(x_min - (x_max - x_min) * 0.24, x_max + (x_max - x_min) * 0.24)
    ax.set_ylim(y_min - (y_max - y_min) * 0.24, y_max + (y_max - y_min) * 0.24)

    ev = pca.explained_variance_ratio_
    ax.set_title(ds, color=DATASET_COLORS[ds], weight="semibold", pad=8)
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
    ax.text(0.045, 0.055, f"n={len(data['y'])}, p={X_real.shape[1]}", transform=ax.transAxes,
            color=DATASET_COLORS[ds], fontsize=8.5, weight="bold", ha="left", va="bottom")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#BDBDBD",
              framealpha=0.92, fontsize=7.8, handlelength=1.2, borderpad=0.35,
              labelspacing=0.25, handletextpad=0.35)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    ax.tick_params(labelsize=8.5, width=1.2, length=4)
    add_panel_label(ax, panel)

def _plot_auc_violin_panel(ax, auc_runs, ds, panel):
    sub = auc_runs[auc_runs["dataset"] == ds]
    positions = np.arange(len(METHOD_ORDER))
    values = [sub[sub["method"] == method]["separability_auc"].dropna().to_numpy() for method in METHOD_ORDER]

    violins = ax.violinplot(values, positions=positions, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body, method in zip(violins["bodies"], METHOD_ORDER):
        body.set_facecolor(METHOD_PASTELS[method])
        body.set_edgecolor(METHOD_COLORS[method])
        body.set_linewidth(1.45)
        body.set_alpha(0.95)

    for pos, vals, method in zip(positions, values, METHOD_ORDER):
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        mean = np.mean(vals)
        ax.vlines(pos, q1, q3, color=METHOD_COLORS[method], linewidth=3.0, alpha=0.92)
        ax.scatter(pos, med, s=28, color="white", edgecolor=METHOD_COLORS[method], linewidth=1.3, zorder=4)
        ax.scatter(pos, mean, s=22, color=METHOD_COLORS[method], edgecolor="white", linewidth=0.7, zorder=5)

    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.2)
    ax.set_xticks(positions)
    ax.set_xticklabels(METHOD_ORDER,fontsize=8.5)
    ax.set_ylabel(r"$\langle \mathrm{AUC} \rangle$")
    # Match PCA-panel style
    clean_axis(ax, grid_axis="y")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    ax.tick_params(labelsize=8.5, width=1.2, length=4)
    add_panel_label(ax, panel)

def _plot_auc_line_panel(ax, auc_runs, ds, panel):
    sub = auc_runs[auc_runs["dataset"] == ds]
    means = sub.groupby("method")["separability_auc"].mean().reindex(METHOD_ORDER)
    sds = sub.groupby("method")["separability_auc"].std().reindex(METHOD_ORDER)
    x = np.arange(len(METHOD_ORDER))
    for i, method in enumerate(METHOD_ORDER):
        ax.errorbar(i, means.loc[method], yerr=sds.loc[method], marker="o", markersize=7,
                    color=METHOD_COLORS[method], capsize=4, linewidth=2.0)
    ax.plot(x, means.to_numpy(), color=DATASET_COLORS[ds], linewidth=2.0, alpha=0.65)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=25, ha="right", fontsize=8.5)
    ax.set_title(rf"{ds}: $\langle \mathrm{{AUC}} \rangle$ +/- SD", color=DATASET_COLORS[ds], weight="bold", pad=8)
    ax.set_ylabel(r"$\langle \mathrm{AUC} \rangle$")
    clean_axis(ax, grid_axis="y")
    add_panel_label(ax, panel)

def plot_figure2_six_panel(auc_runs=None):
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))

    for ax, ds, panel in zip(axes, DATASET_ORDER, ["A", "B", "C"]):
        _plot_pca_panel(ax, ds, panel)

    fig.subplots_adjust(left=0.065, right=0.99, top=0.88, bottom=0.18, wspace=0.28)
    return fig

def plot_figure2_single_pca(dataset, seed=SEED, cvae_epochs=CVAE_EPOCHS):
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.4))
    _plot_pca_panel(ax, dataset, "", seed=seed, cvae_epochs=cvae_epochs)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.16)
    return fig

def plot_figure2_nine_panel(auc_runs):
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 11.4))
    panels = list("ABCDEFGHI")
    for ax, ds, panel in zip(axes[0], DATASET_ORDER, panels[:3]):
        _plot_pca_panel(ax, ds, panel)
    for ax, ds, panel in zip(axes[1], DATASET_ORDER, panels[3:6]):
        _plot_auc_violin_panel(ax, auc_runs, ds, panel)
    for ax, ds, panel in zip(axes[2], DATASET_ORDER, panels[6:]):
        _plot_auc_line_panel(ax, auc_runs, ds, panel)
    # fig.suptitle("Figure 2. PCA geometry, separability distributions, and summary trends", y=0.99, fontsize=15, weight="semibold")
    fig.subplots_adjust(left=0.065, right=0.99, top=0.94, bottom=0.08, wspace=0.30, hspace=0.50)
    return fig
