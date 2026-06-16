"""Figure 6: reverse-ablation and feature-removal sensitivity."""

from src.revision.common import *
from src.revision.stats import ablation_grid, one_run_origin_auc, rank_discriminating_features
from src.revision.cache import _read_cache, _write_cache

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

def plot_figure6_ablation_ac(ablation_df):
    """Compact Figure 6 panel: A-C show reverse ablation plus repeated-run boxplots."""
    fig = plt.figure(figsize=(13.4, 7.0), constrained_layout=False)
    gs = fig.add_gridspec(2, len(DATASET_ORDER), height_ratios=[1.35, 0.95])
    curve_axes = [fig.add_subplot(gs[0, i]) for i in range(len(DATASET_ORDER))]
    box_axes = [fig.add_subplot(gs[1, i], sharey=curve_axes[0]) for i in range(len(DATASET_ORDER))]

    legend_handles = []
    for panel_idx, (ax_curve, ax_box, ds) in enumerate(zip(curve_axes, box_axes, DATASET_ORDER)):
        sub = ablation_df[ablation_df["dataset"] == ds]
        box_values, box_labels = [], []
        if sub.empty:
            ax_curve.set_visible(False)
            ax_box.set_visible(False)
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
            box_values.append(_flat_values(m["auc_values"]))
            box_labels.append(method)

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

        bp = ax_box.boxplot(box_values, tick_labels=box_labels, patch_artist=True, showmeans=True, widths=0.60)
        for patch, method in zip(bp["boxes"], box_labels):
            patch.set_facecolor(METHOD_PASTELS[method])
            patch.set_edgecolor(METHOD_COLORS[method])
            patch.set_linewidth(1.55)
        for key in ["whiskers", "caps", "medians", "means"]:
            for artist in bp.get(key, []):
                artist.set_color(NEUTRAL)
                artist.set_linewidth(1.25)
        ax_box.axhline(0.5, color="#777777", linestyle="--", linewidth=1.25)
        ax_box.tick_params(axis="x", rotation=25, labelsize=9.0, width=1.4, length=5)
        clean_axis(ax_box, grid_axis="y")
        ax_box.spines["left"].set_linewidth(1.8)
        ax_box.spines["bottom"].set_linewidth(1.8)

    curve_axes[0].set_ylabel(r"$\langle \mathrm{AUC} \rangle$")
    box_axes[0].set_ylabel(r"$\langle \mathrm{AUC} \rangle$")
    for ax in curve_axes[1:] + box_axes[1:]:
        ax.tick_params(labelleft=False)

    y_values = []
    for _, row in ablation_df.iterrows():
        y_values.extend([row["auc_mean"] - row["auc_sd"], row["auc_mean"] + row["auc_sd"]])
        y_values.extend(row.get("auc_values", []))
    y_min = max(0.45, np.nanmin(y_values) - 0.03)
    y_max = min(1.02, np.nanmax(y_values) + 0.03)
    for ax in curve_axes + box_axes:
        if ax.get_visible():
            ax.set_ylim(y_min, y_max)

    handles = legend_handles[:len(METHOD_ORDER)]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=len(handles), frameon=True,
               facecolor="white", edgecolor="black", framealpha=1.0, borderpad=0.55)
    fig.suptitle("Reverse feature ablation", y=0.98, fontsize=15, weight="semibold")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.20, wspace=0.18, hspace=0.35)
    plt.show()

def _flat_values(series):
    vals = []
    for item in series:
        vals.extend(item if isinstance(item, (list, tuple, np.ndarray)) else [item])
    return np.asarray(vals, dtype=float)

def plot_figure6_ablation_all_datasets(ablation_df):
    fig = plt.figure(figsize=(13.4, 8.6), constrained_layout=False)
    gs = fig.add_gridspec(
        2, len(DATASET_ORDER),
        height_ratios=[1.25, 1.15]
    )
    curve_axes = [fig.add_subplot(gs[0, i]) for i in range(len(DATASET_ORDER))]
    box_axes = [fig.add_subplot(gs[1, i], sharey=curve_axes[0]) for i in range(len(DATASET_ORDER))]

    legend_handles = []
    for panel_idx, (ax_curve, ax_box, ds) in enumerate(zip(curve_axes, box_axes, DATASET_ORDER)):
        sub = ablation_df[ablation_df["dataset"] == ds]
        box_values, box_labels = [], []
        if sub.empty:
            ax_curve.set_visible(False)
            ax_box.set_visible(False)
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
            box_values.append(_flat_values(m["auc_values"]))
            box_labels.append(method)

        ax_curve.axhline(0.5, color="#777777", linestyle="--", linewidth=1.25)
        ax_curve.set_title(ds, color=DATASET_COLORS[ds], weight="semibold", pad=8, fontsize=13)
        col_letter = chr(ord("A") + panel_idx)

        ax_curve.text(
            -0.13, 1.08, f"{col_letter}1",
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

        bp = ax_box.boxplot(box_values, tick_labels=box_labels, patch_artist=True, showmeans=True, widths=0.60)
        for patch, method in zip(bp["boxes"], box_labels):
            patch.set_facecolor(METHOD_PASTELS[method])
            patch.set_edgecolor(METHOD_COLORS[method])
            patch.set_linewidth(1.55)
        for key in ["whiskers", "caps", "medians", "means"]:
            for artist in bp.get(key, []):
                artist.set_color(NEUTRAL)
                artist.set_linewidth(1.25)
        ax_box.axhline(0.5, color="#777777", linestyle="--", linewidth=1.25)
        ax_box.tick_params(axis="x", labelsize=9.0, width=1.4, length=5)
        ax_box.text(
            -0.13, 1.08, f"{col_letter}2",
            transform=ax_box.transAxes,
            fontsize=15,
            weight="bold",
            va="top",
            ha="left"
        )


        clean_axis(ax_box, grid_axis="y")

        for spine in ax_box.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)

        ax_box.tick_params(labelsize=9.0, width=1.2, length=4)
    curve_axes[0].set_ylabel(r"$\langle \mathrm{AUC} \rangle$")
    box_axes[0].set_ylabel(r"$\langle \mathrm{AUC} \rangle$")
    for ax in curve_axes[1:] + box_axes[1:]:
        ax.set_ylabel("") 
        ax.tick_params(labelleft=True) 

    y_values = []
    for _, row in ablation_df.iterrows():
        y_values.extend([row["auc_mean"] - row["auc_sd"], row["auc_mean"] + row["auc_sd"]])
        y_values.extend(row.get("auc_values", []))
    y_min = max(0.45, np.nanmin(y_values) - 0.03)
    y_max = min(1.02, np.nanmax(y_values) + 0.03)
    for ax in curve_axes + box_axes:
        if ax.get_visible():
            ax.set_ylim(y_min, y_max)

    handles = legend_handles[:len(METHOD_ORDER)]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=len(handles), frameon=True,
               facecolor="white", edgecolor="black", framealpha=0, borderpad=0.55)
    # fig.suptitle("Reverse feature ablation", y=0.98, fontsize=15, weight="semibold")

    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.15, wspace=0.18, hspace=0.45)
    return fig
