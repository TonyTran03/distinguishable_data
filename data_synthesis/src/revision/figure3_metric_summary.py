"""Figure 3: utility and distributional metric summaries."""

from src.revision.common import *
from src.revision.stats import mean_kld_by_feature, nn_distance_mean, tstr_values

def build_metric_table(datasets, auc_runs, seed=SEED, cvae_epochs=CVAE_EPOCHS):
    rows = []
    for ds in DATASET_ORDER:
        data = datasets[ds]
        X_real = np.asarray(data["X"], dtype=np.float32)
        y_real = np.asarray(data["y"], dtype=int)
        for method in METHOD_ORDER:
            print(f"[metric table] {ds} - {method}")
            X_syn, y_syn = sample_synthetic(ds, data, method, seed=seed, cvae_epochs=cvae_epochs)
            tstr, trtr = tstr_values(X_real, y_real, X_syn, y_syn, seed=seed)
            gap = np.abs(trtr - tstr)
            auc_vals = auc_runs[(auc_runs["dataset"] == ds) & (auc_runs["method"] == method)]["separability_auc"].to_numpy()
            klds = mean_kld_by_feature(X_real, X_syn)
            rows.append({
                "dataset": ds,
                "method": method,
                "rf_auc_mean": float(np.mean(auc_vals)),
                "rf_auc_sd": float(np.std(auc_vals)),
                "rf_auc_values": auc_vals.tolist(),
                "tstr_f1_mean": float(np.mean(tstr)),
                "tstr_f1_sd": float(np.std(tstr)),
                "tstr_f1_values": tstr.tolist(),
                "utility_gap_abs_mean": float(np.mean(gap)),
                "utility_gap_abs_sd": float(np.std(gap)),
                "utility_gap_abs_values": gap.tolist(),
                "mean_feature_kld": float(np.mean(klds)),
                "nn_distance_mean": nn_distance_mean(X_real, X_syn),
            })
    return pd.DataFrame(rows)

def format_mean_sd(mean, sd):
    return f"{mean:.2f} +/- {sd:.2f}"

def _blend_with_white(hex_color, amount=0.78):
    rgb = np.asarray(mpl.colors.to_rgb(hex_color))
    return tuple(rgb * (1 - amount) + np.ones(3) * amount)

def _metric_cell_color(values, value, higher_is_better):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or not np.isfinite(value):
        return "#FFFFFF"
    lo, hi = float(values.min()), float(values.max())
    if np.isclose(lo, hi):
        score = 0.5
    else:
        score = (float(value) - lo) / (hi - lo)
    if not higher_is_better:
        score = 1 - score
    good = np.asarray(mpl.colors.to_rgb("#5AA469"))
    mid = np.asarray(mpl.colors.to_rgb("#F7F7F2"))
    bad = np.asarray(mpl.colors.to_rgb("#D6604D"))
    if score >= 0.5:
        t = (score - 0.5) / 0.5
        rgb = mid * (1 - t) + good * t
    else:
        t = score / 0.5
        rgb = bad * (1 - t) + mid * t
    return tuple(rgb * 0.48 + np.ones(3) * 0.52)

def build_main_metric_summary_table(metric_table):
    summary = metric_table.copy()
    summary = summary.sort_values(["dataset", "method"], key=lambda s: s.map({**{d: i for i, d in enumerate(DATASET_ORDER)}, **{m: i for i, m in enumerate(METHOD_ORDER)}}).fillna(99))
    display_df = pd.DataFrame({
        "Dataset": summary["dataset"],
        "Method": summary["method"],
        "AUC": summary["rf_auc_mean"].map(lambda v: f"{v:.2f}"),
        "TSTR F1": summary["tstr_f1_mean"].map(lambda v: f"{v:.2f}"),
        "|TRTR-TSTR|": summary["utility_gap_abs_mean"].map(lambda v: f"{v:.2f}"),
    })
    numeric_df = summary[["rf_auc_mean", "tstr_f1_mean", "utility_gap_abs_mean"]].copy()
    return display_df, numeric_df

def plot_main_metric_summary_table(metric_table):
    display_df, numeric_df = build_main_metric_summary_table(metric_table)
    fig, ax = plt.subplots(figsize=(10.8, 5.15))
    ax.axis("off")
    tbl = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.24, 0.22, 0.15, 0.16, 0.18],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.8)
    tbl.scale(1.0, 1.42)

    metric_specs = {
        2: ("rf_auc_mean", False),
        3: ("tstr_f1_mean", True),
        4: ("utility_gap_abs_mean", False),
    }
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.6)
        if r == 0:
            cell.set_facecolor(NEUTRAL)
            cell.set_text_props(color="white", weight="bold")
            continue

        ds = display_df.iloc[r - 1]["Dataset"]
        method = display_df.iloc[r - 1]["Method"]
        base = "#FFFFFF" if r % 2 else "#F7F7F7"
        cell.set_facecolor(base)
        if c == 0:
            cell.set_facecolor(_blend_with_white(DATASET_COLORS.get(ds, NEUTRAL), 0.84))
            cell.set_text_props(color=DATASET_COLORS.get(ds, NEUTRAL), weight="bold")
        elif c == 1:
            cell.set_facecolor(_blend_with_white(METHOD_COLORS.get(method, NEUTRAL), 0.83))
            cell.set_text_props(color=METHOD_COLORS.get(method, NEUTRAL), weight="bold")
        elif c in metric_specs:
            col, higher_is_better = metric_specs[c]
            value = numeric_df.iloc[r - 1][col]
            cell.set_facecolor(_metric_cell_color(numeric_df[col], value, higher_is_better))

    ax.set_title("Main metric summary", loc="left", fontsize=16, weight="bold", pad=10)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.10)
    return fig, display_df

def build_supplementary_metric_table(metric_table, structure_metrics=None):
    supp = metric_table[["dataset", "method", "mean_feature_kld", "nn_distance_mean"]].copy()
    structural_cols = ["frobenius_deviation", "edge_recovery", "synthetic_only_rate"]
    if structure_metrics is not None and not structure_metrics.empty:
        structure_metrics = structure_metrics.copy()
        if "frobenius_deviation" not in structure_metrics.columns and "frobenius_corr_discrepancy" in structure_metrics.columns:
            structure_metrics["frobenius_deviation"] = structure_metrics["frobenius_corr_discrepancy"]
        keep = ["dataset", "method"]
        keep += [c for c in structural_cols if c in structure_metrics.columns]
        supp = supp.merge(structure_metrics[keep], on=["dataset", "method"], how="left")

    for col in structural_cols:
        if col not in supp.columns:
            supp[col] = np.nan

    supp = supp.sort_values(["dataset", "method"], key=lambda s: s.map({**{d: i for i, d in enumerate(DATASET_ORDER)}, **{m: i for i, m in enumerate(METHOD_ORDER)}}).fillna(99))
    display_df = pd.DataFrame({
        "Dataset": supp["dataset"],
        "Method": supp["method"],
        "Mean KLD": supp["mean_feature_kld"].map(lambda v: f"{v:.2f}"),
        "NN realism dist.": supp["nn_distance_mean"].map(lambda v: f"{v:.2f}"),
        "Structure Frobenius": supp["frobenius_deviation"].map(lambda v: "--" if pd.isna(v) else f"{v:.2f}"),
        "Edge recovery": supp["edge_recovery"].map(lambda v: "--" if pd.isna(v) else f"{v:.2f}"),
        "Synthetic-only rate": supp["synthetic_only_rate"].map(lambda v: "--" if pd.isna(v) else f"{v:.2f}"),
    })
    numeric_df = supp[["mean_feature_kld", "nn_distance_mean", "frobenius_deviation", "edge_recovery", "synthetic_only_rate"]].copy()
    return display_df, numeric_df

def plot_supplementary_metric_table(metric_table, structure_metrics=None):
    display_df, numeric_df = build_supplementary_metric_table(metric_table, structure_metrics)
    fig, ax = plt.subplots(figsize=(14.2, 5.35))
    ax.axis("off")
    tbl = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.16, 0.13, 0.12, 0.14, 0.16, 0.13, 0.16],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)
    tbl.scale(1.0, 1.42)

    metric_specs = {
        2: ("mean_feature_kld", False),
        3: ("nn_distance_mean", False),
        4: ("frobenius_deviation", False),
        5: ("edge_recovery", True),
        6: ("synthetic_only_rate", False),
    }
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.4)
        if r == 0:
            cell.set_facecolor(NEUTRAL)
            cell.set_text_props(color="white", weight="bold")
            continue

        ds = display_df.iloc[r - 1]["Dataset"]
        method = display_df.iloc[r - 1]["Method"]
        cell.set_facecolor("#FFFFFF" if r % 2 else "#F7F7F7")
        if c == 0:
            cell.set_facecolor(_blend_with_white(DATASET_COLORS.get(ds, NEUTRAL), 0.84))
            cell.set_text_props(color=DATASET_COLORS.get(ds, NEUTRAL), weight="bold")
        elif c == 1:
            cell.set_facecolor(_blend_with_white(METHOD_COLORS.get(method, NEUTRAL), 0.83))
            cell.set_text_props(color=METHOD_COLORS.get(method, NEUTRAL), weight="bold")
        elif c in metric_specs:
            col, higher_is_better = metric_specs[c]
            value = numeric_df.iloc[r - 1][col]
            cell.set_facecolor(_metric_cell_color(numeric_df[col], value, higher_is_better))

    ax.set_title("Supplementary metric summary", loc="left", fontsize=15.5, weight="bold", pad=10)
    ax.text(
        0.0, -0.035,
        "KLD and nearest-neighbor distance come from conventional fidelity diagnostics; structural metrics come from the active Figure 4 Graphical Lasso analysis.",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color="#555555",
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.10)
    return fig, display_df

def _metric_values(row, value_col, mean_col):
    vals = row.get(value_col, None)
    if isinstance(vals, (list, tuple, np.ndarray)) and len(vals) > 0:
        return np.asarray(vals, dtype=float)
    return np.asarray([float(row[mean_col])], dtype=float)

def _draw_grouped_violin_panel(ax, metric_table, value_col, mean_col, sd_col, ylabel, panel, ylim=None):
    x = np.arange(len(METHOD_ORDER))
    offsets = np.linspace(-0.27, 0.27, len(DATASET_ORDER))
    width = 0.20
    legend_handles = []

    for i, method in enumerate(METHOD_ORDER):
        if i % 2 == 0:
            ax.axvspan(i - 0.48, i + 0.48, color="#F3F3F3", zorder=0)
        if i > 0:
            ax.axvline(i - 0.5, color="#D8D8D8", linewidth=0.9, zorder=0)

    for offset, ds in zip(offsets, DATASET_ORDER):
        sub = metric_table[metric_table["dataset"] == ds].set_index("method").reindex(METHOD_ORDER)
        positions = x + offset
        values = [_metric_values(row, value_col, mean_col) for _, row in sub.iterrows()]
        violins = ax.violinplot(values, positions=positions, widths=width, showmeans=False, showmedians=False, showextrema=False)
        for body in violins["bodies"]:
            body.set_facecolor(DATASET_COLORS[ds])
            body.set_edgecolor(DATASET_COLORS[ds])
            body.set_alpha(0.28)
            body.set_linewidth(1.4)
            body.set_zorder(2)
        for pos, vals in zip(positions, values):
            jitter = np.linspace(-0.030, 0.030, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=14, color=DATASET_COLORS[ds], alpha=0.50,
                       edgecolors="white", linewidths=0.25, zorder=3)
        means = sub[mean_col].to_numpy(dtype=float)
        sds = sub[sd_col].to_numpy(dtype=float)
        handle = ax.errorbar(positions, means, yerr=sds, color=DATASET_COLORS[ds], marker=DATASET_MARKERS[ds],
                             linestyle="none", markersize=8.2, elinewidth=2.2, capsize=5.0, capthick=2.0,
                             markeredgecolor="white", markeredgewidth=0.6, label=ds, zorder=4)
        legend_handles.append(handle)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER, rotation=25, ha="right")
    ax.set_xlabel("Synthetic generator")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if mean_col == "rf_auc_mean":
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.6)
    if mean_col == "utility_gap_abs_mean":
        ax.axhline(0, color="#777777", linestyle="--", linewidth=1.6)
        ax.text(0.96, 0.92, "Smaller is better", ha="right", va="top", transform=ax.transAxes, fontsize=10.0)
    ax.set_title(ylabel, loc="left", weight="bold")
    clean_axis(ax, grid_axis="y")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(2.2)
    ax.tick_params(width=1.8, length=6)
    add_panel_label(ax, panel)
    return legend_handles

def plot_figure3_metric_panels(metric_table):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=False)
    panel_specs = [
        ("rf_auc_values", "rf_auc_mean", "rf_auc_sd", r"Discriminator $\langle \mathrm{AUC} \rangle$", "A", (0.45, 1.03)),
        ("tstr_f1_values", "tstr_f1_mean", "tstr_f1_sd", "TSTR F1", "B", None),
        ("utility_gap_abs_values", "utility_gap_abs_mean", "utility_gap_abs_sd", "|TRTR - TSTR|", "C", None),
    ]
    legend_handles = None
    for ax, spec in zip(axes, panel_specs):
        handles = _draw_grouped_violin_panel(ax, metric_table, *spec)
        if legend_handles is None:
            legend_handles = handles
    fig.legend(legend_handles, DATASET_ORDER, loc="upper left", bbox_to_anchor=(0.075, 0.985), ncol=3,
               frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.23, wspace=0.30)
    plt.show()
