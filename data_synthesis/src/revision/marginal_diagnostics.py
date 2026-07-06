"""Feature-level real-versus-synthetic marginal diagnostics.

The heatmaps retain every feature, while the ECDF grids provide readable detail
for the features with the largest two-sample KS discrepancies.  Welch's t-tests
are reported as a table and are intentionally not used to select plot features.
"""

from dataclasses import dataclass
from collections.abc import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

from src.revision.common import *


DEFAULT_COMPARISON_METHODS = ["Bootstrap", "Column-wise", "CVAE", "GMM"]

def _dataset_significance_cmap(dataset, color):
    """Build discrete significance colours from a dataset's house colour."""
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=float)
    white = np.asarray(mpl.colors.to_rgb("#FAFAF7"), dtype=float)
    pale = 0.76 * white + 0.24 * rgb
    medium = 0.38 * white + 0.62 * rgb
    dark = 0.62 * rgb
    return mpl.colors.ListedColormap(
        [dark, rgb, medium, white],
        name=f"marginal_pvalue_{str(dataset).lower().replace(' ', '_')}",
    )


DATASET_SIGNIFICANCE_CMAPS = {
    dataset: _dataset_significance_cmap(dataset, color)
    for dataset, color in DATASET_COLORS.items()
}

PVALUE_BOUNDARIES = [0.0, 0.001, 0.01, 0.05, 1.0000001]
PVALUE_TICKS = [0.0005, 0.0055, 0.03, 0.525]
PVALUE_TICK_LABELS = ["<0.001", "0.001–0.01", "0.01–0.05", "≥0.05"]


@dataclass
class MarginalDiagnosticsResult:
    heatmap: object
    overlap_figures: dict
    t_test_table: pd.DataFrame
    ks_table: pd.DataFrame

    @property
    def ecdf_figures(self):
        """Backward-compatible alias for notebooks created before overlap plots."""
        return self.overlap_figures


def _finite(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _feature_statistics(real, synthetic):
    real = _finite(real)
    synthetic = _finite(synthetic)
    if len(real) == 0 or len(synthetic) == 0:
        return {
            "n_real": len(real),
            "n_synthetic": len(synthetic),
            "real_mean": np.nan,
            "synthetic_mean": np.nan,
            "mean_difference": np.nan,
            "t_statistic": np.nan,
            "p_value": np.nan,
            "ks_statistic": np.nan,
        }

    t_result = ttest_ind(real, synthetic, equal_var=False, nan_policy="omit")
    ks_result = ks_2samp(real, synthetic, alternative="two-sided", method="auto")
    real_mean = float(np.mean(real))
    synthetic_mean = float(np.mean(synthetic))
    return {
        "n_real": len(real),
        "n_synthetic": len(synthetic),
        "real_mean": real_mean,
        "synthetic_mean": synthetic_mean,
        "mean_difference": synthetic_mean - real_mean,
        "t_statistic": float(t_result.statistic),
        "p_value": float(t_result.pvalue),
        "ks_statistic": float(ks_result.statistic),
    }


def compute_marginal_test_table(
    real_data,
    synthetic_data,
    feature_names,
    dataset_order=None,
    method_order=None,
):
    """Return one Welch t-test and KS statistic per dataset, method, and feature."""
    dataset_order = list(dataset_order or real_data.keys())
    method_order = list(
        method_order or synthetic_data[dataset_order[0]].keys()
    )
    rows = []

    for dataset in dataset_order:
        X_real = np.asarray(real_data[dataset], dtype=np.float64)
        names = list(
            feature_names[dataset]
            if isinstance(feature_names, Mapping)
            else feature_names
        )
        if len(names) != X_real.shape[1]:
            raise ValueError(
                f"{dataset} has {X_real.shape[1]} columns but {len(names)} feature names."
            )

        for method in method_order:
            X_syn = np.asarray(synthetic_data[dataset][method], dtype=np.float64)
            if X_syn.shape[1] != X_real.shape[1]:
                raise ValueError(
                    f"{dataset}/{method} has {X_syn.shape[1]} synthetic columns; "
                    f"expected {X_real.shape[1]}."
                )
            for feature_index, feature in enumerate(names):
                stats = _feature_statistics(
                    X_real[:, feature_index], X_syn[:, feature_index]
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "feature_index": feature_index + 1,
                        "feature": feature,
                        **stats,
                    }
                )

    columns = [
        "dataset",
        "method",
        "feature_index",
        "feature",
        "n_real",
        "n_synthetic",
        "real_mean",
        "synthetic_mean",
        "mean_difference",
        "t_statistic",
        "p_value",
        "ks_statistic",
    ]
    return pd.DataFrame(rows, columns=columns)


def _marginal_table_summary(marginal_table, dataset_order, method_order):
    """Summarize feature-level marginal comparisons for a compact lead table."""
    table = marginal_table.copy()
    table["p_below_0_05"] = table["p_value"] < 0.05
    summary = (
        table.groupby(["dataset", "method"], sort=False)
        .agg(
            features_tested=("feature", "count"),
            features_p_below_0_05=("p_below_0_05", "sum"),
            median_p_value=("p_value", "median"),
            mean_ks_statistic=("ks_statistic", "mean"),
        )
        .reset_index()
    )
    summary["percent_p_below_0_05"] = (
        100
        * summary["features_p_below_0_05"]
        / summary["features_tested"]
    )
    dataset_rank = {name: index for index, name in enumerate(dataset_order)}
    method_rank = {name: index for index, name in enumerate(method_order)}
    summary["_dataset_order"] = summary["dataset"].map(dataset_rank)
    summary["_method_order"] = summary["method"].map(method_rank)
    return (
        summary.sort_values(["_dataset_order", "_method_order"])
        .drop(columns=["_dataset_order", "_method_order"])
        .reset_index(drop=True)
    )


def _latex_table_fragment(table, caption, label, column_format, longtable):
    """Render a booktabs table and apply supplement-friendly sizing."""
    latex = table.to_latex(
        index=False,
        escape=True,
        longtable=longtable,
        caption=caption,
        label=label,
        column_format=column_format,
        na_rep="--",
    )
    environment = "longtable" if longtable else "table"
    if longtable:
        return (
            "\\begin{landscape}\n"
            "\\begingroup\n"
            "\\scriptsize\n"
            "\\setlength{\\tabcolsep}{3pt}\n"
            f"{latex}"
            "\\endgroup\n"
            "\\end{landscape}\n"
        )
    return (
        f"\\begin{{{environment}}}[!htbp]\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        + latex.replace(
            f"\\begin{{{environment}}}",
            "",
            1,
        ).replace(
            f"\\end{{{environment}}}",
            "",
            1,
        )
        + f"\\end{{{environment}}}\n"
    )


def export_marginal_comparison_tables(
    marginal_table,
    output_dir,
    dataset_order=None,
    method_order=None,
):
    """Export complete original-method marginal comparisons for Overleaf.

    The returned master file can be uploaded with the other generated files and
    included from the supplement with
    ``\\input{supplementary_marginal_comparisons}``.
    """
    dataset_order = list(dataset_order or marginal_table["dataset"].unique())
    method_order = list(method_order or DEFAULT_COMPARISON_METHODS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = marginal_table.copy()
    table["p_below_0_05"] = table["p_value"] < 0.05
    dataset_rank = {name: index for index, name in enumerate(dataset_order)}
    method_rank = {name: index for index, name in enumerate(method_order)}
    table["_dataset_order"] = table["dataset"].map(dataset_rank)
    table["_method_order"] = table["method"].map(method_rank)
    table = (
        table.sort_values(
            ["_dataset_order", "_method_order", "feature_index"]
        )
        .drop(columns=["_dataset_order", "_method_order"])
        .reset_index(drop=True)
    )

    csv_path = output_dir / "supplementary_marginal_comparisons.csv"
    table.to_csv(csv_path, index=False)

    summary = _marginal_table_summary(
        table,
        dataset_order=dataset_order,
        method_order=method_order,
    ).rename(
        columns={
            "dataset": "Dataset",
            "method": "Method",
            "features_tested": "Features",
            "features_p_below_0_05": "p < 0.05",
            "median_p_value": "Median p",
            "mean_ks_statistic": "Mean KS",
            "percent_p_below_0_05": "Percent p < 0.05",
        }
    )
    summary["Median p"] = summary["Median p"].map(lambda value: f"{value:.3g}")
    summary["Mean KS"] = summary["Mean KS"].map(lambda value: f"{value:.3f}")
    summary["Percent p < 0.05"] = summary["Percent p < 0.05"].map(
        lambda value: f"{value:.1f}%"
    )
    summary_path = output_dir / "supplementary_marginal_summary.tex"
    summary_path.write_text(
        _latex_table_fragment(
            summary,
            (
                "Summary of feature-level marginal comparisons for the three "
                "datasets and four original synthesis methods. Raw Welch "
                "two-sample t-test p-values are reported without multiplicity "
                "adjustment; KS denotes the two-sample Kolmogorov--Smirnov statistic."
            ),
            "tab:marginal-summary",
            "@{}llrrrrr@{}",
            longtable=False,
        ),
        encoding="utf-8",
    )

    detail_paths = {}
    detail_columns = [
        "method",
        "feature_index",
        "feature",
        "n_real",
        "n_synthetic",
        "real_mean",
        "synthetic_mean",
        "mean_difference",
        "t_statistic",
        "p_value",
        "ks_statistic",
        "p_below_0_05",
    ]
    display_names = {
        "method": "Method",
        "feature_index": "No.",
        "feature": "Feature",
        "n_real": "n real",
        "n_synthetic": "n synthetic",
        "real_mean": "Real mean",
        "synthetic_mean": "Synthetic mean",
        "mean_difference": "Difference",
        "t_statistic": "Welch t",
        "p_value": "p",
        "ks_statistic": "KS",
        "p_below_0_05": "p < 0.05",
    }
    number_formats = {
        "real_mean": lambda value: f"{value:.4g}",
        "synthetic_mean": lambda value: f"{value:.4g}",
        "mean_difference": lambda value: f"{value:+.4g}",
        "t_statistic": lambda value: f"{value:+.3f}",
        "p_value": lambda value: f"{value:.3g}",
        "ks_statistic": lambda value: f"{value:.3f}",
        "p_below_0_05": lambda value: "Yes" if value else "No",
    }
    for dataset in dataset_order:
        detail = (
            table.loc[table["dataset"] == dataset, detail_columns]
            .reset_index(drop=True)
        )
        for column, formatter in number_formats.items():
            detail[column] = detail[column].map(formatter)
        detail = detail.rename(columns=display_names)
        slug = str(dataset).lower().replace(" ", "_").replace("-", "_")
        detail_path = output_dir / f"supplementary_marginal_{slug}.tex"
        detail_path.write_text(
            _latex_table_fragment(
                detail,
                (
                    f"Complete feature-level marginal comparisons for {dataset}. "
                    "Difference is synthetic minus real. Welch p-values are raw "
                    "and unadjusted; KS is the two-sample Kolmogorov--Smirnov statistic."
                ),
                f"tab:marginal-{slug.replace('_', '-')}",
                "@{}lrp{4.0cm}rrrrrrrrr@{}",
                longtable=True,
            ),
            encoding="utf-8",
        )
        detail_paths[dataset] = detail_path

    master_path = output_dir / "supplementary_marginal_comparisons.tex"
    input_lines = [
        "% Auto-generated by make_main_figures_revision.ipynb.",
        "% Preamble requirements: \\usepackage{booktabs,longtable,pdflscape}",
        "% Upload this file and the four referenced .tex files to one Overleaf folder.",
        "\\input{supplementary_marginal_summary}",
    ]
    input_lines.extend(
        f"\\input{{{path.stem}}}" for path in detail_paths.values()
    )
    master_path.write_text("\n".join(input_lines) + "\n", encoding="utf-8")

    maintext_path = output_dir / "maintext_marginal_reference.tex"
    detail_labels = [
        f"\\ref{{tab:marginal-{str(dataset).lower().replace(' ', '-')}}}"
        for dataset in dataset_order
    ]
    if len(detail_labels) == 1:
        detail_refs = f"Table~{detail_labels[0]}"
    elif len(detail_labels) == 2:
        detail_refs = f"Tables~{detail_labels[0]} and {detail_labels[1]}"
    else:
        detail_refs = (
            f"Tables~{', '.join(detail_labels[:-1])}, and {detail_labels[-1]}"
        )
    maintext_path.write_text(
        (
            "Complete feature-level marginal comparisons for all three datasets "
            "and all four original synthesis methods are reported in "
            f"{detail_refs}; a compact overview is provided in "
            "Table~\\ref{tab:marginal-summary}.\n"
        ),
        encoding="utf-8",
    )

    archive_path = output_dir / "overleaf_marginal_tables.zip"
    archive_members = [
        master_path,
        summary_path,
        *detail_paths.values(),
        csv_path,
        maintext_path,
    ]
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in archive_members:
            archive.write(path, arcname=path.name)

    return {
        "master": master_path,
        "summary": summary_path,
        "details": detail_paths,
        "csv": csv_path,
        "maintext_reference": maintext_path,
        "archive": archive_path,
    }


def _short_feature_name(name, max_len=28):
    name = str(name).replace("_", " ")
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "…"


def plot_pvalue_heatmaps(
    marginal_table,
    dataset_order=None,
    method_order=None,
    save_path=None,
):
    """Plot raw Welch t-test p-values as discrete significance categories."""
    dataset_order = list(dataset_order or marginal_table["dataset"].unique())
    method_order = list(method_order or DEFAULT_COMPARISON_METHODS)
    method_order = [m for m in method_order if m in set(marginal_table["method"])]
    if not method_order:
        raise ValueError("No requested methods are present in marginal_table.")

    fig, axes = plt.subplots(
        1,
        len(dataset_order),
        figsize=(4.35 * len(dataset_order) + 0.8, 13.5),
        squeeze=False,
        constrained_layout=False,
        gridspec_kw={"wspace": 0.62},
    )
    axes = axes[0]
    panel_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    for panel_index, (ax, dataset) in enumerate(zip(axes, dataset_order)):
        sub = marginal_table[marginal_table["dataset"] == dataset]
        matrix = sub.pivot(index="feature", columns="method", values="p_value")
        matrix = matrix.reindex(columns=method_order)
        order = matrix.min(axis=1).sort_values(ascending=True).index
        matrix = matrix.reindex(order)

        cmap = DATASET_SIGNIFICANCE_CMAPS.get(
            dataset,
            _dataset_significance_cmap(dataset, "#666666"),
        )
        norm = mpl.colors.BoundaryNorm(PVALUE_BOUNDARIES, cmap.N)

        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        ax.set_xticks(np.arange(len(method_order)))
        ax.set_xticklabels(
            ["Column-\nwise" if m == "Column-wise" else m for m in method_order],
            fontsize=8.6,
        )
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(
            [_short_feature_name(name) for name in matrix.index],
            fontsize=5.5 if len(matrix.index) > 35 else 7.0,
        )
        ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=5)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_title(
            dataset,
            fontsize=12.0,
            weight="semibold",
            color=DATASET_COLORS.get(dataset, "#333333"),
            pad=12,
        )
        ax.text(
            -0.10,
            1.035,
            panel_letters[panel_index],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            weight="bold",
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("#333333")

        cbar = fig.colorbar(
            image,
            ax=ax,
            orientation="horizontal",
            fraction=0.025,
            pad=0.035,
            aspect=28,
            boundaries=PVALUE_BOUNDARIES,
            ticks=PVALUE_TICKS,
            spacing="uniform",
        )
        cbar.ax.tick_params(labelsize=7.8, length=2.5)
        cbar.ax.set_xticklabels(PVALUE_TICK_LABELS)
        cbar.outline.set_linewidth(0.7)
        cbar.set_label("Welch t-test p-value", fontsize=8.3, labelpad=3)

    fig.subplots_adjust(left=0.13, right=0.985, top=0.93, bottom=0.065)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def _standardize_from_real(real, synthetic):
    real = _finite(real)
    synthetic = _finite(synthetic)
    if len(real) == 0:
        return real, synthetic
    center = float(np.mean(real))
    scale = float(np.std(real, ddof=1)) if len(real) > 1 else 1.0
    if not np.isfinite(scale) or np.isclose(scale, 0):
        scale = 1.0
    return (real - center) / scale, (synthetic - center) / scale


def _pooled_histogram_spec(
    value_groups,
    lower_quantile=0.01,
    upper_quantile=0.99,
    min_bins=10,
    max_bins=26,
    axis_pad_fraction=0.08,
):
    """Return robust shared limits and bins for one feature across all methods."""
    finite_groups = [_finite(values) for values in value_groups]
    finite_groups = [values for values in finite_groups if len(values)]
    combined = np.concatenate(finite_groups) if finite_groups else np.array([])
    if len(combined) == 0:
        return -1.0, 1.0, np.linspace(-1, 1, min_bins + 1)
    lo, hi = np.quantile(combined, [lower_quantile, upper_quantile])
    lo, hi = float(lo), float(hi)
    if np.isclose(lo, hi):
        pad = max(0.5, abs(lo) * 0.05)
        lo, hi = lo - pad, hi + pad
    visible = combined[(combined >= lo) & (combined <= hi)]
    edges = np.histogram_bin_edges(visible, bins="fd")
    n_bins = int(np.clip(len(edges) - 1, min_bins, max_bins))
    histogram_edges = np.linspace(lo, hi, n_bins + 1)
    axis_pad = axis_pad_fraction * (hi - lo)
    return lo - axis_pad, hi + axis_pad, histogram_edges


def plot_distribution_overlaps(
    real_data,
    synthetic_data,
    feature_names,
    marginal_table,
    dataset,
    method_order=None,
    top_n=6,
    save_path=None,
):
    """Plot overlapping marginal histograms for features with the largest KS gaps."""
    method_order = list(method_order or DEFAULT_COMPARISON_METHODS)
    available_methods = set(synthetic_data[dataset])
    method_order = [method for method in method_order if method in available_methods]
    X_real = np.asarray(real_data[dataset], dtype=np.float64)
    names = list(
        feature_names[dataset]
        if isinstance(feature_names, Mapping)
        else feature_names
    )
    name_to_index = {name: index for index, name in enumerate(names)}
    synthetic_arrays = {
        method: np.asarray(synthetic_data[dataset][method], dtype=np.float64)
        for method in method_order
    }

    sub = marginal_table[marginal_table["dataset"] == dataset]
    ranked_features = (
        sub.groupby("feature", sort=False)["ks_statistic"]
        .max()
        .sort_values(ascending=False)
        .head(min(top_n, len(names)))
        .index.tolist()
    )

    fig, axes = plt.subplots(
        len(ranked_features),
        len(method_order),
        figsize=(3.05 * len(method_order) + 1.5, 1.72 * len(ranked_features) + 1.25),
        squeeze=False,
        sharex="row",
        sharey=True,
        constrained_layout=False,
    )
    panel_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    panel_index = 0

    for row, feature in enumerate(ranked_features):
        feature_index = name_to_index[feature]
        real_z, _ = _standardize_from_real(
            X_real[:, feature_index], X_real[:, feature_index]
        )
        synthetic_z = {
            method: _standardize_from_real(
                X_real[:, feature_index], synthetic_arrays[method][:, feature_index]
            )[1]
            for method in method_order
        }
        x_min, x_max, edges = _pooled_histogram_spec(
            [real_z, *synthetic_z.values()],
            lower_quantile=0.01,
            upper_quantile=0.99,
        )

        for col, method in enumerate(method_order):
            ax = axes[row, col]
            syn_z = synthetic_z[method]
            syn_color = METHOD_COLORS.get(method, "#D55E00")
            ax.hist(
                real_z,
                bins=edges,
                density=True,
                histtype="stepfilled",
                color="#777777",
                alpha=0.36,
                linewidth=0,
                label="Real",
            )
            ax.hist(
                syn_z,
                bins=edges,
                density=True,
                histtype="stepfilled",
                color=syn_color,
                alpha=0.36,
                linewidth=0,
                label="Synthetic",
            )
            ax.hist(real_z, bins=edges, density=True, histtype="step", color="#333434", linewidth=1.2)
            ax.hist(syn_z, bins=edges, density=True, histtype="step", color=syn_color, linewidth=1.2)
            ks_value = sub.loc[
                (sub["method"] == method) & (sub["feature"] == feature),
                "ks_statistic",
            ].iloc[0]
            ax.text(
                0.97,
                0.08,
                f"KS = {ks_value:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
            )
            ax.text(
                0.025,
                0.97,
                panel_letters[panel_index],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                weight="bold",
            )
            panel_index += 1

            if row == 0:
                ax.set_title(method, fontsize=10.5, weight="semibold", pad=7)
            if col == 0:
                ax.set_ylabel(
                    _short_feature_name(feature, max_len=24) + "\nDensity",
                    fontsize=8.1,
                    weight="semibold",
                    labelpad=7,
                )
            if row == len(ranked_features) - 1:
                ax.set_xlabel("Standardized value", fontsize=8.2)

            ax.set_xlim(x_min, x_max)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.45)
            ax.tick_params(labelsize=7.2, width=0.8, length=3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("#555555")

    handles = [
        mpl.patches.Patch(facecolor="#777777", edgecolor="#333434", alpha=0.36, label="Real"),
        mpl.patches.Patch(facecolor="#777777", edgecolor="#777777", alpha=0.36, label="Synthetic (column method colour)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=2,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle(
        f"{dataset}: real–synthetic marginal distribution overlap",
        fontsize=12.5,
        weight="semibold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.047,
        "Axes are based on the pooled 1st–99th percentile range with visual padding; KS uses all observations.",
        ha="center",
        va="bottom",
        fontsize=8.3,
        color="#555555",
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.925, bottom=0.12, wspace=0.12, hspace=0.18)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def build_marginal_diagnostics(
    real_data,
    synthetic_data,
    feature_names,
    dataset_order=None,
    method_order=None,
    top_n=6,
):
    """Build the complete t-test table, all-feature heatmap, and ECDF details."""
    dataset_order = list(dataset_order or real_data.keys())
    method_order = list(method_order or DEFAULT_COMPARISON_METHODS)
    table = compute_marginal_test_table(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_names,
        dataset_order=dataset_order,
        method_order=method_order,
    )
    heatmap = plot_pvalue_heatmaps(
        table,
        dataset_order=dataset_order,
        method_order=method_order,
    )
    overlap_figures = {
        dataset: plot_distribution_overlaps(
            real_data=real_data,
            synthetic_data=synthetic_data,
            feature_names=feature_names,
            marginal_table=table,
            dataset=dataset,
            method_order=method_order,
            top_n=top_n,
        )
        for dataset in dataset_order
    }
    ks_table = table[
        ["dataset", "method", "feature_index", "feature", "ks_statistic"]
    ].sort_values(["dataset", "method", "ks_statistic"], ascending=[True, True, False])
    return MarginalDiagnosticsResult(
        heatmap=heatmap,
        overlap_figures=overlap_figures,
        t_test_table=table,
        ks_table=ks_table,
    )
