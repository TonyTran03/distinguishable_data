"""Figure 5: label-preserving noise sensitivity analysis."""

import ast

from src.revision.common import *
from src.revision.stats import stratified_subsample
from src.revision.cache import _read_cache, _write_cache

def compute_noise_sensitivity(datasets, seed=SEED, repeats=NOISE_REPEATS, sigmas=PROBE_SIGMAS, frac=NOISE_FRAC, cvae_epochs=CVAE_EPOCHS):
    rows = []
    for ds in DATASET_ORDER:
        data = datasets[ds]
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=int)
        n0 = max(2, int((y == 0).sum() * frac))
        n1 = max(2, int((y == 1).sum() * frac))
        X_sub, y_sub = stratified_subsample(X, y, n0, n1, seed=seed)
        stds = X_sub.std(axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        feat_cols = [f"f{i}" for i in range(X_sub.shape[1])]
        print(f"[noise] training CVAE for {ds}")
        state = train_cvae(X_sub, y_sub, cfg=Config(seed=seed, epochs=cvae_epochs, batch_size=32), verbose=False)
        generators = {
            "Bootstrap": lambda s: sample_bootstrap(X_sub, y_sub, n0, n1, seed=s),
            "Column-wise": lambda s: sample_columnwise(X_sub, y_sub, n0, n1, seed=s),
            "GMM": lambda s: sample_gmm(X_sub, y_sub, n0, n1, seed=s),
            "CVAE": lambda s: sample_trained_cvae(state, n0, n1, seed=s),
        }
        for method, gen in generators.items():
            for sigma in sigmas:
                vals = []
                for r in range(repeats):
                    rep_seed = seed + r
                    X_syn, _ = gen(rep_seed)
                    X_syn = np.asarray(X_syn, dtype=np.float64)
                    if sigma > 0:
                        rng = np.random.default_rng(rep_seed + 500)
                        X_syn = X_syn + rng.standard_normal(X_syn.shape) * stds * sigma
                    real_df = pd.DataFrame(X_sub, columns=feat_cols)
                    real_df["target"] = 1
                    syn_df = pd.DataFrame(X_syn, columns=feat_cols)
                    syn_df["target"] = 0
                    combined = pd.concat([real_df, syn_df], ignore_index=True)
                    avg, _, _ = RFWrapper.from_combined(combined)
                    vals.append(max(float(avg), 1.0 - float(avg)))
                rows.append({"dataset": ds, "method": method, "sigma": sigma,
                             "sep_mean": float(np.mean(vals)), "sep_sd": float(np.std(vals)),
                             "sep_values": [float(v) for v in vals]})
                print(f"[noise] {ds} {method} sigma={sigma}")
    return pd.DataFrame(rows)

def get_noise_sensitivity(force=False):
    cached = None if force else _read_cache("noise_df")
    if cached is not None:
        return cached
    result = compute_noise_sensitivity(require_datasets())
    return _write_cache("noise_df", result)

def noise_axis_position(sigma, expand_until=0.5, expand_factor=2.4):
    sigma = np.asarray(sigma, dtype=float)
    return np.where(sigma <= expand_until, sigma * expand_factor, expand_until * expand_factor + (sigma - expand_until))

def compute_noise_auc_floor(noise_df):
    return (
        noise_df.groupby(["dataset", "method"], as_index=False)["sep_mean"]
        .min()
        .rename(columns={"sep_mean": "min_auc"})
    )

def _coerce_sep_values(values):
    if isinstance(values, str):
        values = ast.literal_eval(values)
    return np.asarray(values, dtype=float)

def _noise_percentile_band(m, lower=2.5, upper=97.5):
    if "sep_values" not in m:
        y = m["sep_mean"].to_numpy(dtype=float)
        sd = m["sep_sd"].to_numpy(dtype=float)
        return y - sd, y + sd
    vals = [_coerce_sep_values(v) for v in m["sep_values"]]
    lo = np.asarray([np.percentile(v, lower) for v in vals], dtype=float)
    hi = np.asarray([np.percentile(v, upper) for v in vals], dtype=float)
    return lo, hi

def _plot_noise_method_panels(axes, noise_df, tick_positions, tick_labels):
    floor_df = compute_noise_auc_floor(noise_df)
    x_max = noise_axis_position(float(np.max(noise_df["sigma"])))

    for ax, method, panel in zip(axes, METHOD_ORDER, ["D1", "D2", "D3", "D4"]):
        sub = noise_df[noise_df["method"] == method]
        for ds in DATASET_ORDER:
            m = sub[sub["dataset"] == ds].sort_values("sigma")
            if m.empty:
                continue
            x = noise_axis_position(m["sigma"])
            y = m["sep_mean"].to_numpy(dtype=float)
            lo, hi = _noise_percentile_band(m)
            ax.plot(
                x,
                y,
                marker=DATASET_MARKERS[ds],
                color=DATASET_COLORS[ds],
                label=ds,
                linewidth=2.0,
                markersize=4.8,
                zorder=3,
            )
            ax.fill_between(x, lo, hi, color=DATASET_COLORS[ds], alpha=0.10, linewidth=0, zorder=2)

        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.15)
        ax.set_title(f"{panel}. {method}", color=METHOD_COLORS[method], weight="semibold", fontsize=11.5)
        ax.set_xlabel("Noise (sigma)")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(noise_axis_position(0) - 0.05, x_max + 0.08)
        ax.set_ylim(0.45, 1.03)
        clean_axis(ax, grid_axis="y")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
        ax.tick_params(labelsize=8.0, width=1.2, length=4, labelleft=True)

    axes[0].set_ylabel(r"$\langle \mathrm{AUC} \rangle$")
    return floor_df

def plot_figure5_noise(noise_df):
    fig = plt.figure(figsize=(14.8, 3.7), constrained_layout=False)
    gs = fig.add_gridspec(1, 4, wspace=0.34)
    method_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
    ]
    dataset_handles = []
    tick_sigmas = np.asarray(sorted(noise_df["sigma"].unique()), dtype=float)
    tick_sigmas = tick_sigmas[np.isin(tick_sigmas, [0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0])]
    tick_positions = noise_axis_position(tick_sigmas)
    tick_labels = [f"{s:g}" for s in tick_sigmas]

    floor_df = _plot_noise_method_panels(method_axes, noise_df, tick_positions, tick_labels)
    for ds in DATASET_ORDER:
        handle, = method_axes[0].plot([], [], marker=DATASET_MARKERS[ds], color=DATASET_COLORS[ds],
                                      linewidth=2.0, markersize=4.8, label=ds)
        dataset_handles.append(handle)
    fig.legend(dataset_handles, DATASET_ORDER, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=len(DATASET_ORDER),
               frameon=True, facecolor="white", edgecolor="black", framealpha=0, borderpad=0.55, fontsize=8.6)
    # fig.suptitle("Noise sensitivity", y=0.98, fontsize=15, weight="semibold")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.25)
    
    return fig, floor_df
