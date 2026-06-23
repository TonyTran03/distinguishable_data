"""Cache helpers for expensive revision computations."""

from src.revision.common import *
import importlib


def _cache_path(name):
    return CACHE_DIR / f"{name}_{RUN_MODE}.pkl"


def _read_cache(name):
    path = _cache_path(name)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"[cache] Could not read {path.name}: {exc}. Recomputing.")
        return None


def cache_status(names=("auc_runs", "metric_table", "ablation_df", "noise_df", "sample_size_auc", "rf_structure_rank_overlap", "refit_structural_ablation_sparse", "discriminator_importance")):
    """Return existence/readability status for notebook cache artifacts."""
    rows = []
    for name in names:
        path = _cache_path(name)
        row = {"name": name, "path": str(path), "exists": path.exists(), "readable": False, "error": ""}
        if path.exists():
            try:
                with path.open("rb") as f:
                    pickle.load(f)
                row["readable"] = True
            except Exception as exc:
                row["error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_cache(name, obj):
    path = _cache_path(name)
    with path.open("wb") as f:
        pickle.dump(obj, f)
    return obj


def get_auc_runs(force=False):
    cached = None if force else _read_cache("auc_runs")
    if cached is not None:
        return cached
    from src.revision.figure2_auc import compute_auc_run_table
    return _write_cache("auc_runs", compute_auc_run_table(require_datasets()))


def get_metric_table(force=False):
    cached = None if force else _read_cache("metric_table")
    if cached is not None:
        return cached
    from src.revision.figure3_metric_summary import build_metric_table
    auc_runs = get_auc_runs(force=force)
    return _write_cache("metric_table", build_metric_table(require_datasets(), auc_runs))


def get_reverse_ablation(force=False):
    from src.revision.figure6_ablation import get_reverse_ablation as _get_reverse_ablation
    return _get_reverse_ablation(force=force)


def get_noise_sensitivity(force=False):
    from src.revision.figure5_noise import get_noise_sensitivity as _get_noise_sensitivity
    return _get_noise_sensitivity(force=force)


def get_sample_size_auc(force=False):
    from src.revision.figure7_sample_size import get_sample_size_auc as _get_sample_size_auc
    return _get_sample_size_auc(force=force)


def get_rf_structure_rank_overlap(force=False):
    from src.revision import figure6_ablation
    figure6_ablation = importlib.reload(figure6_ablation)
    return figure6_ablation.get_rf_structure_rank_overlap(force=force)


def get_refit_structural_ablation(force=False):
    from src.revision import figure6_ablation
    figure6_ablation = importlib.reload(figure6_ablation)
    return figure6_ablation.get_refit_structural_ablation(force=force)


def get_discriminator_importance_table(force=False):
    from src.revision import figure6_ablation
    figure6_ablation = importlib.reload(figure6_ablation)
    return figure6_ablation.get_discriminator_importance_table(force=force)
