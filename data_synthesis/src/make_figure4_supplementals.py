from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_diabetes
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = ROOT / "data_synthesis"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from models.bootstrap import sample_bootstrap
from models.cvae import sample_cvae
from models.gmm import sample_gmm
from models.iid_columnwise import sample_columnwise
from src.data import load_rdata_xy_names
from src.figure4_neighborhood import (
    plot_figure4_edge_status_matrices,
    plot_supplemental_edge_status_matrices,
)
from util.config import Config


SEED = 42
DATASET_ORDER = ["HIV", "Breast Cancer", "Diabetes"]
METHOD_ORDER = ["Bootstrap", "Column-wise", "GMM", "CVAE"]
FIGURE4_ALPHAS = {"HIV": 0.504, "Breast Cancer": 0.502, "Diabetes": 0.0159}


def _to_numpy_X(X):
    return np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float32)


def _to_numpy_y(y):
    y = np.asarray(y.values if hasattr(y, "values") else y).reshape(-1)
    if y.dtype.kind in {"i", "u", "b"}:
        return y.astype(int)
    return LabelEncoder().fit_transform(y).astype(int)


def load_hiv():
    X, y, names = load_rdata_xy_names(PKG_ROOT / "data" / "allSyntheticData.RData")
    return {"dataset": "HIV", "X": _to_numpy_X(X), "y": _to_numpy_y(y), "feature_names": names}


def load_breast():
    raw = load_breast_cancer()
    return {
        "dataset": "Breast Cancer",
        "X": _to_numpy_X(raw.data),
        "y": _to_numpy_y(raw.target),
        "feature_names": list(raw.feature_names),
    }


def load_diabetes_dataset():
    try:
        raw = fetch_openml("diabetes", version=1, as_frame=False)
        return {
            "dataset": "Diabetes",
            "X": _to_numpy_X(raw.data),
            "y": _to_numpy_y(raw.target),
            "feature_names": list(raw.feature_names),
        }
    except Exception as exc:
        print(f"[fallback] OpenML diabetes unavailable ({exc}); using sklearn diabetes target median split.")
        raw = load_diabetes()
        y = (raw.target >= np.median(raw.target)).astype(int)
        return {
            "dataset": "Diabetes",
            "X": _to_numpy_X(raw.data),
            "y": y.astype(int),
            "feature_names": list(raw.feature_names),
        }


def load_datasets():
    return {
        "HIV": load_hiv(),
        "Breast Cancer": load_breast(),
        "Diabetes": load_diabetes_dataset(),
    }


def class_counts(y):
    y = np.asarray(y, dtype=int)
    return int(np.sum(y == 0)), int(np.sum(y == 1))


def sample_synthetic(data, method, seed=SEED, cvae_epochs=200):
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=int)
    n0, n1 = class_counts(y)
    if method == "Bootstrap":
        return sample_bootstrap(X, y, n0, n1, seed=seed)
    if method == "Column-wise":
        return sample_columnwise(X, y, n0, n1, seed=seed)
    if method == "GMM":
        return sample_gmm(X, y, n0, n1, seed=seed)
    if method == "CVAE":
        cfg = Config(seed=seed, epochs=cvae_epochs)
        return sample_cvae(X, y, n0, n1, seed=seed, cfg=cfg, verbose=False)
    raise ValueError(f"Unknown method: {method}")


def build_precision_inputs(datasets, seed=SEED, cvae_epochs=200):
    real_data = {}
    synthetic_data = {}
    feature_names = {}
    for dataset in DATASET_ORDER:
        data = datasets[dataset]
        real_data[dataset] = np.asarray(data["X"], dtype=np.float64)
        feature_names[dataset] = list(data["feature_names"])
        synthetic_data[dataset] = {}
        for method in METHOD_ORDER:
            print(f"[generate] {dataset} - {method} seed={seed}")
            X_syn, _ = sample_synthetic(data, method, seed=seed, cvae_epochs=cvae_epochs)
            synthetic_data[dataset][method] = np.asarray(X_syn, dtype=np.float64)
    return real_data, synthetic_data, feature_names


def verify_generator_seeds(datasets, seed=SEED, cvae_epochs=3):
    rows = []
    for dataset in DATASET_ORDER:
        data = datasets[dataset]
        for method in METHOD_ORDER:
            X1, y1 = sample_synthetic(data, method, seed=seed, cvae_epochs=cvae_epochs)
            X2, y2 = sample_synthetic(data, method, seed=seed, cvae_epochs=cvae_epochs)
            X3, _ = sample_synthetic(data, method, seed=seed + 1, cvae_epochs=cvae_epochs)
            same_seed_equal = bool(np.array_equal(X1, X2) and np.array_equal(y1, y2))
            different_seed_changed = bool(not np.array_equal(X1, X3))
            rows.append({
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "same_seed_equal": same_seed_equal,
                "different_seed_changed": different_seed_changed,
                "shape": tuple(np.asarray(X1).shape),
                "class_counts": class_counts(y1),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Export Figure 4 supplemental edge-status matrices.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cvae-epochs", type=int, default=200)
    parser.add_argument("--verify-epochs", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PKG_ROOT / "notebooks" / "revision_exports",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets()

    seed_checks = verify_generator_seeds(
        datasets,
        seed=args.seed,
        cvae_epochs=args.verify_epochs,
    )
    print(seed_checks.to_string(index=False))
    if not seed_checks["same_seed_equal"].all():
        raise RuntimeError("At least one generator failed same-seed determinism.")

    real_data, synthetic_data, feature_names = build_precision_inputs(
        datasets,
        seed=args.seed,
        cvae_epochs=args.cvae_epochs,
    )

    main_result = plot_figure4_edge_status_matrices(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_names,
        alphas=FIGURE4_ALPHAS,
        dataset_order=DATASET_ORDER,
        method_order=METHOD_ORDER,
        exemplar_ds="HIV",
        save_path=args.output_dir / "figure_4_edge_status_matrices.png",
    )

    all_metrics = [main_result.metrics.assign(figure_dataset="HIV")]
    for dataset in ["Breast Cancer", "Diabetes"]:
        result = plot_supplemental_edge_status_matrices(
            real_data=real_data,
            synthetic_data=synthetic_data,
            feature_names=feature_names,
            alphas=FIGURE4_ALPHAS,
            dataset_order=DATASET_ORDER,
            method_order=METHOD_ORDER,
            exemplar_ds=dataset,
            save_path=args.output_dir / f"supplemental_figure4_{dataset.lower().replace(' ', '_')}_edge_status_matrices.png",
        )
        all_metrics.append(result.metrics.assign(figure_dataset=dataset))

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(args.output_dir / "figure4_structural_metrics.csv", index=False)
    seed_checks.to_csv(args.output_dir / "figure4_seed_checks.csv", index=False)
    print(f"Saved Figure 4 and supplementals to {args.output_dir}")


if __name__ == "__main__":
    main()
