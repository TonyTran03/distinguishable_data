from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    plot_edge_status_examples,
    plot_figure4_cluster_summary_grid,
    plot_figure4_edge_status_matrices,
    plot_figure4_tsne_edge_supplement,
)
from util.config import Config


SEED = 42
DATASET_ORDER = ["HIV", "Breast Cancer", "Diabetes"]
SUPPLEMENTAL_DATASETS = ["Breast Cancer", "Diabetes"]
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


def export_figure4_supplemental_figures(
    real_data,
    synthetic_data,
    feature_names,
    output_dir=None,
    seed=SEED,
):
    output_dir = Path(output_dir or PKG_ROOT / "notebooks" / "revision_exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    tsne_output_dir = output_dir / "t-SNE analysis"
    tsne_output_dir.mkdir(parents=True, exist_ok=True)
    exported_figures = []

    examples_path = output_dir / "supplemental_figure_s1_edge_status_examples.png"
    examples_result = plot_edge_status_examples(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_names,
        alphas=FIGURE4_ALPHAS,
        dataset_order=DATASET_ORDER,
        method_order=METHOD_ORDER,
        exemplar_ds="HIV",
        save_path=examples_path,
    )
    exported_figures.append({
        "section": "Supplement",
        "dataset": "HIV",
        "figure": "Edge-status examples",
        "path": str(examples_path),
    })

    all_metrics = [
        examples_result.metrics.assign(figure_dataset="HIV examples"),
    ]
    results = {"HIV examples": examples_result}
    plt.close(examples_result.fig)
    hiv_matrices_path = output_dir / "figure4_hiv_edge_status_matrices.png"
    hiv_matrices_result = plot_figure4_edge_status_matrices(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_names,
        alphas=FIGURE4_ALPHAS,
        dataset_order=DATASET_ORDER,
        method_order=METHOD_ORDER,
        exemplar_ds="HIV",
        save_path=hiv_matrices_path,
    )
    exported_figures.append({
        "section": "Main text",
        "dataset": "HIV",
        "figure": "Graphical Lasso edge-status matrices A-D",
        "path": str(hiv_matrices_path),
    })
    all_metrics.append(hiv_matrices_result.metrics.assign(figure_dataset="HIV A-D matrices"))
    results["HIV A-D matrices"] = hiv_matrices_result
    plt.close(hiv_matrices_result.fig)
    cluster_summary_path = output_dir / "figure4_cluster_summary_all_datasets.png"
    cluster_summary_result = plot_figure4_cluster_summary_grid(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_names,
        alphas=FIGURE4_ALPHAS,
        dataset_order=["HIV", "Diabetes", "Breast Cancer"],
        method_order=METHOD_ORDER,
        seed=seed,
        save_path=cluster_summary_path,
    )
    exported_figures.append({
        "section": "Main text",
        "dataset": "All datasets",
        "figure": "Cluster-level t-SNE summaries",
        "path": str(cluster_summary_path),
    })
    all_metrics.append(cluster_summary_result.metrics.assign(figure_dataset="all dataset I-K cluster summaries"))
    results["all dataset I-K cluster summaries"] = cluster_summary_result
    plt.close(cluster_summary_result.fig)
    for dataset in DATASET_ORDER:
        tsne_path = tsne_output_dir / f"supplemental_tsne_analysis_{dataset.lower().replace(' ', '_')}.png"
        result = plot_figure4_tsne_edge_supplement(
            real_data=real_data,
            synthetic_data=synthetic_data,
            feature_names=feature_names,
            alphas=FIGURE4_ALPHAS,
            dataset_order=DATASET_ORDER,
            method_order=METHOD_ORDER,
            exemplar_ds=dataset,
            seed=seed,
            save_path=tsne_path,
        )
        exported_figures.append({
            "section": "Supplement: t-SNE analysis",
            "dataset": dataset,
            "figure": "Graphical Lasso t-SNE edge overlays E-H",
            "path": str(tsne_path),
        })
        all_metrics.append(result.metrics.assign(figure_dataset=f"{dataset} t-SNE analysis"))
        results[f"{dataset} t-SNE analysis"] = result
        plt.close(result.fig)

        tsne_cosine_path = tsne_output_dir / f"supplemental_tsne_analysis_{dataset.lower().replace(' ', '_')}_cosine_clusters.png"
        cosine_result = plot_figure4_tsne_edge_supplement(
            real_data=real_data,
            synthetic_data=synthetic_data,
            feature_names=feature_names,
            alphas=FIGURE4_ALPHAS,
            dataset_order=DATASET_ORDER,
            method_order=METHOD_ORDER,
            exemplar_ds=dataset,
            seed=seed,
            cluster_metric="cosine",
            cluster_linkage="average",
            save_path=tsne_cosine_path,
        )
        exported_figures.append({
            "section": "Supplement: t-SNE analysis",
            "dataset": dataset,
            "figure": "Graphical Lasso t-SNE edge overlays E-H with cosine cluster distance",
            "path": str(tsne_cosine_path),
        })
        all_metrics.append(cosine_result.metrics.assign(figure_dataset=f"{dataset} t-SNE analysis cosine clusters"))
        results[f"{dataset} t-SNE analysis cosine clusters"] = cosine_result
        plt.close(cosine_result.fig)

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(output_dir / "figure4_structural_metrics.csv", index=False)
    pd.DataFrame(exported_figures).to_csv(output_dir / "figure4_exported_figures.csv", index=False)
    return results, metrics


def export_figure4_supplementals(
    output_dir=None,
    seed=SEED,
    cvae_epochs=200,
    verify_epochs=3,
    datasets=None,
):
    output_dir = Path(output_dir or PKG_ROOT / "notebooks" / "revision_exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets() if datasets is None else datasets

    seed_checks = verify_generator_seeds(
        datasets,
        seed=seed,
        cvae_epochs=verify_epochs,
    )
    print(seed_checks.to_string(index=False))
    if not seed_checks["same_seed_equal"].all():
        raise RuntimeError("At least one generator failed same-seed determinism.")

    real_data, synthetic_data, feature_names = build_precision_inputs(
        datasets,
        seed=seed,
        cvae_epochs=cvae_epochs,
    )

    results, metrics = export_figure4_supplemental_figures(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_names,
        output_dir=output_dir,
        seed=seed,
    )
    seed_checks.to_csv(output_dir / "figure4_seed_checks.csv", index=False)
    print(f"Saved Figure 4 supplementals to {output_dir}")
    return results, metrics, seed_checks


def main():
    parser = argparse.ArgumentParser(description="Export Figure 4 supplemental panels.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cvae-epochs", type=int, default=200)
    parser.add_argument("--verify-epochs", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PKG_ROOT / "notebooks" / "revision_exports",
    )
    args = parser.parse_args()
    export_figure4_supplementals(
        output_dir=args.output_dir,
        seed=args.seed,
        cvae_epochs=args.cvae_epochs,
        verify_epochs=args.verify_epochs,
    )


if __name__ == "__main__":
    main()
