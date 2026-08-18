"""Shared setup, data loading, styling, and small plotting helpers."""

from pathlib import Path
import sys
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyreadr
from IPython.display import display, Markdown
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import chi2, entropy, ks_2samp, ttest_ind
from sklearn.datasets import load_breast_cancer, fetch_openml, load_diabetes
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

pkg_root = Path(__file__).resolve().parents[2]
if str(pkg_root) not in sys.path:
    sys.path.insert(0, str(pkg_root))

from models.bootstrap import sample_bootstrap
from models.cvae import sample_cvae, train_cvae, sample_trained_cvae
from models.gmm import AIC_COMPONENTS_BY_DATASET, sample_gmm
from models.iid_columnwise import sample_columnwise
from util.config import Config
from rfhack.core.rf_wrapper import RFWrapper

def _find_repo_root(start):
    for path in [start, *start.parents]:
        if (path / "data_synthesis" / "data").exists():
            return path
    return start.parents[4]

repo_root = _find_repo_root(Path(__file__).resolve())

CACHE_DIR = repo_root / "data_synthesis" / "notebooks" / "revision_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# The main revision notebook uses the conventional CVAE baseline: raw features
# standardized in the loader and a shared standard-normal latent prior.
MAIN_CVAE_X_TRANSFORM = "none"
MAIN_CVAE_LATENT_PRIOR = "normal"

REVISION_RUN_MODE = "final"

RUN_MODE = REVISION_RUN_MODE

RUN_PRESETS = {
    "preview": {
        "CVAE_EPOCHS": 10,
        "AUC_REPEATS": 5,
        "TSTR_REPEATS": 3,
        "ABLATION_REPEATS": 3,
        "NOISE_REPEATS": 2,
        "PROBE_SIGMAS": [0.0, 0.2, 0.5, 1.0],
    },
    "final": {
        "CVAE_EPOCHS": 200,
        "AUC_REPEATS": 50,
        "TSTR_REPEATS": 20,
        "ABLATION_REPEATS": 20,
        "NOISE_REPEATS": 5,
        "PROBE_SIGMAS": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
    },
}

if RUN_MODE not in RUN_PRESETS:
    raise ValueError(f"RUN_MODE must be one of {list(RUN_PRESETS)}")

_run_cfg = RUN_PRESETS[RUN_MODE]
CVAE_EPOCHS = _run_cfg["CVAE_EPOCHS"]
AUC_REPEATS = _run_cfg["AUC_REPEATS"]

TSTR_REPEATS = _run_cfg["TSTR_REPEATS"]

ABLATION_REPEATS = _run_cfg["ABLATION_REPEATS"]

NOISE_REPEATS = _run_cfg["NOISE_REPEATS"]

PROBE_SIGMAS = _run_cfg["PROBE_SIGMAS"]
NOISE_FRAC = 0.3

OVERRIDE_CVAE_EPOCHS = None

OVERRIDE_AUC_REPEATS = None

OVERRIDE_TSTR_REPEATS = None

OVERRIDE_ABLATION_REPEATS = None

OVERRIDE_NOISE_REPEATS = None

OVERRIDE_PROBE_SIGMAS = None

CVAE_EPOCHS = OVERRIDE_CVAE_EPOCHS or CVAE_EPOCHS
AUC_REPEATS = OVERRIDE_AUC_REPEATS or AUC_REPEATS
TSTR_REPEATS = OVERRIDE_TSTR_REPEATS or TSTR_REPEATS
ABLATION_REPEATS = OVERRIDE_ABLATION_REPEATS or ABLATION_REPEATS
NOISE_REPEATS = OVERRIDE_NOISE_REPEATS or NOISE_REPEATS
PROBE_SIGMAS = OVERRIDE_PROBE_SIGMAS or PROBE_SIGMAS

DATASET_ORDER = ["HIV", "Breast Cancer", "Diabetes"]

DATASET_COLORS = {"HIV": "#D62728", "Breast Cancer": "#1DB100", "Diabetes": "#0076BA"}

FIGURE_TEXT_COLOR = "#222222"
FIGURE_SANS_SERIF = ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"]
FIGURE_FONT_SCALE = 1.20
FIGURE_SIZE_SCALE = 1.18
FIGURE_LABEL_REPLACEMENTS = {"GMM-guided SMOTE": "GMM-SMOTE"}

# Keep notebook figures independent of machine-specific serif defaults.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FIGURE_SANS_SERIF,
    "mathtext.fontset": "dejavusans",
})


def apply_notebook_figure_style(fig):
    """Apply the shared typography and label alignment used in both notebooks."""
    if getattr(fig, "_full_page_notebook_style_applied", False):
        return fig

    width, height = fig.get_size_inches()
    fig.set_size_inches(width * FIGURE_SIZE_SCALE, height * FIGURE_SIZE_SCALE)

    for ax in fig.axes:
        tick_text = [tick.get_text() for tick in ax.get_xticklabels()]
        display_tick_text = tick_text.copy()
        for index, label in enumerate(display_tick_text):
            for old_label, new_label in FIGURE_LABEL_REPLACEMENTS.items():
                display_tick_text[index] = label.replace(old_label, new_label)
        if display_tick_text != tick_text:
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(display_tick_text)

    text_artists = list(fig.findobj(match=mpl.text.Text))
    for ax in fig.axes:
        for table in ax.tables:
            text_artists.extend(cell.get_text() for cell in table.get_celld().values())

    for text_artist in dict.fromkeys(text_artists):
        text_artist.set_fontfamily("DejaVu Sans")
        label = text_artist.get_text().strip()
        for old_label, new_label in FIGURE_LABEL_REPLACEMENTS.items():
            if old_label in label:
                label = label.replace(old_label, new_label)
                text_artist.set_text(label)
        if any(
            label == dataset or label.startswith(f"{dataset}:")
            for dataset in DATASET_ORDER
        ):
            text_artist.set_color(FIGURE_TEXT_COLOR)
        text_artist.set_fontsize(float(text_artist.get_fontsize()) * FIGURE_FONT_SCALE)

    for ax in fig.axes:
        for tick_label in ax.get_xticklabels():
            tick_label.set_horizontalalignment("center")
            if abs(float(tick_label.get_rotation())) > 1e-8:
                # Align the rotated bounding box below the tick.  Anchoring
                # rotation at the text center lets the upper half intrude into
                # the plotting area.
                tick_label.set_rotation_mode("default")
                tick_label.set_verticalalignment("top")
    fig._full_page_notebook_style_applied = True
    return fig

DATASET_MARKERS = {"HIV": "^", "Breast Cancer": "o", "Diabetes": "s"}

METHOD_ORDER = ["Bootstrap", "Column-wise", "GMM", "CVAE"]

METHOD_COLORS = {"Bootstrap": "#6A5ACD", "Column-wise": "#CC79A7", "GMM": "#009E73", "CVAE": "#D55E00"}

METHOD_PASTELS = {"Bootstrap": "#C7C2F4", "Column-wise": "#E8B4D2", "GMM": "#A8DEC9", "CVAE": "#F2B49B"}

REAL_BLUE = "#9ECAE1"

SYN_RED = "#F4A6A6"

REAL_GREY = "#333434"

CVAE_ORANGE = "#D55E00"

LABEL_RED = "#C62828"

NEUTRAL = "#333434"

PASTEL_CORR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "pastel_corr", ["#7B5EA7", "#F7F7F7", "#58A4B0"]
)
ABS_CORR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "abs_corr", ["#FFFFFF", "#F4A6A6", "#C62828"]
)

METHODS = {
    "Bootstrap": sample_bootstrap,
    "Column-wise": sample_columnwise,
    "GMM": sample_gmm,
    "CVAE": sample_cvae,
}

def _to_numpy_X(X):
    return np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float32)

def _to_numpy_y(y):
    y = np.asarray(y.values if hasattr(y, "values") else y).reshape(-1)
    if y.dtype.kind in {"i", "u", "b"}:
        return y.astype(int)
    return LabelEncoder().fit_transform(y).astype(int)

def load_hiv():
    rdata_path = repo_root / "data_synthesis" / "data" / "allSyntheticData.RData"
    obj = pyreadr.read_r(str(rdata_path))
    x_df = obj["x"]
    y_df = obj["y"]
    return {
        "dataset": "HIV",
        "X": _to_numpy_X(x_df),
        "y": _to_numpy_y(y_df),
        "feature_names": list(x_df.columns),
    }

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
        print(f"[fallback] OpenML diabetes unavailable ({exc}); using sklearn diabetes binarized at median target.")
        raw = load_diabetes()
        y = (raw.target >= np.median(raw.target)).astype(int)
        return {
            "dataset": "Diabetes",
            "X": _to_numpy_X(raw.data),
            "y": y,
            "feature_names": list(raw.feature_names),
        }

LOADERS = {"HIV": load_hiv, "Breast Cancer": load_breast, "Diabetes": load_diabetes_dataset}

def load_datasets():
    out = {}
    for name in DATASET_ORDER:
        out[name] = LOADERS[name]()
    return out

def class_counts(y):
    y = np.asarray(y).astype(int)
    return int(np.sum(y == 0)), int(np.sum(y == 1))

def sample_synthetic(dataset, data, method, seed=SEED, cvae_epochs=CVAE_EPOCHS):
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=int)
    n0, n1 = class_counts(y)
    if method == "CVAE":
        cfg = Config(
            seed=seed,
            epochs=cvae_epochs,
            x_transform=MAIN_CVAE_X_TRANSFORM,
            latent_prior=MAIN_CVAE_LATENT_PRIOR,
        )
        return sample_cvae(X, y, n0, n1, seed=seed, cfg=cfg)
    if method == "GMM":
        return sample_gmm(
            X,
            y,
            n0,
            n1,
            seed=seed,
            n_components=AIC_COMPONENTS_BY_DATASET.get(dataset, 2),
        )
    return METHODS[method](X, y, n0, n1, seed=seed)

def standardize_pair(X_real, X_syn):
    scaler = StandardScaler()
    Xr = scaler.fit_transform(np.asarray(X_real, dtype=np.float64))
    Xs = scaler.transform(np.asarray(X_syn, dtype=np.float64))
    return Xr, Xs

def clean_axis(ax, grid_axis=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.8, alpha=0.45)
    else:
        ax.grid(False)

def add_panel_label(ax, label, color=NEUTRAL):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, ha="left", va="top", fontsize=18, weight="bold", color=color)

def add_confidence_ellipse(ax, xy, color, linestyle="-", linewidth=2.0):
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] < 3:
        return
    cov = np.cov(xy, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = np.sqrt(chi2.ppf(0.95, df=2))
    width, height = 2 * scale * np.sqrt(vals)
    ax.add_patch(Ellipse(xy.mean(axis=0), width=width, height=height, angle=angle,
                         facecolor="none", edgecolor=color, linewidth=linewidth, linestyle=linestyle))

datasets = None
summary = None

def initialize_datasets(loaded_datasets=None):
    """Load or register datasets used by notebook plotting helpers."""
    global datasets, summary
    datasets = load_datasets() if loaded_datasets is None else loaded_datasets
    summary = pd.DataFrame([
        {
            "dataset": name,
            "samples": len(data["y"]),
            "features": np.asarray(data["X"]).shape[1],
            "class_0": int((np.asarray(data["y"]) == 0).sum()),
            "class_1": int((np.asarray(data["y"]) == 1).sum()),
        }
        for name, data in datasets.items()
    ])
    return datasets, summary

def require_datasets():
    if datasets is None:
        initialize_datasets()
    return datasets
