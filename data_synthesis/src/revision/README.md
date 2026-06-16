# Revision Figure Modules

Use these modules from notebooks instead of importing all revision helpers from
`src.main_figures_revision`. The figure modules contain the actual editable
plotting/stat code for each figure.

- `config.py`: shared constants, run mode, colors, and method order.
- `data_io.py`: dataset loading, synthetic sampling, and shared transforms.
- `cache.py`: cached compute entry points for expensive tables.
- `stats.py`: reusable metrics and statistical computations.
- `plot_utils.py`: shared axis and panel-label helpers.
- `display.py`: notebook display helpers.
- `figure2_auc.py`: Figure 2 PCA and separability AUC plots.
- `figure3_metric_summary.py`: Figure 3 metric tables and summary panels.
- `figure4_graphical_lasso.py`: Figure 4 Graphical Lasso workflow/export helpers.
- `figure4_graphical_lasso_plots.py`: lower-level Graphical Lasso graph, edge-status, t-SNE, and cluster plotting internals.
- `figure5_noise.py`: Figure 5 noise sensitivity helpers.
- `figure6_ablation.py`: Figure 6 reverse-ablation helpers.

`common.py` contains shared setup, data loading, and small plot helpers.
`src.main_figures_revision` and `_private/main_figures_revision_impl.py` are now
compatibility facades for older notebooks.

Figure 4 supplementals are exported from `figure4_graphical_lasso.py`; the old
standalone `make_figure4_supplementals.py` script has been folded into:

```python
from src.revision import figure4_graphical_lasso

results, metrics, seed_checks = figure4_graphical_lasso.export_figure4_supplementals(
    datasets=my_datasets,
    dataset_order=["Dataset A", "Dataset B"],
    alphas={"Dataset A": 0.1, "Dataset B": 0.02},
    exemplar_ds="Dataset A",
)
```
