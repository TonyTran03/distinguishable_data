# Real vs Synthetic Data Investigation

Generated from `data_synthesis/notebooks/make_main_figures_revision.ipynb` and the same `src.revision` helper modules used by that notebook.

## Output Inventory

All plot outputs are in this directory:

- `figure2_six_panel_pca_auc.png`
- `figure2_nine_panel_pca_auc_trends.png`
- `figure3_main_metric_summary_table.png`
- `figure3_metric_panels.png`
- `figure4_hiv_supplemental_HIV examples.png`
- `figure4_hiv_edge_status.png`
- `figure4_breast_cancer_edge_status.png`
- `figure4_diabetes_edge_status.png`
- `figure4_all_dataset_cluster_summary_labeled.png`
- `figure4_all_dataset_cluster_summary_cosine_average.png`
- `figure4_hiv_structural_deviation_panel.png`
- `figure4_hiv_tsne_analysis_supplement.png`
- `figure4_hiv_tsne_analysis_supplement_cosine_average.png`
- `figure4_breast_cancer_tsne_analysis_supplement.png`
- `figure4_breast_cancer_tsne_analysis_supplement_cosine_average.png`
- `figure4_diabetes_tsne_analysis_supplement.png`
- `figure4_diabetes_tsne_analysis_supplement_cosine_average.png`
- `figure5_noise_sensitivity.png`
- `figure6_reverse_ablation_all_datasets.png`
- `figure6_reverse_ablation_ac.png`

The supporting numeric exports include `dataset_summary.csv`, `figure2_auc_runs.csv`, `figure3_metric_table.csv`, `figure4_all_edge_status_metrics.csv`, `feature_correlation_diagnostics.csv`, `figure5_noise_sensitivity.csv`, and `figure6_reverse_ablation.csv`.

## Dataset Attributes

| Dataset | Samples | Features | n / p | Class 0 | Class 1 | PC1+PC2 variance | Mean abs. real feature corr. |
|---|---:|---:|---:|---:|---:|---:|---:|
| HIV | 91 | 63 | 1.44 | 23 | 68 | 0.440 | 0.291 |
| Breast Cancer | 569 | 30 | 18.97 | 212 | 357 | 0.632 | 0.395 |
| Diabetes | 768 | 8 | 96.00 | 500 | 268 | 0.478 | 0.172 |

This matters. HIV is the high-risk regime: many features relative to samples, strong class imbalance, and enough correlation structure for synthetic generators to accidentally preserve marginal behavior while failing conditional structure. Breast Cancer is easier statistically because it has many more samples per feature and strong low-dimensional geometry. Diabetes is low-dimensional and sample-rich, so marginal and pairwise statistics are easier to preserve, but utility can still fail.

## Main Patterns

The discriminator AUC results show that visual overlap in PCA is not enough. AUC near 0.5 means real and synthetic are hard to distinguish; AUC near 1.0 means a classifier can nearly perfectly separate them.

| Dataset | Bootstrap | Column-wise | GMM | CVAE |
|---|---:|---:|---:|---:|
| HIV AUC | 0.710 | 0.855 | 1.000 | 0.995 |
| Breast Cancer AUC | 0.710 | 1.000 | 0.992 | 0.824 |
| Diabetes AUC | 0.717 | 0.793 | 0.978 | 0.994 |

Bootstrap is consistently the least distinguishable, but that is partly because it reuses real observations or close neighbors. It preserves realism, but with limited novelty and privacy value. GMM and CVAE often create synthetic data that supports downstream classification, yet the origin discriminator sees them easily. That is the central tension in the notebook: utility can look strong while the synthetic distribution is still detectably artificial.

## Hidden Correlation Structure

Pairwise feature-correlation diagnostics make the generator differences explicit.

| Dataset | Method | Corr-profile r | Mean abs. corr. delta | Real mean abs. corr. | Synthetic mean abs. corr. |
|---|---|---:|---:|---:|---:|
| HIV | Bootstrap | 0.948 | 0.064 | 0.291 | 0.288 |
| HIV | Column-wise | 0.111 | 0.300 | 0.291 | 0.094 |
| HIV | GMM | 0.869 | 0.097 | 0.291 | 0.295 |
| HIV | CVAE | 0.817 | 0.168 | 0.291 | 0.375 |
| Breast Cancer | Bootstrap | 0.993 | 0.026 | 0.395 | 0.396 |
| Breast Cancer | Column-wise | 0.672 | 0.214 | 0.395 | 0.225 |
| Breast Cancer | GMM | 0.930 | 0.078 | 0.395 | 0.330 |
| Breast Cancer | CVAE | 0.986 | 0.060 | 0.395 | 0.447 |
| Diabetes | Bootstrap | 0.977 | 0.030 | 0.172 | 0.182 |
| Diabetes | Column-wise | -0.036 | 0.153 | 0.172 | 0.048 |
| Diabetes | GMM | 0.983 | 0.030 | 0.172 | 0.163 |
| Diabetes | CVAE | 0.823 | 0.117 | 0.172 | 0.249 |

Column-wise sampling is the clearest failure mode. It can preserve univariate feature distributions but strips cross-feature dependence. In HIV, the synthetic mean absolute correlation collapses from 0.291 to 0.094; in Diabetes it collapses from 0.172 to 0.048. This is why marginal fidelity is a weak target for synthetic data.

CVAE shows a different failure mode: it often retains broad geometry but over-densifies or distorts dependencies. HIV CVAE raises mean absolute correlation from 0.291 to 0.375 and has a larger mean absolute correlation delta than GMM or Bootstrap. Breast Cancer CVAE also increases mean absolute correlation from 0.395 to 0.447. That suggests latent generators can create plausible-looking clusters while introducing over-regularized hidden structure.

## Conditional-Dependence Structure

Graphical Lasso edge recovery is the strongest evidence that real and synthetic data differ beyond visible PCA.

| Dataset | Method | Real edges | Synthetic edges | Edge recovery | Synthetic-only rate | Frobenius deviation |
|---|---|---:|---:|---:|---:|---:|
| HIV | Bootstrap | 186 | 191 | 0.812 | 0.209 | 1.184 |
| HIV | Column-wise | 186 | 0 | 0.000 | NA | 3.162 |
| HIV | GMM | 186 | 202 | 0.790 | 0.272 | 1.276 |
| HIV | CVAE | 186 | 335 | 0.742 | 0.588 | 2.508 |
| Breast Cancer | Bootstrap | 97 | 95 | 0.928 | 0.053 | 0.379 |
| Breast Cancer | Column-wise | 97 | 30 | 0.278 | 0.100 | 2.413 |
| Breast Cancer | GMM | 97 | 78 | 0.794 | 0.013 | 0.922 |
| Breast Cancer | CVAE | 97 | 114 | 0.948 | 0.193 | 0.751 |
| Diabetes | Bootstrap | 21 | 24 | 1.000 | 0.125 | 0.370 |
| Diabetes | Column-wise | 21 | 19 | 0.667 | 0.263 | 1.662 |
| Diabetes | GMM | 21 | 25 | 0.952 | 0.200 | 0.283 |
| Diabetes | CVAE | 21 | 25 | 0.952 | 0.200 | 2.851 |

The HIV case is the strongest argument for structural evaluation. Column-wise produces no conditional-dependence graph. CVAE recovers many real edges but also creates 335 synthetic edges, with a synthetic-only rate of 0.588. That means the generator invents a large amount of conditional structure not supported by the real data.

Diabetes shows why edge recovery alone is insufficient: GMM and CVAE both recover 95.2% of real edges and have 25 synthetic edges, but CVAE has much larger Frobenius deviation. The same edge count can hide different edge weights or partial-correlation magnitudes.

## Hidden Feature Modules

The t-SNE/cluster summaries suggest real feature structure is not simply a small number of clean independent modules.

Under Euclidean single-linkage clustering at requested k=7:

| Dataset | Features | Largest cluster | Cluster sizes | Silhouette |
|---|---:|---:|---|---:|
| HIV | 63 | 57 | 57,1,1,1,1,1,1 | 0.061 |
| Breast Cancer | 30 | 18 | 18,6,2,1,1,1,1 | 0.101 |
| Diabetes | 8 | 2 | 2,1,1,1,1,1,1 | 0.097 |

Under cosine average-linkage at requested k=7:

| Dataset | Features | Largest cluster | Cluster sizes | Silhouette |
|---|---:|---:|---|---:|
| HIV | 63 | 15 | 15,15,14,8,6,3,2 | 0.194 |
| Breast Cancer | 30 | 8 | 8,6,5,4,3,2,2 | 0.335 |
| Diabetes | 8 | 2 | 2,1,1,1,1,1,1 | 0.171 |

This means the hidden structure is distance-sensitive. HIV looks like one dominant backbone under Euclidean single-linkage, but resolves into multiple modules under cosine average-linkage. For synthetic data evaluation, this argues against a single graph or embedding score. A generator can pass one structural view while failing another.

## Noise And Ablation Arguments

The noise-sensitivity figure asks whether obvious separability survives perturbation. It does. Minimum AUC under the tested noise settings remains high for many methods: HIV GMM 0.958, HIV CVAE 0.915, Breast Cancer GMM 0.979, Breast Cancer Column-wise 0.939, Diabetes CVAE 0.963, and Diabetes GMM 0.943. These are not tiny visual artifacts; the classifier can still find origin signal after noise injection.

The reverse-ablation figure asks where the origin signal lives. The important interpretation is that distinguishability is feature-concentrated. If removing top discriminator-ranked features sharply lowers AUC, then synthetic artifacts are not uniformly spread; they live in particular coordinates or correlated feature groups. For future synthetic data, this supports targeted repair: audit feature groups, conditional edges, and interaction-heavy variables rather than only global distribution scores.

## Overall Conclusions

1. No generator is uniformly best. Bootstrap looks most real by AUC and correlation preservation, but it has weak novelty. GMM often preserves correlation and conditional structure better than CVAE in small or low-dimensional settings, but it remains highly distinguishable. CVAE can preserve task utility while creating artificial hidden structure.

2. The small-sample, high-feature HIV setting is the decisive stress test. With only 91 samples and 63 features, generators face a hard covariance-estimation problem. Synthetic data may match labels and marginal features but still fail the dependency graph.

3. Marginal distribution matching is not enough. Column-wise synthesis is the demonstration case: it can keep per-feature distributions while destroying feature dependence and conditional edges.

4. Utility is not fidelity. Several synthetic datasets have strong TSTR F1 and low utility gaps while still having discriminator AUC near 1.0. Future synthetic-data claims should report utility, origin separability, pairwise correlation preservation, and conditional-dependence preservation together.

5. Structural metrics should become standard. Pairwise correlation, partial-correlation edge recovery, synthetic-only edge rate, graph-weight deviation, feature-cluster stability, and ablation-localized origin signal reveal failures that PCA and headline utility metrics miss.

6. The next generation of synthetic data should optimize for constrained realism: preserve useful downstream signal, avoid record copying, and explicitly penalize invented or erased dependency structure. For biomedical and tabular scientific data, the decisive question is not only whether synthetic samples look plausible, but whether they preserve the conditional relationships researchers would use for inference.

## Caveats

The run emitted Graphical Lasso convergence warnings for repeated calls on at least one setting. The outputs were still generated, but any formal manuscript claim should either stabilize those fits with tuned solver parameters or report sensitivity to the Graphical Lasso regularization path.
