# Consolidated GRR experiment notebooks

The experiments are grouped by manuscript placement and purpose. Each analysis estimates both ATE and ATT whenever the implemented method supports both targets. If a method is target-specific, the executable notebook keeps the available target and records failures explicitly.

| Notebook | Placement | Description |
|---|---|---|
| `01_main_simulation_study.ipynb` | Main text | Three-DGP simulation with compatible loss--link pairs, incompatible loss--link pairs, and regularization paths. ATE and ATT are both reported. |
| `02_main_empirical_ihdp.ipynb` | Main text | IHDP semi-synthetic benchmark for ATE and ATT. |
| `03_main_empirical_lalonde_att.ipynb` | Main text | Lalonde NSW treated ATT benchmark, with ATE reported as sensitivity. |
| `04_appendix_simulation_crossfit.ipynb` | Appendix | Cross fitting versus no cross fitting for ATE and ATT. |
| `05_appendix_dimension_variation.ipynb` | Appendix | Dimension variation for ATE and ATT. |
| `06_appendix_model_variation.ipynb` | Appendix | RKHS, polynomial, random forest, nearest-neighbor matching, and random Fourier features. GRR models estimate ATE and ATT; nearest-neighbor matching is kept as an ATE-only baseline when ATT is unsupported. |
| `07_appendix_additional_experiments.ipynb` | Appendix | Zhao/Kang--Schafer balance path, kernel-GP basis mismatch, ACIC, HDMA, and NSW randomized checks, with ATE and ATT wherever possible. |

Each notebook is executable and displays tables and figures in the notebook. Plot titles, table titles, grid sizes, and figure settings are defined in cells inside each notebook so that manuscript formatting can be edited directly.

Default mode is `FAST_MODE=True` for smoke runs. Set `FAST_MODE=False` in each notebook for publication-scale grids.

Remote datasets are disabled in FAST_MODE by default to avoid hanging in offline environments. Set `DOWNLOAD_DATA=True` in empirical notebooks, or set `GRR_ALLOW_REMOTE_DATA=1`, to download public benchmark datasets.
