# ScoreMatchingRiesz replication notebooks

This folder contains the replication code for **ScoreMatchingRiesz: Score Matching for Debiased Machine Learning and Policy Path Estimation** (ICML 2026). Reusable primitives are in `src/genriesz/`; data-generating processes, Monte Carlo loops, empirical-data construction, tables, and figures live directly in the notebooks here.

## Notebook index

| Paper item | Notebook | Contents |
|---|---|---|
| Section 9.1, Table 2(a), Figure 1(a) | `Experiments_Section_9_1_Table_2a_Figure_1a_AME.ipynb` | Baseline nonlinear Gaussian AME experiment. Methods: Data-SMR, Time-SMR, Riesz regression. |
| Section 9.1, Table 2(b), Figure 1(b) | `Experiments_Section_9_1_Table_2b_Figure_1b_APE.ipynb` | Baseline nonlinear Gaussian pushforward APE experiment at δ=1. Methods: Data-SMR, Time-SMR, Riesz regression. |
| Section 9.1, Figure 2 | `Figure_2_Policy_Path_Visualization.ipynb` | Full translation-policy path, pointwise intervals, true path, and local AME approximation. |
| Section 9.2, Table 3 and Appendix R | `Experiments_Section_9_2_Table_3_IHDP_ATE_Benchmark.ipynb` | IHDP semi-synthetic ATE benchmark over 100 training replications. Methods: Time-SMR, Joint-SMR, SQ-Riesz, BKL-Riesz, logistic MLE/AIPW. |
| Appendix P, Tables 5 and 6 | `Experiments_Appendix_P_Tables_5_6_HighDimensional_Pushforward.ipynb` | High-dimensional pushforward designs: sparse Gaussian and Gaussian-mixture random-feature cases. |
| Appendix P, Table 7 | `Experiments_Appendix_P_Table_7_NonPushforward_Stochastic_Policies.ipynb` | Non-pushforward stochastic-intervention experiment; Data-SMR is not applicable for APE here. |
| Appendix Q, Table 8 and Figures 4–6 | `Experiments_Appendix_Q_Table_8_Figures_4_6_LocalProjection_PolicyPaths.ipynb` | Local-projection policy-path application using public macro-finance data, HAC intervals, and horizon-wise path plots. |

## Running

From the repository root:

```bash
pip install -e ".[scorematchingriesz]"
pip install -r notebooks/scorematchingriesz/requirements.txt
jupyter notebook notebooks/scorematchingriesz
```

The default settings match the paper: 200 Monte Carlo trials for the main synthetic experiments, 100 IHDP replications, and the full horizon/shift grid for the local-projection application. To run a quick debugging pass, edit the relevant variables at the top of each notebook.

## Data

IHDP train/test NPZ files are under `data/ihdp/`. The local-projection notebook downloads macro-finance data from FRED and Yahoo Finance at runtime.
