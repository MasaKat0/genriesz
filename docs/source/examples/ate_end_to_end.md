# ATE end-to-end (SQ / UKL / BP)

Notebook:

- `notebooks/ATE_end_to_end.ipynb`

This notebook demonstrates:

- how to estimate the **ATE** with `grr_ate`,
- how to choose a basis (polynomial + treatment interactions),
- how to compare **SQ-Riesz / UKL-Riesz / BP-Riesz**,
- how to sweep over multiple regularization norms (`l2`, `l1`, `l_p`) and strengths,
- and how to report **RA / RW / ARW / TMLE** with inference.

For UKL/BP we set a **branch function** that forces the positive branch for treated units and the
negative branch for control units, matching the sign structure of treatment-effect Riesz representers.
