# ATT simulation with true value (SQ / UKL / BP)

Notebook:

- `notebooks/ATT_simulation_true_value.ipynb`

This notebook provides a simulation with a Monte Carlo **true ATT**, and compares:

- **SQ-Riesz / UKL-Riesz / BP-Riesz**,
- multiple regularization norms (`l2`, `l1`, `l_p`) and strengths,
- multiple BP power parameters (`omega`),
- and the estimators **RA / RW / ARW / TMLE**.

The notebook uses a **branch function** for UKL/BP that selects the sign by treatment status.
