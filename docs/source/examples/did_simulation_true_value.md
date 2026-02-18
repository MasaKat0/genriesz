# DID simulation with true value (SQ / UKL / BP)

Notebook:

- `notebooks/DID_simulation_true_value.ipynb`

This notebook implements DID as **ATT on the differenced outcome** (panel setting):

- define $\Delta Y = Y_1 - Y_0$,
- run `grr_did(X, Y0=..., Y1=...)`.

It includes a simulation with a Monte Carlo **true value**, and compares:

- **SQ-Riesz / UKL-Riesz / BP-Riesz**,
- multiple regularization norms and strengths,
- multiple BP power parameters (`omega`),
- and the estimators **RA / RW / ARW / TMLE**.

As in other treatment-effect examples, UKL/BP are run with a branch function that selects the sign
by treatment status.
