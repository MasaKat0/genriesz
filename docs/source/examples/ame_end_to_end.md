# AME end-to-end (SQ / UKL / BP)

Notebook:

- `notebooks/AME_end_to_end.ipynb`

This notebook demonstrates:

- how to estimate an **Average Marginal Effect (AME)** with `grr_ame`,
- how to specify a polynomial basis on the regressor `X`,
- how to compare **SQ-Riesz / UKL-Riesz / BP-Riesz** and multiple `omega` values for BP,
- how to sweep over multiple regularization norms and strengths,
- and how to report **RA / RW / ARW / TMLE** with inference.

Because the outcome is unbounded in this example, the notebook uses **Gaussian TMLE**.
