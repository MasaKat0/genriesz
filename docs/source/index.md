# genriesz documentation

`genriesz` is a Python library around **Riesz representer** ideas for causal and semiparametric
estimation. This documentation currently focuses on **nearest-neighbor matching as LSIF / Riesz
regression** and its **local-polynomial extension**.

## References

If you use `genriesz` in academic work, please cite:

- **Kato (2026)**, *Riesz Representer Fitting under Bregman Divergence: A Unified Framework for Debiased Machine Learning* (arXiv:2601.07752).
  - Consolidates earlier related drafts: arXiv:2509.22122, arXiv:2510.26783, arXiv:2510.23534.
- **Kato (2026)**, *Nearest Neighbor Matching as Least Squares Density Ratio Estimation and Riesz Regression* (arXiv:2510.24433).

## Example notebooks

- [ATE end-to-end (SQ / UKL / BP)](examples/ate_end_to_end.md)
- [AME end-to-end (SQ / UKL / BP)](examples/ame_end_to_end.md)
- [ATT simulation with true value (SQ / UKL / BP)](examples/att_simulation_true_value.md)
- [DID simulation with true value (SQ / UKL / BP)](examples/did_simulation_true_value.md)
- [Lin et al. replication + Local-Polynomial NN–LSIF](examples/lin_et_al_local_polynomial_nn_lsif.md)

## Diagnostics

`genriesz` includes simple covariate-balance diagnostics for treatment-effect
estimands (ATE / ATT / DID), including a **Love plot** based on standardized
mean differences (SMDs).

- [Love plot / balance diagnostics](diagnostics.md)
