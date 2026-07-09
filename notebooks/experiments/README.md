# GRR publication experiment notebooks

These notebooks contain the publication-scale experiments for the Generalized Riesz Regression manuscript.

The exploratory fast-mode code, synthetic real-data fallbacks, and external plotting/table helper module have been removed. Each notebook contains its own table and plotting code so titles, axis labels, method labels, colors, line styles, and font sizes can be edited directly in the notebook. ATE and ATT results are displayed in separate tables and separate figures, not pooled into a single table or axis.

Run from the repository root:

```bash
pip install -e .
pip install -r notebooks/experiments/requirements.txt
jupyter notebook notebooks/experiments
```

The IHDP notebook uses the npci-format files under `notebooks/experiments/data/ihdp`. The Lalonde notebook downloads and caches the public MatchIt/Rdatasets Lalonde CSV if it is not already present. No fallback data are generated.


Display convention: ATE and ATT are never placed in the same table or the same figure. Each notebook filters by `estimand` and then displays a separate table or plot for each target.

Stability convention: KL-type branchwise generators use estimand-specific shift constants. ATE uses `C=1`, while ATT uses a smaller shift because the ATT control-branch Riesz representer can have magnitude below one. This avoids the artificial weight explosions that occur when an ATT experiment is forced into an ATE-style `|alpha|>1` domain.


Note on the score-guided balancing appendix: the current high-level `grr_att` wrapper targets the full ATT effect functional. A pure covariate-balancing ATT counterfactual-mean variant requires a separate functional, so that unsupported combination is skipped rather than plotted as a degenerate result.


The notebooks use positive regularization grids and a finite-overlap guard for the UKL, BKL, BP, and propensity-index fits. The guard is defined directly in each notebook so it can be edited together with the plotting code.

Recent numerical fixes:

- `06_appendix_model_variation.ipynb` now uses the defined `BASIS_KINDS` grid and includes an explicit notebook-local `fit_matching_ate` function for the ATE-only nearest-neighbor matching baseline.
- The score-guided appendix fixes the loss at UKL-Riesz and uses a separate regressor-basis outcome model. This avoids the degenerate `alpha=0` solution that occurs when an unconstrained SQ loss is combined with covariate-only ATE features.
- The Zhao/Kang--Schafer appendix now reports balance-path diagnostics rather than treatment-effect MSE for that subsection. This follows the purpose of Zhao's Figure 1 and avoids displaying unstable outcome estimates from deliberately misspecified early-step models.
- The weak-overlap synthetic DGP is kept nonlinear but no longer uses extremely heavy-tailed covariates. Propensities are clipped to `[0.05, 0.95]` to make the comparison about loss-link behavior rather than rare numerical outliers.
