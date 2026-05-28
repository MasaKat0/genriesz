Diagnostics and balance checks
===============================

After fitting a GRR model, several diagnostic quantities help assess whether the Riesz representer
:math:`\hat{\alpha}` is well-behaved and whether the resulting estimator is reliable.

Alpha (Riesz representer) diagnostics
--------------------------------------

The fitted Riesz representer :math:`\hat{\alpha}_i = \hat{\alpha}(X_i)` appears as the reweighting
factor in the RW and ARW estimators.  Large or erratic values indicate numerical instability or
insufficient regularisation.  Three summary statistics are available on
:class:`~genriesz.FunctionalEstimate`:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Description
   * - ``alpha_abs_mean``
     - Mean of :math:`|\hat{\alpha}_i|` over all observations.
       For the ATE this equals the average inverse propensity weight.
   * - ``alpha_abs_p95``
     - 95th percentile of :math:`|\hat{\alpha}_i|`.
       A large gap between ``alpha_abs_p95`` and ``alpha_abs_mean`` signals heavy-tailed weights.
   * - ``alpha_abs_max``
     - Maximum of :math:`|\hat{\alpha}_i|`.
       A very large maximum (e.g., > 50 for ATE on balanced data) suggests near-positivity
       violations or insufficient regularisation.

**Practical thresholds (ATE)**

- ``alpha_abs_max < 10``: weights are well-behaved.
- ``alpha_abs_max`` in 10–50: mild extrapolation; increase ``riesz_lam`` or widen the basis.
- ``alpha_abs_max > 50``: strong positivity concerns; consider trimming or a more flexible basis.

Example::

   res = grr_ate(X=X, Y=Y, basis=phi, generator=gen, cross_fit=True)
   print("mean |α|:", res.arw.alpha_abs_mean)
   print("p95  |α|:", res.arw.alpha_abs_p95)
   print("max  |α|:", res.arw.alpha_abs_max)


Covariate balance (Love plot)
------------------------------

For treatment-effect estimands (ATE, ATT, DID) the weighted covariate distribution should be
balanced between treated and control units.  :class:`~genriesz.FunctionalEstimate` exposes three
balance statistics on the ARW estimate:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``max_abs_smd_weighted``
     - Maximum absolute standardised mean difference (SMD) across all covariate columns,
       after applying the estimated weights.  The unweighted benchmark is also stored
       as ``max_abs_smd_unweighted``.
   * - ``ess_treated``
     - Effective sample size (ESS) of the weighted treated group,
       :math:`\bigl(\sum_i w_i\bigr)^2 / \sum_i w_i^2` where :math:`w_i = \hat{\alpha}_i`
       for treated units.
   * - ``ess_control``
     - Corresponding ESS for the control group.

A **Love plot** visualises the per-covariate SMD before and after weighting::

   import matplotlib.pyplot as plt

   res = grr_ate(X=X, Y=Y, basis=phi, generator=gen, cross_fit=True)
   est = res.arw

   smd_uw = est.smd_unweighted   # dict: covariate name → unweighted SMD
   smd_w  = est.smd_weighted     # dict: covariate name → weighted SMD

   names = list(smd_uw.keys())
   fig, ax = plt.subplots(figsize=(6, 0.4 * len(names) + 1))
   ax.scatter([smd_uw[n] for n in names], names, marker="o", label="Unweighted")
   ax.scatter([smd_w[n]  for n in names], names, marker="^", label="Weighted")
   ax.axvline(0, color="black", lw=0.8)
   ax.axvline( 0.1, color="gray", lw=0.8, ls="--")
   ax.axvline(-0.1, color="gray", lw=0.8, ls="--")
   ax.set_xlabel("Standardised mean difference")
   ax.legend()
   plt.tight_layout()
   plt.show()

The dashed lines at ±0.1 mark the common rule-of-thumb for acceptable balance.

**Practical thresholds**

- ``max_abs_smd_weighted < 0.10``: good balance.
- ``max_abs_smd_weighted`` in 0.10–0.25: moderate imbalance; consider a more flexible basis
  or increasing the number of KNN neighbours.
- ``max_abs_smd_weighted > 0.25``: poor balance; the weighted estimator may be biased.

- ``ess_treated`` and ``ess_control`` should be at least 30–50 % of the raw group sizes.
  Much lower values indicate extreme weights.


Cross-fitting and fold-level diagnostics
-----------------------------------------

When ``cross_fit=True`` the estimation loop runs :math:`K` times (one per held-out fold).
The per-fold estimates are stored in ``res.fold_estimates`` (a list of
:class:`~genriesz.SingleEstimate` objects).  Inspecting these helps detect unstable folds::

   for k, fe in enumerate(res.fold_estimates):
       print(f"Fold {k}: ARW={fe.arw:.3f}  max|α|={fe.alpha_abs_max:.1f}")

A fold whose ``alpha_abs_max`` is much larger than the others is a sign that the nuisance fit
in that fold extrapolated badly, usually because of data imbalance or a poorly chosen basis.

If one fold diverges, try:

1. Increasing ``folds`` (more folds → smaller held-out sets → stabler weights).
2. Increasing ``riesz_lam``.
3. Switching to a smoother basis (e.g., :class:`~genriesz.RBFRandomFourierBasis`).
