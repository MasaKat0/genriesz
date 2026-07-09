Diagnostics and balance checks
==============================

After fitting a GRR model, the returned :class:`~genriesz.FunctionalEstimate`
contains two diagnostic interfaces that are part of the public API:

* scalar and array diagnostics in ``result.diagnostics``;
* Love-plot helpers ``result.love_plot_data()`` and ``result.love_plot()`` for
  treatment-effect functionals.

The point estimates themselves are stored in ``result.estimates``. Standard
entries can also be accessed as ``result.ra``, ``result.rw``, ``result.arw``, and
``result.tmle`` when the corresponding estimator was requested. If
``outcome_models="both"``, these short attributes prefer the shared-basis
estimate.

Alpha diagnostics
-----------------

The fitted Riesz representer values
:math:`\hat{\alpha}_i = \hat{\alpha}(X_i)` enter the RW and ARW estimators as
weights. Large or erratic values indicate numerical instability, weak overlap,
infeasible or nearly infeasible balancing constraints, or insufficient
regularization.

The following diagnostics are stored in ``result.diagnostics`` for all estimands:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``"alpha_abs_mean"``
     - Mean of :math:`|\hat{\alpha}_i|` over all observations.
   * - ``"alpha_abs_p95"``
     - 95th percentile of :math:`|\hat{\alpha}_i|`.
   * - ``"alpha_abs_max"``
     - Maximum of :math:`|\hat{\alpha}_i|`.

Example:

.. code-block:: python

   res = grr_ate(X=X, Y=Y, basis=phi, generator=gen, cross_fit=True)

   print("mean |alpha|:", res.diagnostics["alpha_abs_mean"])
   print("p95 |alpha|:", res.diagnostics["alpha_abs_p95"])
   print("max |alpha|:", res.diagnostics["alpha_abs_max"])

For ATE applications, a very large ``alpha_abs_max`` often signals weak overlap
or an overly flexible representer fit. In such cases, increase ``riesz_lam``, use
a smoother basis, inspect overlap, or switch to a loss-link pair with a more
stable finite-sample geometry.

Covariate balance diagnostics
-----------------------------

For ATE, ATT, and DID, the package computes standardized mean differences before
and after applying the absolute Riesz weights. The scalar summaries are stored in
``result.diagnostics``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Description
   * - ``"max_abs_smd_unweighted"``
     - Maximum absolute standardized mean difference before weighting.
   * - ``"max_abs_smd_weighted"``
     - Maximum absolute standardized mean difference after weighting.
   * - ``"ess_treated"``
     - Kish effective sample size for the weighted treated group.
   * - ``"ess_control"``
     - Kish effective sample size for the weighted control group.

Example:

.. code-block:: python

   print("max unweighted SMD:", res.diagnostics["max_abs_smd_unweighted"])
   print("max weighted SMD:", res.diagnostics["max_abs_smd_weighted"])
   print("treated ESS:", res.diagnostics["ess_treated"])
   print("control ESS:", res.diagnostics["ess_control"])

A common balance rule of thumb is that ``max_abs_smd_weighted`` should be below
0.10. This threshold is not a theorem; it is a practical screen that should be
interpreted together with the outcome scale, the estimand, and overlap.

Love-plot data
--------------

The method ``result.love_plot_data()`` returns one row per covariate with
unweighted and weighted standardized mean differences. If pandas is installed,
the default return value is a ``pandas.DataFrame``. To always get a list of
plain Python dictionaries, pass ``as_pandas=False``.

.. code-block:: python

   rows = res.love_plot_data(as_pandas=False)
   for row in rows[:3]:
       print(row)

The underlying raw data are also stored under ``result.diagnostics["love_plot"]``.
The public helper is preferred because it returns a stable, table-like format.

Love plots
----------

A graphical balance check is available through ``result.love_plot()``. This
requires matplotlib, which is intentionally not a core dependency.

.. code-block:: python

   fig, ax = res.love_plot(threshold=0.1, max_covariates=30)
   fig.tight_layout()

The plot compares standardized mean differences before and after weighting. If
``absolute=True`` (the default), the horizontal axis shows absolute standardized
mean differences and the threshold line is placed at ``threshold``.

Cross fitting
-------------

When ``cross_fit=True``, nuisance models are fitted on training folds and
evaluated on held-out folds. Exact KKT balance holds, when it holds, on the
training sample used for Riesz fitting; it generally does not hold exactly on the
held-out fold. For this reason, the aggregate diagnostics above should be read as
held-out finite-sample diagnostics rather than exact constraints.

The current public result class stores aggregate out-of-fold diagnostics. It does
not store per-fold ``FunctionalEstimate`` objects.

Optimizer and clip diagnostics
------------------------------

For ``riesz_method="grr"`` the per-fold Riesz optimizer state is surfaced so that
failures and internal domain clipping are visible rather than swallowed:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Key
     - Description
   * - ``"riesz_fit_success_rate"``
     - Fraction of folds whose Riesz optimization reported success.
   * - ``"riesz_gradient_norm_max"``
     - Largest final gradient norm across folds.
   * - ``"riesz_kkt_residual_max"``
     - Largest KKT residual across folds.
   * - ``"riesz_clip_binding_rate_max"``
     - Largest fraction of training rows on which the generator's internal
       domain clip was active. A positive value means at least one candidate
       targeted a clipped (modified) estimand; a warning is also emitted.
   * - ``"optimizer"``
     - Per-fold lists (``success``, ``status``, ``gradient_norm``,
       ``kkt_residual``, ``clip_binding_rate``).

Working-span imbalance
----------------------

The covariate SMD balance above is computed on the *raw* covariates. GRR instead
enforces balance on the fitted basis span: the balancing condition is
:math:`\mathbb{E}[\hat\alpha(X)\,\phi_j(X)] = \mathbb{E}[m(X, \phi_j)]` for every
basis coordinate :math:`\phi_j`. Its out-of-fold violation

.. math::

   \Delta_j^{(k)} = \frac{1}{|I_k|}\sum_{i\in I_k}
       \bigl(\hat\alpha^{(-k)}(X_i)\,\phi_j(X_i) - M_{i,j}\bigr)

is the natural driver of the augmented estimator's remainder. The summaries are:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Key
     - Description
   * - ``"held_out_imbalance_max"``
     - :math:`\max_k \max_j |\Delta_j^{(k)}|` over folds and coordinates.
   * - ``"held_out_imbalance_mean"``
     - Mean over folds of the per-fold mean :math:`|\Delta_j^{(k)}|`.
   * - ``"imbalance"``
     - Dict with per-fold ``held_out_working_span_max`` /
       ``held_out_working_span_mean`` lists.

Kernel health
-------------

When the fitted Riesz basis exposes a ``diagnostics()`` method (currently
:class:`~genriesz.GaussianRKHSBasis`), a per-fold kernel-health table is
aggregated. A tiny bandwidth collapses off-diagonal kernel values to zero (each
point only sees itself: underfitting), while a huge bandwidth makes every feature
nearly constant. Small "balanced" weights can therefore *hide* severe
underfitting; the underfitting flag exposes it.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Key
     - Description
   * - ``"kernel_median_min"``
     - Smallest per-fold median kernel value (small ⇒ underfitting risk).
   * - ``"kernel_feature_variance_min"``
     - Smallest per-fold minimum feature variance (small ⇒ near-constant
       features / over-smoothing).
   * - ``"kernel_gram_condition_max"``
     - Largest per-fold Gram condition number (after ridge).
   * - ``"kernel_effective_rank_min"``
     - Smallest per-fold effective rank of the feature Gram matrix.
   * - ``"kernel_underfitting_any"``
     - ``True`` if any fold had ``kernel_median < 1e-3``.
   * - ``"kernel"``
     - Dict with the full per-fold health tables under ``per_fold``.

You can also call the basis probe directly:

.. code-block:: python

   basis = GaussianRKHSBasis(n_centers=200, sigma=1.0).fit(X_train)
   health = basis.diagnostics(X_train)
   print(health["median_pairwise_distance"], health["underfitting"])

Bias proxy
----------

On a shared outcome/Riesz span the empirical second-order term that balancing is
meant to remove is exactly :math:`\Delta^\top\theta`, where :math:`\theta` are the
outcome coefficients. This *directional* proxy is reported together with a
conservative Cauchy–Schwarz bound and a standardized version. These are
**diagnostics only** and are never used to select hyper-parameters.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``"bias_proxy"``
     - Headline directional proxy :math:`b = \mathrm{mean}_k|\Delta^{(k)\top}\theta^{(-k)}|`.
   * - ``"std_bias"``
     - Standardized bias ``b / se`` of the primary (ARW) estimate.
   * - ``"bias"``
     - Dict with ``b_hat``, ``b_hat_max``, ``b_bound`` (the
       :math:`\lVert\Delta\rVert\,\lVert\theta\rVert` bound), ``v_hat``,
       ``std_bias``, ``outcome_coef_norm_mean``, ``outcome_tag``.

The helpers :func:`genriesz.bias_proxy` and
:func:`genriesz.coverage_decomposition` build these quantities (and, in
simulation, a coverage indicator against a supplied truth) for tables assembled
in the experiment notebooks.

Outcome-nuisance diagnostics
----------------------------

Coverage collapse is driven by the *product* of the Riesz and outcome errors, so
the outcome regression gets its own diagnostics (computed out of fold):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Description
   * - ``"outcome_cv_risk"``
     - Cross-validated prediction risk of the primary outcome model (mean
       squared residual for identity link, mean log-loss for logit). This is the
       off-span residual proxy.
   * - ``"outcome_residual_var"``
     - Variance of the out-of-fold residual ``y - gamma_hat``.
   * - ``"outcome"``
     - Dict keyed by outcome tag (``"shared"``/``"separate"``) with ``cv_risk``,
       ``residual_mean``, ``residual_var``, per-fold ``residual_fold_mean`` /
       ``residual_fold_var``, and the working-span ``coef_norm_mean``.

For a simulation-only decomposition of *where* a failure comes from (Riesz side,
outcome side, or their interaction), :func:`genriesz.oracle_decomposition` takes
the fitted and true nuisances and returns the RMS nuisance errors, the empirical
product drift, and the one-step estimators obtained by substituting each true
nuisance in turn. These are for evaluation only and must not drive selection.

Riesz cross-validation
----------------------

When any of ``riesz_sigma_grid`` / ``riesz_lam_grid`` / ``riesz_n_centers_grid``
is supplied, the per-outer-fold selection is recorded:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Description
   * - ``"riesz_cv"``
     - Dict with ``selected`` (per-fold ``sigma``/``lam``/``n_centers``,
       ``n_admissible``, ``best_score``) and, when
       ``return_riesz_cv_path=True``, the full candidate ``path`` per fold.
   * - ``"riesz_cv_selection_score"``
     - The selection criterion used (default ``"bias_variance"``).
   * - ``"riesz_cv_lam_median"`` / ``"riesz_cv_sigma_median"``
     - Median of the selected values across outer folds.
