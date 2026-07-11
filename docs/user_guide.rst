User guide
==========

This guide summarises how to use **Generalized Riesz Regression (GRR)** in this
package.  The main entry point is :func:`genriesz.grr_functional`.

Conceptual procedure
--------------------

To estimate a target parameter :math:`\theta` written as a linear functional of the
outcome regression :math:`\gamma(x) = \mathbb{E}[Y\mid X=x]`, you provide:

- a *functional* ``m`` (either a built-in :class:`genriesz.LinearFunctional` or a plain callable ``m(x_row, gamma)``),
- a feature map (basis) ``phi(X)``, and
- either a Bregman generator object ``generator`` **or** a generator function ``g``.

If you pass a plain callable ``m``, :func:`genriesz.grr_functional` wraps it as
:class:`genriesz.CallableFunctional`.  The callable must be **linear in** the
function argument ``gamma``.

The package then:

1. builds the link function induced by the generator (automatic regressor balancing),
2. fits a Riesz representer model :math:`\hat\alpha(x)`,
3. optionally fits an outcome model :math:`\hat\gamma(x)`,
4. reports RA, RW, ARW, and TMLE estimates with standard errors, confidence intervals,
   and p-values.


Bases
-----

All bases in this library implement the same interface:

- batched input: ``basis(X)`` with ``X.shape == (n, d)`` returns ``(n, p)``,
- single-row input: ``basis(x)`` with ``x.shape == (d,)`` returns ``(p,)``,
- ``basis.fit(X, y=None)`` fits any data-dependent parameters (centers, standardisation, etc.),
- ``basis.copy()`` returns a deep copy (used internally by cross-fitting).

Bases passed to :func:`genriesz.grr_functional` (and all convenience wrappers) are
**copied and re-fitted** inside each cross-fitting fold.  You do not need to call
``fit`` manually before passing a basis to a ``grr_*`` function.


Polynomial
^^^^^^^^^^

Use :class:`genriesz.PolynomialBasis` for polynomial expansions up to a given total
degree.

.. code-block:: python

   from genriesz import PolynomialBasis

   psi = PolynomialBasis(degree=2, include_bias=True)
   psi.fit(X)       # only needed when calling the basis directly
   Phi = psi(X)     # (n, p)

``PolynomialBasis`` implements ``derivative()`` analytically and is compatible with
:func:`genriesz.grr_ame`.


Treatment interactions
^^^^^^^^^^^^^^^^^^^^^^

For binary-treatment causal estimands (ATE, ATT, and DID), interact a base basis
:math:`\psi(Z)` with the treatment :math:`D`:

.. math::

   \phi(X) = [D \cdot \psi(Z),\; (1-D)\cdot\psi(Z)].

Use :class:`genriesz.TreatmentInteractionBasis`:

.. code-block:: python

   from genriesz import PolynomialBasis, TreatmentInteractionBasis

   psi = PolynomialBasis(degree=2, include_bias=True)   # base basis on Z
   phi = TreatmentInteractionBasis(base_basis=psi)      # [D*psi(Z), (1-D)*psi(Z)]

``TreatmentInteractionBasis`` delegates ``derivative()`` to the base basis, so it
inherits AME compatibility from the base.


RKHS bases (RBF kernel)
^^^^^^^^^^^^^^^^^^^^^^^

Three classes approximate the Gaussian (RBF-kernel) RKHS:

- :class:`genriesz.RBFRandomFourierBasis` — random Fourier features (Rahimi & Recht),
- :class:`genriesz.RBFNystromBasis` — Nyström feature map with eigendecomposition whitening,
- :class:`genriesz.GaussianRKHSBasis` — explicit kernel feature map (one basis function per center).

.. code-block:: python

   from genriesz import RBFRandomFourierBasis, RBFNystromBasis, GaussianRKHSBasis

   rff = RBFRandomFourierBasis(n_features=500, sigma="auto", standardize=True, random_state=0)
   nys = RBFNystromBasis(n_centers=300, sigma="auto", standardize=True, random_state=0)
   krn = GaussianRKHSBasis(n_centers=300, sigma="auto", standardize=True, random_state=0)

All three accept ``sigma="auto"``, which resolves the bandwidth by the
**median heuristic** (the median pairwise distance of the standardized
training sample) at fit time — inside each cross-fitting fold when used with
the ``grr_*`` functions.  A fixed bandwidth such as ``sigma=1.0`` is rarely
well scaled across data sets; prefer ``sigma="auto"`` unless you tune
``sigma`` explicitly (e.g. via ``riesz_sigma_grid`` below).

.. note::

   :class:`genriesz.RBFRandomFourierBasis` implements ``derivative()`` analytically
   (the derivative of :math:`\cos(\omega^\top x + b)` is :math:`-\omega \sin(\ldots)`),
   and is therefore compatible with :func:`genriesz.grr_ame`.
   :class:`genriesz.RBFNystromBasis` and :class:`genriesz.GaussianRKHSBasis` do not
   currently implement ``derivative()`` and cannot be used with ``grr_ame``.


kNN nearest-neighbor indicator basis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nearest-neighbor matching can be interpreted as a squared-loss Riesz (LSIF)
construction with a **nearest-neighbor indicator** basis:

.. math::

   \phi_j(z) = \mathbf{1}\{c_j \in \mathrm{NN}_k(z)\},

where :math:`\{c_j\}` are fitted centers and :math:`\mathrm{NN}_k(z)` denotes the
:math:`k`-nearest centers of :math:`z`.

For ATE, ATT, and DID, combine with :class:`genriesz.TreatmentInteractionBasis`:

.. code-block:: python

   import numpy as np
   from genriesz import (
       KNNCatchmentBasis, TreatmentInteractionBasis,
       grr_ate, SquaredGenerator,
   )

   gen = SquaredGenerator(C=0.0).as_generator()

   # The basis is fitted on the training fold inside each cross-fitting iteration.
   basis_knn = KNNCatchmentBasis(n_neighbors=5, include_bias=False)
   phi       = TreatmentInteractionBasis(base_basis=basis_knn)

   res = grr_ate(X=X, Y=Y, basis=phi, generator=gen,
                 cross_fit=True, folds=5)

When cross-fitting, each training fold becomes the set of centers, so the feature
dimension :math:`p = n_\text{train}`.  The internal outcome model uses a dual
(Woodbury) solve when :math:`p > n_\text{test}`, keeping cost at :math:`O(n^3)`.

.. note::

   ``KNNCatchmentBasis`` is piecewise-constant and does **not** implement
   ``derivative()``.  It cannot be used with :func:`genriesz.grr_ame`.


Random forest leaf basis
^^^^^^^^^^^^^^^^^^^^^^^^

:class:`genriesz.sklearn_basis.RandomForestLeafBasis` turns any scikit-learn
``RandomForest*`` model into a feature map via one-hot leaf indicators.

.. code-block:: python

   from sklearn.ensemble import RandomForestRegressor
   from genriesz import TreatmentInteractionBasis
   from genriesz.sklearn_basis import RandomForestLeafBasis

   rf  = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0)
   psi = RandomForestLeafBasis(rf)          # include_bias=False, normalize=True (defaults)
   phi = TreatmentInteractionBasis(base_basis=psi)

This keeps GRR convex (linear in parameters) while using a nonparametric partition
of the covariate space.

.. note::

   ``RandomForestLeafBasis`` normalizes each leaf-indicator row by
   :math:`1/\!\sqrt{T}` (where :math:`T` is the number of trees) by default
   (``normalize=True``).  Without this, the row :math:`\ell_2`-norm grows as
   :math:`\sqrt{T}`, making the effective regularisation scale tree-count dependent.

   Tree-based bases are **piecewise-constant** and do **not** implement
   ``derivative()``.  They cannot be used with :func:`genriesz.grr_ame`.


Neural network feature maps (PyTorch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you install PyTorch (optional), you can use a neural network as a **basis function**
via :class:`genriesz.torch_basis.TorchEmbeddingBasis`.

.. important::

   If you train the neural network jointly inside the GRR objective, you leave the
   convex (GLM) setting.  The recommended approach is:

   1) train the embedding network separately (e.g., supervised pre-training),
   2) call ``phi.fit(X, y, ...)`` to run the optional training loop,
   3) use the frozen embedding as a feature map in GRR.

.. code-block:: python

   from genriesz.torch_basis import MLPEmbeddingNet, TorchEmbeddingBasis

   net = MLPEmbeddingNet(input_dim=X.shape[1], hidden_dims=(64, 32), output_dim=16)

   phi = TorchEmbeddingBasis(net=net, include_bias=True)
   phi.fit(X, y, epochs=10, lr=1e-3, verbose=True)  # optional supervised pretraining

   Phi = phi(X)   # (n, 17)  — 16 embedding dims + 1 bias

.. note::

   ``TorchEmbeddingBasis`` does not implement ``derivative()`` and cannot be used
   with :func:`genriesz.grr_ame` without additional autograd integration.


AME-compatible bases summary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`genriesz.grr_ame` requires ``basis.derivative(X, coordinate)``.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Basis
     - AME-compatible
     - Note
   * - :class:`~genriesz.PolynomialBasis`
     - Yes
     - Analytical derivative
   * - :class:`~genriesz.RBFRandomFourierBasis`
     - Yes
     - Analytical derivative of :math:`\cos(\omega^\top x + b)`
   * - :class:`~genriesz.TreatmentInteractionBasis`
     - Depends on base
     - Delegates to base basis
   * - :class:`~genriesz.KNNCatchmentBasis`
     - No
     - Piecewise-constant
   * - :class:`~genriesz.sklearn_basis.RandomForestLeafBasis`
     - No
     - Piecewise-constant
   * - :class:`~genriesz.torch_basis.TorchEmbeddingBasis`
     - No
     - No autograd bridge


Density ratio estimation
------------------------

The function :func:`genriesz.fit_density_ratio` estimates the covariate-shift
density ratio

.. math::

   r(x) = p(x)/q(x)

from two samples ``X_num ~ p`` and ``X_den ~ q``.

* You choose a generator (either a built-in name like ``generator='ukl'`` or a
  custom ``g``).
* The function constructs the corresponding link function
  :math:`(\partial g)^{-1}` automatically.
* It fits the ratio by minimising the induced convex objective.

By default a Gaussian-kernel RKHS basis is used.  You can optionally cross-validate
the bandwidth ``sigma`` and regularisation ``lam``.

.. code-block:: python

   from genriesz import fit_density_ratio

   res = fit_density_ratio(
       X_num,
       X_den,
       generator="ukl",      # "sq", "bkl", "bp", "pu", or a generator instance
       n_centers=200,
       cv=True,
       folds=5,
       sigma_grid=[0.1, 0.3, 1.0, 3.0],
       lam_grid=[1e-3, 1e-2, 1e-1],
       random_state=0,
   )

   r_hat = res.predict_ratio(X_test)

.. important::

   For the **squared generator** (``generator='sq'``, i.e. :class:`genriesz.SquaredGenerator`),
   the fit uses a closed-form ridge solution.  For the **binary KL generator**,
   it uses classification-based density ratio estimation.  For all other generators,
   a numerical optimizer (L-BFGS-B) is used.


Generators and automatic links
-------------------------------

A Bregman generator defines both the loss and the induced link function used for
automatic regressor balancing.

The easiest option is to use one of the built-in generator objects:

- :class:`genriesz.SquaredGenerator` (SQ-Riesz)
- :class:`genriesz.UKLGenerator` (UKL-Riesz)
- :class:`genriesz.BKLGenerator` (BKL-Riesz)
- :class:`genriesz.BPGenerator` (BP-Riesz)
- :class:`genriesz.PUGenerator` (PU-Riesz)

Call ``.as_generator()`` to get a :class:`genriesz.BregmanGenerator` instance:

.. code-block:: python

   from genriesz import SquaredGenerator, UKLGenerator

   gen_sq  = SquaredGenerator(C=0.0).as_generator()

   # UKL/BP with a branch function: use + branch for treated, - for control.
   branch = lambda x: int(x[0] == 1.0)
   gen_ukl = UKLGenerator(C=1.0, branch_fn=branch).as_generator()

You can also define a completely custom generator:

.. code-block:: python

   from genriesz import BregmanGenerator

   gen = BregmanGenerator(g=g, grad=grad_g, inv_grad=inv_grad_g, grad2=grad2_g)

Custom generator call signatures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A custom generator can be regressor-dependent:

- ``g(x, alpha)``
- optional: ``grad_g(x, alpha)`` (first derivative w.r.t. ``alpha``)
- optional: ``inv_grad_g(x, v)``  (inverse derivative map)
- optional: ``grad2_g(x, alpha)`` (second derivative w.r.t. ``alpha``)

If derivatives are omitted, the package falls back to finite differences and a
Newton solver.

.. important::

   For speed and numerical stability, providing ``inv_grad_g`` (and ideally
   ``grad_g`` and ``grad2_g``) is strongly recommended for custom generators.

Branch functions for UKL and BP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`genriesz.UKLGenerator` and :class:`genriesz.BPGenerator` are defined on
:math:`\{|\alpha| > C\}`, which has two branches (positive and negative).  When the
sign of :math:`\alpha(x)` is known in advance (e.g., positive for treated, negative
for control in ATE/ATT/DID), pass a ``branch_fn`` to avoid ambiguity:

.. code-block:: python

   branch = lambda x: int(x[0] == 1.0)   # 1 = positive branch (treated), 0 = negative
   gen    = UKLGenerator(C=1.0, branch_fn=branch).as_generator()

Without ``branch_fn``, :math:`\text{sign}(v)` is used, which is correct only when
:math:`|\alpha| > C + 1`.  A :class:`UserWarning` is raised if ``branch_fn`` is
omitted.


Estimators, cross fitting, and outcome models
---------------------------------------------

The high-level function :func:`genriesz.grr_functional` can report multiple estimators
at once via ``estimators=(...)``:

- ``"ra"``:   regression adjustment (plug-in)
- ``"rw"``:   Riesz weighting (weighting only)
- ``"arw"``:  augmented Riesz weighting (doubly robust)
- ``"tmle"``: targeted maximum likelihood estimation (one-step fluctuation)

.. note::

   :func:`genriesz.grr_att` and :func:`genriesz.grr_did` estimate the treatment
   share :math:`\pi = P(D=1)` from the sample and mark the functional with
   ``pi_is_estimated=True``.  The influence function then includes the
   correction term for the estimated :math:`\pi`, so the reported standard
   errors account for this extra estimation step automatically.

Cross fitting
^^^^^^^^^^^^^

Set ``cross_fit=True`` (default) to enable K-fold cross fitting.  The number of
folds is controlled by ``folds`` (default 5).

With cross fitting:

1. the data are split into ``folds`` training and test splits,
2. all nuisance models (Riesz representer and outcome regression) are fitted on the
   **training** fold and evaluated on the **test** fold,
3. estimates are computed from the full-sample out-of-fold predictions.

This removes the overfitting bias that would arise if the nuisance models were
evaluated on the same data used to fit them.

Outcome models
^^^^^^^^^^^^^^

For RA, ARW, and TMLE you need an outcome regression :math:`\hat\gamma`.  Control it
via ``outcome_models``:

- ``"shared"``  — use the **same basis** (and penalty settings) as the Riesz model.
  When ``riesz_method='grr'`` (the default), the already-fitted Riesz basis is
  **reused** directly — no second fit is performed.  This is important for stochastic
  bases (RBF, Nyström, KNN) where a separate fit would draw different random features.
- ``"separate"`` — use a user-provided ``outcome_basis``.
- ``"both"``    — fit both and report both versions.
- ``"none"``    — skip outcome modeling (then only ``"rw"`` is available).

The outcome link function is specified by ``outcome_link`` (``"identity"`` or
``"logit"``).  TMLE infers the likelihood from this link:

- ``outcome_link="identity"`` ⟹ Gaussian targeting
- ``outcome_link="logit"``    ⟹ Bernoulli targeting

Cross-validating the Riesz hyper-parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Riesz bandwidth, regularization and number of kernel centers can be selected
by an **inner** cross-validation, run *inside each outer training fold* so the
outer evaluation fold is never used for selection (no leakage).  Pass any of the
grids to switch it on; the default (all ``None``) keeps the fixed ``riesz_lam``
behaviour:

.. code-block:: python

   res = grr_ate(
       X=X, Y=Y,
       basis=GaussianRKHSBasis(n_centers=120, sigma=1.0, random_state=0),
       generator=SquaredGenerator(),
       folds=5, random_state=0,
       riesz_sigma_grid="auto",             # median heuristic * (0.25,0.5,1,2,4)
       riesz_lam_grid=[1e-3, 1e-2, 1e-1],
       riesz_n_centers_grid=[80, 160],
       riesz_selection_score="bias_variance",   # default
       return_riesz_cv_path=True,
   )
   print(res.diagnostics["riesz_cv"]["selected"])   # per-fold sigma/lambda/centers

Selection is two-stage.  A candidate must first pass an **admissibility screen**
(optimizer success, an effective-sample-size floor, a scale-free *kernel-health
band* that rejects both collapsed and saturated bandwidths, and any thresholds
you set via ``riesz_admissibility_thresholds``).  Among admissible candidates the
criterion chosen by ``riesz_selection_score`` is minimized.  Neither coverage
nor any true nuisance is used for selection.
``riesz_sigma_grid``/``riesz_n_centers_grid`` require a kernel basis
with ``copy_with_params`` (e.g. :class:`~genriesz.GaussianRKHSBasis`); with any
other basis you can still cross-validate ``riesz_lam_grid`` alone.

Selection scores
""""""""""""""""

``riesz_selection_score`` accepts:

- ``"bias_variance"`` (default) — the criterion
  :math:`\hat B^2 + \hat V/n + \tau_R \hat R + \tau_K \hat K` combining the
  held-out imbalance, variance, KKT residual, and kernel-degeneracy penalties.
- ``"bregman_validation"`` — the held-out Bregman risk of each candidate under
  its *own* generator.  Values from different generators are on different
  scales, so this cannot compare candidates across generators.
- ``"squared_loss_validation"`` — the held-out squared-loss (LSIF) risk
  :math:`\tfrac12 \mathbb{E}[\hat\alpha^2] - \mathbb{E}[m(\hat\alpha)]`,
  regardless of the generator used to fit the candidate.  This is a
  generator-agnostic yardstick (minimized at the true Riesz representer,
  uLSIF-style): paths obtained from separate calls with different generators
  (e.g. :class:`~genriesz.SquaredGenerator` vs. a
  :class:`~genriesz.BPGenerator` with varying ``omega``/``C``) are directly
  comparable.  Functionals whose :math:`m(\alpha)` needs a representer
  derivative (e.g. AME) do not support this score and raise a ``ValueError``.
- ``"imbalance_validation"`` — the held-out working-span imbalance alone.

Regularization: :math:`\ell_p`
-------------------------------

For the Riesz model, set:

- ``riesz_penalty="l2"`` for ridge,
- ``riesz_penalty="l1"`` for lasso,
- ``riesz_penalty="lp"`` with ``riesz_p_norm=p`` for general :math:`p \ge 1`,
- shorthand: ``riesz_penalty="l1.5"`` is equivalent to ``"lp"`` with ``p_norm=1.5``.

The outcome model (linear or logistic regression on a basis) supports the same
interface via ``outcome_penalty`` and ``outcome_p_norm``.

When the basis dimension :math:`p` exceeds the number of training observations
:math:`n`, the ridge outcome regression automatically switches to a dual
(Woodbury kernel-ridge) solve:

.. math::

   \hat\theta = \Phi^\top \bigl(\tfrac{1}{n}\Phi\Phi^\top + \lambda I_n\bigr)^{-1}
                \tfrac{\mathbf{y}}{n},

which costs :math:`O(n^3)` instead of :math:`O(p^3)`.  This is triggered
automatically for ``outcome_penalty="l2"`` (``"ridge"``).


Diagnostics and balance checks
-------------------------------

After estimation, ``result.diagnostics`` is populated with several quantities.
See :doc:`diagnostics` for a detailed guide.

Alpha tail statistics (all estimands)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``alpha_abs_mean`` — :math:`n^{-1}\sum_i |\hat\alpha(X_i)|`
- ``alpha_abs_p95``  — 95th percentile of :math:`|\hat\alpha(X_i)|`
- ``alpha_abs_max``  — :math:`\max_i |\hat\alpha(X_i)|`

Large values (relative to the outcome scale) suggest that the estimated Riesz
representer has heavy tails.  This inflates the variance of the RW and ARW estimators
and may indicate that ``riesz_lam`` should be increased or the basis reconsidered.

Covariate balance (ATE, ATT, and DID)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``max_abs_smd_unweighted`` — maximum absolute standardised mean difference before weighting
- ``max_abs_smd_weighted``   — maximum absolute SMD after weighting
- ``ess_treated``, ``ess_control`` — Kish effective sample sizes

A graphical balance check (Love plot) is available via :meth:`genriesz.FunctionalEstimate.love_plot`:

.. code-block:: python

   fig, ax = result.love_plot()
