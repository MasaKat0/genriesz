User guide
==========

This guide summarizes how to use **Generalized Riesz Regression (GRR)** in this
package. The main entry point is :func:`genriesz.grr_functional`.

Conceptual procedure
--------------------

To estimate a target parameter :math:`\theta` written as a linear functional of the
outcome regression :math:`\gamma(x) = \mathbb{E}[Y\mid X=x]`, you provide:

- a *functional* ``m`` (either a built-in :class:`genriesz.LinearFunctional` or a plain callable ``m(x_row, gamma)``),
- a feature map / basis ``phi(X)``, and
- either a Bregman generator object ``generator`` **or** a generator function ``g``.

If you pass a plain callable ``m``, :func:`genriesz.grr_functional` wraps it as
:class:`genriesz.CallableFunctional`. The callable is assumed to be **linear in**
the function argument ``gamma``.

The package then:

1. builds the link function induced by the generator (automatic regressor balancing),
2. fits a Riesz representer model :math:`\hat\alpha(x)`,
3. optionally fits an outcome model :math:`\hat\gamma(x)`,
4. reports RA/RW/ARW/TMLE estimates with standard errors, confidence intervals, and p-values.


Bases
-----

All bases in this library implement the same interface:

- batched input: ``basis(X)`` with ``X.shape == (n, d)`` returns ``(n, p)``,
- single-row input: ``basis(x)`` with ``x.shape == (d,)`` returns ``(p,)``.

Polynomial
^^^^^^^^^^

Use :class:`genriesz.PolynomialBasis` for simple polynomial expansions.

.. code-block:: python

   from genriesz import PolynomialBasis

   psi = PolynomialBasis(degree=2, include_bias=True)
   # Fit is only needed when you call the basis directly.
   # When passed to grr_* functions, bases are copied and fit internally on each training fold.
   psi.fit(X)
   Phi = psi(X)


Treatment interactions
^^^^^^^^^^^^^^^^^^^^^^

For binary-treatment causal estimands, it is common to interact a base basis
:math:`\psi(Z)` with the treatment :math:`D`. Use :class:`genriesz.TreatmentInteractionBasis`.

.. code-block:: python

   from genriesz import PolynomialBasis, TreatmentInteractionBasis

   psi = PolynomialBasis(degree=2, include_bias=True)   # base basis
   phi = TreatmentInteractionBasis(base_basis=psi)      # [D*psi(Z), (1-D)*psi(Z)]


RKHS bases (RBF kernel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can build RBF-kernel (Gaussian-kernel RKHS) features using:

- random Fourier features via :class:`genriesz.RBFRandomFourierBasis`,
- a Nyström feature map via :class:`genriesz.RBFNystromBasis`, or
- an explicit kernel-section basis via :class:`genriesz.GaussianRKHSBasis`.

.. code-block:: python

   from genriesz import GaussianRKHSBasis

   psi = GaussianRKHSBasis(n_centers=300, sigma=1.0, standardize=True, random_state=0)
   psi.fit(X)
   Phi = psi(X)


kNN catchment-area basis (nearest-neighbor matching)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nearest-neighbor matching can be interpreted as a (squared-loss) Riesz / LSIF
construction with a **catchment-area indicator** basis.

The class :class:`genriesz.KNNCatchmentBasis` implements features

.. math::

   \phi_j(z) = \mathbf{1}\{c_j \in \mathrm{NN}_k(z)\},

where :math:`\{c_j\}` are fitted centers and :math:`\mathrm{NN}_k(z)` denotes the set of
:math:`k` nearest centers of :math:`z`.

.. code-block:: python

   import numpy as np
   from genriesz import KNNCatchmentBasis

   basis = KNNCatchmentBasis(n_neighbors=3).fit(centers)
   Phi = basis(queries)  # dense (n_queries, n_centers)


Random forest leaf basis
^^^^^^^^^^^^^^^^^^^^^^^^

You can build a flexible tree-induced basis using leaf indicators.

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier
   from genriesz import TreatmentInteractionBasis
   from genriesz.sklearn_basis import RandomForestLeafBasis

   rf = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=0)
   psi = RandomForestLeafBasis(rf, include_bias=True)
   phi = TreatmentInteractionBasis(base_basis=psi)

This keeps the generalized Riesz regression problem linear in parameters while using
a nonparametric partition of the regressor space.


Neural network feature maps (PyTorch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you install PyTorch (optional), you can use a neural network as a **basis function**.

.. important::

   If you train the neural network jointly inside the GRR objective, you leave
   the GLM setting. The recommended approach is:

   1) train the embedding network separately,
   2) use it as a basis function,
   3) use its output as features in GRR.

The package includes :class:`genriesz.torch_basis.TorchEmbeddingBasis`.

.. code-block:: python

   from genriesz.torch_basis import MLPEmbeddingNet, TorchEmbeddingBasis

   net = MLPEmbeddingNet(input_dim=X.shape[1], hidden_dims=(64, 32), output_dim=16)

   # Optionally train the embedding to predict y (supervised pretraining).
   phi = TorchEmbeddingBasis(net=net, include_bias=True)
   phi.fit(X, y, epochs=10, lr=1e-3, verbose=True)

   Phi = phi(X)


Density ratio estimation
------------------------

The function :func:`genriesz.fit_density_ratio` estimates the covariate-shift
*density ratio*

.. math::

   r(x) = p(x)/q(x)

from two samples ``X_num ~ p`` and ``X_den ~ q``.

Unlike classic uLSIF, :func:`genriesz.fit_density_ratio` is written in the same
**Bregman-divergence / GRR** language used throughout this package:

* you choose a generator (either a built-in name like ``generator='ukl'`` or a
  custom ``g``),
* the function automatically constructs the corresponding **link function**
  :math:`(\partial g)^{-1}`,
* and it fits the ratio by minimizing the induced convex objective.

By default we use a Gaussian-kernel RKHS basis. You can optionally select the
RBF bandwidth ``sigma`` and regularization ``lam`` via cross validation.

.. code-block:: python

   from genriesz import fit_density_ratio

   res = fit_density_ratio(
       X_num,
       X_den,
       generator="ukl",  # or "sq", "bkl", "bp", "pu", or a generator instance
       n_centers=200,
       cv=True,
       folds=5,
       sigma_grid=[0.1, 0.3, 1.0, 3.0],
       lam_grid=[1e-3, 1e-2, 1e-1],
       random_state=0,
   )

   r_hat = res.predict_ratio(X_test)

If you want a custom generator, you can pass ``g`` (and optional derivatives)
just like in :func:`genriesz.grr_functional`.

.. important::

   For general generators, the ratio fit uses a numerical optimizer.
   The squared generator (``generator='sq'`` / :class:`genriesz.SquaredGenerator`)
   uses a closed-form ridge solution. The binary KL generator uses probabilistic
   classification based density ratio estimation.

Generators and automatic links
------------------------------

A Bregman generator defines both the loss and the induced link function used for
automatic regressor balancing.

The easiest option is to use one of the built-in generator objects:

- :class:`genriesz.SquaredGenerator` (SQ-Riesz)
- :class:`genriesz.UKLGenerator` (UKL-Riesz)
- :class:`genriesz.BKLGenerator` (BKL-Riesz)
- :class:`genriesz.BPGenerator` (BP-Riesz)
- :class:`genriesz.PUGenerator` (PU-Riesz)

You can also define a custom generator in two equivalent ways:

1) **Instantiate** a :class:`genriesz.BregmanGenerator` directly.
2) Pass ``g=...`` (and optional derivatives) to :func:`genriesz.grr_functional`.

Custom generator call signatures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A custom generator can be regressor-dependent:

- ``g(x, alpha)``
- optional: ``grad_g(x, alpha)`` (first derivative wrt ``alpha``)
- optional: ``inv_grad_g(x, v)``  (inverse derivative map)
- optional: ``grad2_g(x, alpha)`` (second derivative wrt ``alpha``)

If derivatives are omitted, the package falls back to finite differences and a
Newton solver.

.. important::

   For speed and numerical stability, providing ``inv_grad_g`` (and ideally
   ``grad_g`` / ``grad2_g``) is strongly recommended for custom generators.


Estimators, cross fitting, and outcome models
---------------------------------------------

The high-level function :func:`genriesz.grr_functional` can report multiple estimators
at once via ``estimators=(...)``:

- ``"ra"``: regression adjustment (plug-in)
- ``"rw"``: Riesz weighting (weighting only)
- ``"arw"``: augmented Riesz weighting
- ``"tmle"``: targeted maximum likelihood estimation (one-step fluctuation)

Set ``cross_fit=True`` to use cross fitting. The number of folds is
controlled by ``folds``.

For RA/ARW/TMLE you need an outcome regression model :math:`\hat\gamma`. You can control
how it is fitted via ``outcome_models``:

- ``"shared"``: use the same basis and penalty settings as the Riesz model
- ``"separate"``: use a user-provided basis (``outcome_basis``)
- ``"both"``: fit both and report both versions
- ``"none"``: skip outcome modeling (then only RW is available)

The outcome link function is specified by ``outcome_link`` (``"identity"`` or
``"logit"``). TMLE automatically infers the likelihood:

- ``outcome_link="identity"`` => Gaussian targeting
- ``outcome_link="logit"``    => Bernoulli targeting


Regularization: :math:`\ell_p`
------------------------------

For the Riesz model, set:

- ``riesz_penalty="l2"`` for ridge,
- ``riesz_penalty="l1"`` for lasso,
- ``riesz_penalty="lp"`` with ``riesz_p_norm=p`` for general :math:`p\ge 1`, or
- ``riesz_penalty="l1.5"`` as shorthand.

The default outcome model (linear regression / logistic regression on a basis)
supports the same interface via ``outcome_penalty`` and ``outcome_p_norm``.
