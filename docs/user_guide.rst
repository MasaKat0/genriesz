User guide
==========

This guide summarizes how to use **Generalized Riesz Regression (GRR)** in this
package. The high-level interface is :func:`genriesz.grr_functional`.

Conceptual workflow
-------------------

To estimate a target parameter \(\theta\) written as a linear functional of the
outcome regression \(\gamma(x) = \mathbb{E}[Y\mid X=x]\), you provide:

- a *functional* ``m(x, gamma)`` (or one of the built-in functional classes),
- a feature map / basis ``phi(X)``, and
- a Bregman generator ``g(x, alpha)`` (or a pre-built generator object).

The package then:

1. constructs the link function from the generator (automatic covariate balancing),
2. fits a Riesz representer model \(\hat\alpha(x)\),
3. optionally fits an outcome model \(\hat\gamma(x)\),
4. reports DM/IPW/AIPW estimates with standard errors, confidence intervals, and p-values.


Bases
-----

All bases in this library implement the same interface:

- batched input: ``basis(X)`` with ``X.shape == (n, d)`` returns a 2D array ``(n, p)``,
- single-row input: ``basis(x)`` with ``x.shape == (d,)`` returns a 1D array ``(p,)``.

Polynomial
^^^^^^^^^^

Use :class:`genriesz.PolynomialBasis` for simple polynomial expansions.

.. code-block:: python

   from genriesz import PolynomialBasis

   psi = PolynomialBasis(degree=2, include_bias=True)
   Phi = psi(X)


Treatment interactions
^^^^^^^^^^^^^^^^^^^^^^

For binary-treatment causal estimands, it is common to interact a base basis
\(\psi(Z)\) with the treatment \(D\). Use :class:`genriesz.TreatmentInteractionBasis`.

.. code-block:: python

   from genriesz import PolynomialBasis, TreatmentInteractionBasis

   psi = PolynomialBasis(degree=2, include_bias=True)   # base basis
   phi = TreatmentInteractionBasis(base_basis=psi)      # [1, D, psi(Z), D*psi(Z)]


RKHS-style bases (RBF kernel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can approximate an RBF kernel using:

- random Fourier features via :class:`genriesz.RBFRandomFourierBasis`, or
- a Nyström approximation via :class:`genriesz.RBFNystromBasis`.

.. code-block:: python

   from genriesz import RBFRandomFourierBasis, TreatmentInteractionBasis

   psi = RBFRandomFourierBasis(n_features=500, sigma=1.0, standardize=True, random_state=0)
   phi = TreatmentInteractionBasis(base_basis=psi)


kNN catchment-area basis (nearest-neighbor matching)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Nearest-neighbor matching can be interpreted as a (squared-loss) Riesz / LSIF
construction with a **catchment-area indicator** basis.

The class :class:`genriesz.KNNCatchmentBasis` implements features

.. math::

   \phi_j(z) = \mathbf{1}\{c_j \in \mathrm{NN}_k(z)\},

where :math:`\{c_j\}` are fitted centers and :math:`\mathrm{NN}_k(z)` denotes
the set of :math:`k` nearest centers of :math:`z`.

.. code-block:: python

   import numpy as np
   from genriesz import KNNCatchmentBasis

   # centers: (n_centers, d)
   # queries: (n_queries, d)
   basis = KNNCatchmentBasis(n_neighbors=3).fit(centers)
   Phi = basis(queries)  # dense (n_queries, n_centers)


Random forest leaf basis (scikit-learn)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you install the optional scikit-learn dependency (``pip install genriesz[sklearn]``),
you can build a flexible tree-induced basis using leaf indicators.

.. code-block:: python

   from sklearn.ensemble import RandomForestRegressor
   from genriesz.sklearn_basis import RandomForestLeafBasis

   rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=0)
   leaf_basis = RandomForestLeafBasis(rf, include_bias=True).fit(X, y)
   Phi = leaf_basis(X)

This keeps the GRR optimization *linear in parameters* (convex) while using a
nonparametric partition of the covariate space.


Neural network feature maps (PyTorch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you install the optional PyTorch dependency (``pip install genriesz[torch]``), you
can use a neural network as a **fixed feature map**.

.. important::

   If you train the neural network jointly inside the GRR objective, you leave
   the GLM setting. The recommended approach is:

   1) train the embedding network separately,
   2) freeze it,
   3) use its output as features in GRR.

The package includes :class:`genriesz.torch_basis.TorchEmbeddingBasis`.

.. code-block:: python

   import torch
   from genriesz.torch_basis import MLPEmbeddingNet, TorchEmbeddingBasis

   net = MLPEmbeddingNet(input_dim=X.shape[1], hidden_dim=64, output_dim=32)
   # (train net on a separate task if desired)
   phi = TorchEmbeddingBasis(model=net, include_bias=True)


Generators and automatic links
------------------------------

A Bregman generator defines both the loss and the induced link function used for
automatic covariate balancing.

The easiest option is to use one of the built-in generator factories:

- :class:`genriesz.SquaredGenerator`
- :class:`genriesz.UKLGenerator`
- :class:`genriesz.BKLGenerator`
- :class:`genriesz.BPGenerator`

Each provides an ``.as_generator()`` method that returns a
:class:`genriesz.BregmanGenerator` instance.

If you want to define your own generator, you can pass:

- ``g(x, alpha)`` and optionally
- its derivative ``g_grad(x, alpha)`` and inverse derivative ``g_inv_grad(x, v)``.

If derivatives are not provided, the package falls back to numerical
finite-differences and scalar root-finding.


Estimators, cross-fitting, and outcome models
---------------------------------------------

The high-level function :func:`genriesz.grr_functional` can report multiple estimators
at once via ``estimators=(...)``:

- ``"dm"``: direct method (plug-in)
- ``"ipw"``: weighting only
- ``"aipw"``: augmented IPW

Set ``cross_fit=True`` to use K-fold cross-fitting. The number of folds is
controlled by ``folds``.

For DM/AIPW you need an outcome regression model \(\hat\gamma\). You can control
how it is fitted via ``outcome_models``:

- ``"shared"``: use the same basis and penalty settings as the Riesz model
- ``"separate"``: use a user-provided outcome model or a separate basis
- ``"both"``: fit both and report both versions
- ``"none"``: skip outcome modeling (then only IPW is available)


Regularization: \(\ell_p\)
--------------------------

For the Riesz model, set:

- ``riesz_penalty="l2"`` for ridge,
- ``riesz_penalty="l1"`` for lasso,
- ``riesz_penalty="lp"`` with ``riesz_p_norm=p`` for general \(p\ge 1\), or
- ``riesz_penalty="l1.5"`` as shorthand.

The outcome model (when using the default linear outcome regression) supports the
same penalty interface via ``outcome_penalty`` and ``outcome_p_norm``.
