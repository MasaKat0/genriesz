Getting started
===============

Installation
------------

From PyPI (after you publish):

.. code-block:: bash

   pip install genriesz

From a local checkout (editable install):

.. code-block:: bash

   python -m pip install -U pip
   pip install -e .

Optional extras:

.. code-block:: bash

   # scikit-learn integration (random forest leaf basis)
   pip install -e ".[sklearn]"

   # PyTorch integration (neural network feature maps)
   pip install -e ".[torch]"


Quickstart: ATE
---------------

The following example assumes your regressor matrix is ``X = [D, Z]`` where
``D`` is the (0/1) treatment and ``Z`` are covariates.

.. code-block:: python

   import numpy as np
   from genriesz import (
       grr_ate,
       PolynomialBasis,
       TreatmentInteractionBasis,
       UKLGenerator,
   )

   # Synthetic data: X = [D, Z]
   n, d_z = 2000, 5
   rng = np.random.default_rng(0)
   Z = rng.normal(size=(n, d_z))
   D = (rng.normal(size=n) > 0).astype(float)
   Y = 2.0 * D + Z[:, 0] + rng.normal(size=n)

   X = np.column_stack([D, Z])

   # Basis on Z, then interact with D (ATE-friendly)
   psi = PolynomialBasis(degree=2, include_bias=True)
   phi = TreatmentInteractionBasis(base_basis=psi)

   # UKL generator induces the link automatically
   gen = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1.0)).as_generator()

   res = grr_ate(
       X=X,
       Y=Y,
       basis=phi,
       generator=gen,
       cross_fit=True,
       folds=5,
       estimators=("dm", "ipw", "aipw"),
       riesz_penalty="l2",
       riesz_lam=1e-3,
   )

   print(res.summary_text())


Next steps
----------

- See :doc:`user_guide` for details on bases, generators, estimators, and cross-fitting.
- See :doc:`examples` for runnable scripts and a notebook.
- See :doc:`api` for the full API reference.
