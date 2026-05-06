API reference
=============

This page lists the main public objects in *genriesz*.

High-level estimation
---------------------

.. autofunction:: genriesz.grr_functional
.. autofunction:: genriesz.grr_ate
.. autofunction:: genriesz.grr_att
.. autofunction:: genriesz.grr_did
.. autofunction:: genriesz.grr_ame

Density ratio and covariate shift
---------------------------------

.. autofunction:: genriesz.fit_density_ratio

.. autoclass:: genriesz.DensityRatioResult
   :members:

Functionals
-----------

.. autoclass:: genriesz.LinearFunctional
   :members:

.. autoclass:: genriesz.CallableFunctional
   :members:

.. autoclass:: genriesz.ATEFunctional
   :members:

.. autoclass:: genriesz.ATTFunctional
   :members:

.. autoclass:: genriesz.DIDFunctional
   :members:

.. autoclass:: genriesz.AMEFunctional
   :members:

Low-level solvers
-----------------

.. autoclass:: genriesz.GRRGLM
   :members:

Bases
-----

The abstract base class all bases inherit from:

.. autoclass:: genriesz.BaseBasis
   :members:

Built-in bases:

.. autoclass:: genriesz.PolynomialBasis
   :members:

.. autoclass:: genriesz.TreatmentInteractionBasis
   :members:

.. autoclass:: genriesz.RBFRandomFourierBasis
   :members:

.. autoclass:: genriesz.GaussianRKHSBasis
   :members:

.. autoclass:: genriesz.RBFNystromBasis
   :members:

.. autoclass:: genriesz.KNNCatchmentBasis
   :members:

.. autoclass:: genriesz.CallableBasis
   :members:

Optional bases (scikit-learn)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Requires ``pip install "genriesz[sklearn]"``.

.. autoclass:: genriesz.sklearn_basis.RandomForestLeafBasis
   :members:

Optional bases (PyTorch)
~~~~~~~~~~~~~~~~~~~~~~~~

Requires ``pip install "genriesz[torch]"``.

.. autoclass:: genriesz.torch_basis.MLPEmbeddingNet
   :members:

.. autoclass:: genriesz.torch_basis.TorchEmbeddingBasis
   :members:

Generators
----------

.. autoclass:: genriesz.BregmanGenerator
   :members:

.. autoclass:: genriesz.SquaredGenerator
   :members:

.. autoclass:: genriesz.UKLGenerator
   :members:

.. autoclass:: genriesz.BKLGenerator
   :members:

.. autoclass:: genriesz.BPGenerator
   :members:

.. autoclass:: genriesz.PUGenerator
   :members:

Matching helpers
----------------

.. autofunction:: genriesz.nn_matching_inverse_propensity_weights
.. autofunction:: genriesz.local_polynomial_nn_lsif_density_ratio
.. autofunction:: genriesz.local_polynomial_nn_lsif_inverse_propensity_weights

Results
-------

.. autoclass:: genriesz.FunctionalEstimate
   :members:

.. autoclass:: genriesz.SingleEstimate
   :members: