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

Density ratio / covariate shift
-------------------------------

.. autofunction:: genriesz.grr_density_ratio

.. autoclass:: genriesz.DensityRatioResult
   :members:

Functionals
-----------

.. autoclass:: genriesz.LinearFunctional
   :members:

.. autoclass:: genriesz.ATEFunctional
   :members:

.. autoclass:: genriesz.ATTFunctional
   :members:

.. autoclass:: genriesz.DIDFunctional
   :members:

.. autoclass:: genriesz.AMEFunctional
   :members:

Bases
-----

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

Generators
----------

.. autoclass:: genriesz.BregmanGenerator
   :members:

.. autoclass:: genriesz.SquaredGenerator
   :members:

.. autoclass:: genriesz.UKLGenerator
   :members:

.. autoclass:: genriesz.BPGenerator
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
