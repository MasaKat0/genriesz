API reference
=============

High-level functional estimation
--------------------------------

.. autofunction:: genriesz.grr_functional
.. autofunction:: genriesz.grr_ate
.. autofunction:: genriesz.grr_att
.. autofunction:: genriesz.grr_did
.. autofunction:: genriesz.grr_ame
.. autofunction:: genriesz.grr_density_ratio

.. autoclass:: genriesz.estimate_functional.FunctionalEstimateResult
   :members:

.. autoclass:: genriesz.estimate_functional.LinearOutcomeModel
   :members:

.. autoclass:: genriesz.DensityRatioResult
   :members:


Core GRR solver
---------------

.. autoclass:: genriesz.GRR
   :members:

.. autoclass:: genriesz.grr.ARBLink
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


Functionals (estimands)
-----------------------

.. autoclass:: genriesz.ATEFunctional
   :members:

.. autoclass:: genriesz.ATTFunctional
   :members:

.. autoclass:: genriesz.AverageDerivativeFunctional
   :members:


Bases
-----

.. autoclass:: genriesz.PolynomialBasis
   :members:

.. autoclass:: genriesz.RBFRandomFourierBasis
   :members:

.. autoclass:: genriesz.RBFNystromBasis
   :members:

.. autoclass:: genriesz.GaussianRKHSBasis
   :members:

.. autoclass:: genriesz.TreatmentInteractionBasis
   :members:

.. autoclass:: genriesz.KNNCatchmentBasis
   :members:

Optional integrations
---------------------

These require optional dependencies:

- :class:`genriesz.sklearn_basis.RandomForestLeafBasis` (requires ``scikit-learn``)
- :class:`genriesz.torch_basis.TorchEmbeddingBasis` (requires ``torch``)
