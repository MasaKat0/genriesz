API reference
=============

High-level estimation
---------------------

.. autofunction:: genriesz.grr_functional

.. autofunction:: genriesz.grr_ate

.. autofunction:: genriesz.grr_ame

.. autofunction:: genriesz.grr_policy_effect

.. autoclass:: genriesz.FunctionalEstimateResult
   :members:

.. autoclass:: genriesz.LinearOutcomeModel
   :members:


Core solver
-----------

.. autoclass:: genriesz.GRR
   :members:


Generators (Bregman divergences)
--------------------------------

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


Bases
-----

.. autoclass:: genriesz.PolynomialBasis
   :members:

.. autoclass:: genriesz.TreatmentInteractionBasis
   :members:

.. autoclass:: genriesz.RBFRandomFourierBasis
   :members:

.. autoclass:: genriesz.RBFNystromBasis
   :members:

.. autoclass:: genriesz.KNNCatchmentBasis
   :members:


Common functionals
------------------

.. autoclass:: genriesz.ATEFunctional
   :members:

.. autoclass:: genriesz.AverageDerivativeFunctional
   :members:

.. autoclass:: genriesz.PolicyEffectFunctional
   :members:
