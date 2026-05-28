API reference
=============

Core estimation
---------------

.. autofunction:: genriesz.grr_functional
.. autofunction:: genriesz.grr_ate
.. autofunction:: genriesz.grr_att
.. autofunction:: genriesz.grr_did
.. autofunction:: genriesz.grr_ame

ScoreMatchingRiesz
------------------

ScoreMatchingRiesz is an optional PyTorch module. The core package exposes a lazy loader:

.. autofunction:: genriesz.load_scorematchingriesz

After installing the optional dependency, import the reusable primitives directly:

.. code-block:: python

   import genriesz.scorematchingriesz as smr

   model = smr.fit_time_smr_dre_infinity(x_q, x_p)
   log_ratio = smr.log_ratio_from_time_score(model, x, x_p_for_norm=x_p)

Experiment-specific data-generating processes, paper tables, and plots are intentionally not
part of ``src/genriesz``. They are written in the notebooks under
``notebooks/scorematchingriesz``.

Basis functions
---------------

.. autoclass:: genriesz.PolynomialBasis
.. autoclass:: genriesz.TreatmentInteractionBasis
.. autoclass:: genriesz.RBFRandomFourierBasis
.. autoclass:: genriesz.GaussianRKHSBasis
.. autoclass:: genriesz.RBFNystromBasis
.. autoclass:: genriesz.KNNCatchmentBasis

Generators
----------

.. autoclass:: genriesz.SquaredGenerator
.. autoclass:: genriesz.UKLGenerator
.. autoclass:: genriesz.BKLGenerator
.. autoclass:: genriesz.BPGenerator
.. autoclass:: genriesz.PUGenerator
.. autoclass:: genriesz.BregmanGenerator

Results
-------

.. autoclass:: genriesz.FunctionalEstimate
.. autoclass:: genriesz.SingleEstimate
