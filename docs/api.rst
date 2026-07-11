API reference
=============

Core estimation
---------------

.. autofunction:: genriesz.grr_functional
.. autofunction:: genriesz.grr_ate
.. autofunction:: genriesz.grr_att
.. autofunction:: genriesz.grr_did
.. autofunction:: genriesz.grr_ame

Functionals
-----------

Built-in linear functionals passed to (or constructed by) the estimation
functions above. A plain callable ``m(x_row, gamma)`` is wrapped as
:class:`genriesz.CallableFunctional` automatically.

.. autoclass:: genriesz.LinearFunctional
.. autoclass:: genriesz.CallableFunctional
.. autoclass:: genriesz.ATEFunctional
.. autoclass:: genriesz.ATTFunctional
.. autoclass:: genriesz.DIDFunctional
.. autoclass:: genriesz.AMEFunctional

Basis functions
---------------

.. autoclass:: genriesz.BaseBasis
.. autoclass:: genriesz.CallableBasis
.. autoclass:: genriesz.PolynomialBasis
.. autoclass:: genriesz.TreatmentInteractionBasis
.. autoclass:: genriesz.RBFRandomFourierBasis
.. autoclass:: genriesz.GaussianRKHSBasis
.. autoclass:: genriesz.RBFNystromBasis
.. autoclass:: genriesz.KNNCatchmentBasis

Optional basis integrations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bases backed by optional dependencies live in separate modules; install the
matching extra (``pip install "genriesz[sklearn]"`` and
``pip install "genriesz[torch]"``):

.. autoclass:: genriesz.sklearn_basis.RandomForestLeafBasis
.. autoclass:: genriesz.torch_basis.TorchEmbeddingBasis

``genriesz.torch_basis`` also provides ``MLPEmbeddingNet``, a simple MLP
embedding network for use with :class:`~genriesz.torch_basis.TorchEmbeddingBasis`.
Instantiating either class requires PyTorch (an :class:`ImportError` is raised
otherwise); see :doc:`user_guide` for usage.

Generators
----------

.. autoclass:: genriesz.SquaredGenerator
.. autoclass:: genriesz.UKLGenerator
.. autoclass:: genriesz.BKLGenerator
.. autoclass:: genriesz.BoundedBKLGenerator
.. autoclass:: genriesz.BPGenerator
.. autoclass:: genriesz.PUGenerator
.. autoclass:: genriesz.BregmanGenerator
.. autoexception:: genriesz.DomainError

Results
-------

.. autoclass:: genriesz.FunctionalEstimate
.. autoclass:: genriesz.SingleEstimate

Density ratio estimation
------------------------

.. autofunction:: genriesz.fit_density_ratio
.. autoclass:: genriesz.DensityRatioResult

Matching
--------

Nearest-neighbor matching viewed as LSIF/Riesz regression, and its
local-polynomial extensions.

.. autofunction:: genriesz.nn_matching_inverse_propensity_weights
.. autofunction:: genriesz.local_polynomial_nn_lsif_density_ratio
.. autofunction:: genriesz.local_polynomial_nn_lsif_inverse_propensity_weights

Diagnostics helpers
-------------------

.. autofunction:: genriesz.bias_proxy
.. autofunction:: genriesz.coverage_decomposition
.. autofunction:: genriesz.oracle_decomposition

Model selection (inner CV)
--------------------------

.. autoclass:: genriesz.GRRCVConfig
.. autoclass:: genriesz.GRRCVResult
.. autofunction:: genriesz.select_grr_hyperparams

Low-level solvers
-----------------

.. autoclass:: genriesz.GRRGLM
.. autoclass:: genriesz.OutcomeGLM

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
