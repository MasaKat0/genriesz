.. genriesz documentation master file

genriesz: Generalized Riesz Regression
======================================

**genriesz** is a small Python package for **Generalized Riesz Regression (GRR)** under
**Bregman divergences**, with automatic covariate balancing.

The central user-facing API is:

- :func:`genriesz.grr_functional` for a generic linear functional,
- :func:`genriesz.grr_ate` for the ATE,
- :func:`genriesz.grr_ame` for average marginal effects,
- :func:`genriesz.grr_policy_effect` for average policy effects.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   user_guide
   api
   examples

