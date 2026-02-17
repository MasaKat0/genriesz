.. genriesz documentation master file

genriesz: Generalized Riesz Regression
======================================

**genriesz** is a small Python package for **Generalized Riesz Regression (GRR)** under
**Bregman divergences**, with automatic regressor balancing (ARB).

The central user-facing API is:

- :func:`genriesz.grr_functional` for a generic linear functional,
- :func:`genriesz.grr_ate` for the ATE,
- :func:`genriesz.grr_att` for the ATT,
- :func:`genriesz.grr_did` for panel DID (implemented as ATT on ΔY),
- :func:`genriesz.grr_ame` for average marginal effects.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   user_guide
   api
   examples
   references

