Experiment notebooks
====================

The repository includes a consolidated notebook-based experimental suite under
``notebooks/experiments``. The notebooks are organized by manuscript placement:
main-text simulations, main-text empirical studies, and appendix experiments.

Each analysis estimates both ATE and ATT whenever the implemented method supports
both targets. If a method is target-specific, the notebook keeps the available
target and records failures explicitly. Each notebook is executable independently
and displays tables and figures inside Jupyter. Plot titles, table titles, grid
sizes, and figure settings are defined in notebook cells so manuscript formatting
can be edited directly.

Notebook map
------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Notebook
     - Placement
     - Purpose
   * - ``01_main_simulation_study.ipynb``
     - Main text
     - Three-DGP simulation with compatible loss-link pairs, incompatible loss-link pairs, and regularization paths. ATE and ATT are both reported.
   * - ``02_main_empirical_ihdp.ipynb``
     - Main text
     - IHDP semi-synthetic benchmark for ATE and ATT.
   * - ``03_main_empirical_lalonde.ipynb``
     - Main text
     - Lalonde NSW treated ATT benchmark, with ATE reported as sensitivity.
   * - ``04_appendix_crossfit_comparison.ipynb``
     - Appendix
     - Cross fitting versus no cross fitting for ATE and ATT.
   * - ``05_appendix_dimension_variation.ipynb``
     - Appendix
     - Dimension variation for ATE and ATT.
   * - ``06_appendix_model_variation.ipynb``
     - Appendix
     - RKHS, polynomial, random forest, nearest-neighbor matching, and random Fourier features. GRR models estimate ATE and ATT; nearest-neighbor matching is kept as an ATE-only baseline when ATT is unsupported.
   * - ``07_appendix_score_guided_balancing.ipynb``
     - Appendix
     - Score-guided balancing comparison: regressor balancing versus covariate balancing, with ATE and ATT displayed separately.
   * - ``08_appendix_zhao_kernel_experiments.ipynb``
     - Appendix
     - Zhao/Kang-Schafer balance path and kernel-basis mismatch experiments.

ScoreMatchingRiesz notebooks
---------------------------

The repository includes ``notebooks/scorematchingriesz`` for the ICML
ScoreMatchingRiesz replication. The package source contains only reusable
ScoreMatchingRiesz primitives in ``genriesz.scorematchingriesz``; all
paper-specific DGPs, Monte Carlo loops, empirical-data construction, result
tables, and figures are written directly in the notebooks. The folder has no separate index notebook; each notebook name records its paper target.

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Notebook
     - Paper item
     - Purpose
   * - ``Experiments_Section_9_1_Table_2a_Figure_1a_AME.ipynb``
     - Section 9.1
     - AME simulation, Table 2(a), and Figure 1(a).
   * - ``Experiments_Section_9_1_Table_2b_Figure_1b_APE.ipynb``
     - Section 9.1
     - Pushforward APE simulation at ``delta=1``, Table 2(b), and Figure 1(b).
   * - ``Figure_2_Policy_Path_Visualization.ipynb``
     - Section 9.1
     - Policy-path visualization with the true path and local AME approximation.
   * - ``Experiments_Section_9_2_Table_3_IHDP_ATE_Benchmark.ipynb``
     - Section 9.2 and Appendix R
     - IHDP semi-synthetic ATE benchmark.
   * - ``Experiments_Appendix_P_Tables_5_6_HighDimensional_Pushforward.ipynb``
     - Appendix P
     - High-dimensional pushforward cases.
   * - ``Experiments_Appendix_P_Table_7_NonPushforward_Stochastic_Policies.ipynb``
     - Appendix P
     - Non-pushforward stochastic-policy case.
   * - ``Experiments_Appendix_Q_Table_8_Figures_4_6_LocalProjection_PolicyPaths.ipynb``
     - Appendix Q
     - Local-projection policy-path application.
