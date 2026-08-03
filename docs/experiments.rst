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
     - Three data-generating processes with compatible and incompatible loss--link specifications. ATE and ATT are reported.
   * - ``02_main_empirical_ihdp.ipynb``
     - Main text
     - IHDP semi-synthetic ATE and ATT analyses.
   * - ``03_main_empirical_lalonde.ipynb``
     - Main text
     - Lalonde NSW treated ATT analysis, with ATE as a sensitivity analysis.
   * - ``04_appendix_crossfit_comparison.ipynb``
     - Online Appendix
     - Cross fitting compared with same-sample estimation.
   * - ``05_appendix_dimension_variation.ipynb``
     - Online Appendix
     - Changes in covariate dimension for ATE and ATT.
   * - ``06_appendix_model_variation.ipynb``
     - Online Appendix
     - Polynomial, kernel, random-feature, forest-leaf, and matching specifications where the estimator is defined.
   * - ``07_appendix_score_guided_balancing.ipynb``
     - Online Appendix
     - Treatment-specific regressor balancing compared with a treatment-invariant covariate specification.
   * - ``08_appendix_zhao_kernel_experiments.ipynb``
     - Online Appendix
     - Kang--Schafer balance paths and kernel-dictionary comparisons.
   * - ``09_reference_based_loss_link_selection.ipynb``
     - Main text and Online Appendix
     - Reference-based selection with separate training, diagnostic, and evaluation observations.
   * - ``10_coverage_decomposition.ipynb``
     - Online Appendix
     - Coverage decomposition under correct and misspecified dictionaries.
   * - ``11_numerical_failure_selection.ipynb``
     - Online Appendix
     - Numerical failure rates and the direct held-out-imbalance selection diagnostic.

Conventions
-----------

The notebooks import shared statistical calculations from
``genriesz.experiments``. Each notebook keeps its own table construction and
figure commands, including axis labels, legends, and calls to ``plt.show()``.
A missing empirical data file raises an error; no notebook creates substitute
observations or downloads a different data set.

The default UKL and BKL specifications are truncated models whose links
saturate at representer bounds stated before fitting; the binding rates are
reported as ordinary diagnostics. The squared and BP links are exact. If
optimization fails, a KKT condition is not met, a score is nonfinite, or an
exact dual-domain condition is violated, the candidate is recorded as
unavailable. The code does not replace the candidate with another
specification and does not cap the fitted representer after estimation.
Probability bounds that appear in a data-generating process act before
treatment is drawn and do not modify fitted weights.

The reference-based selection design is described in
``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``. Its full configuration
contains 18 scenarios and 24,000 replication jobs. The repository contains the
code and calibration values, but not numerical results from an incomplete run.

After re-running a notebook, use ``make notebooks-strip`` before committing.
``make verify`` checks source files, tests, notebook outputs, and file sizes.

ScoreMatchingRiesz notebooks
----------------------------

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
