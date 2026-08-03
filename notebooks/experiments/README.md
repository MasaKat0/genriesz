# Experiment notebooks

The notebooks in this directory reproduce the simulation and empirical analyses for the generalized Riesz regression manuscript. The statistical code used across notebooks is implemented in `genriesz.experiments`. Each notebook imports that code, runs one stated design, and contains its own table and figure commands. Plot labels and layout therefore remain visible at the point where each figure is created.

Run the notebooks from the repository root after installing the package and the experiment dependencies:

```bash
pip install -e .
pip install -r notebooks/experiments/requirements.txt
jupyter notebook notebooks/experiments
```

The IHDP notebooks require `ihdp_npci_1-100.train.npz` and `ihdp_npci_1-100.test.npz` under `notebooks/experiments/data/ihdp`. The Lalonde notebook requires `notebooks/experiments/data/lalonde/lalonde.csv`. A missing data file raises an error; the notebooks do not create substitute observations or download a different dataset.

## Notebook map

| Notebook | Analysis |
|---|---|
| `01_main_simulation_study.ipynb` | Main simulation designs and comparisons of compatible and incompatible loss--link specifications |
| `02_main_empirical_ihdp.ipynb` | IHDP semi-synthetic ATE and ATT analyses |
| `03_main_empirical_lalonde.ipynb` | Lalonde ATT analysis and ATE sensitivity analysis |
| `04_appendix_crossfit_comparison.ipynb` | Cross fitting compared with same-sample estimation |
| `05_appendix_dimension_variation.ipynb` | Covariate-dimension experiment |
| `06_appendix_model_variation.ipynb` | Polynomial, kernel, random-feature, forest, and matching specifications |
| `07_appendix_score_guided_balancing.ipynb` | Regressor balancing and covariate balancing specifications |
| `08_appendix_zhao_kernel_experiments.ipynb` | Kang--Schafer balance paths and kernel-basis comparisons |
| `09_reference_based_loss_link_selection.ipynb` | Reference-based selection with separate training, diagnostic, and evaluation observations |
| `10_coverage_decomposition.ipynb` | Coverage decomposition under correct and misspecified dictionaries |
| `11_numerical_failure_selection.ipynb` | Numerical failure rates and the direct held-out-imbalance selection diagnostic |
| `12_truncated_model_sensitivity.ipynb` | Truncated (bounded) representer models as a target-sensitivity sweep under weak overlap |

The reference-based selection design is specified in `REFERENCE_SELECTION_PLAN.md`. Its implementation is in `genriesz.experiments.reference_selection`. The notebook always uses the complete publication configuration: 18 scenarios, 24,000 replication jobs, and 10,000 Gaussian multiplier draws for each simultaneous radius. Its opening rescaling table records a held-out dual-domain failure instead of clipping the index or substituting another generator.

## Numerical status

The experiment functions use exact generator links. A candidate is unavailable when its optimizer fails, its exact dual domain is violated, its KKT residual exceeds the stated tolerance, or its score contains a nonfinite value. The functions return that status and do not replace the candidate with another loss, cap its fitted representer after estimation, or reuse a result from a different specification.

The probability bounds in the original simulation designs are part of their data-generating processes. They limit the true propensity before treatment is drawn and do not alter a fitted representer.

## Output files

Executed notebooks write results below `notebooks/experiments/results`. The repository does not contain numerical output from partial runs. The reference-based experiment writes a manifest containing the scenarios, candidate specifications, numerical settings, resolved replication counts, and batch identities. A run directory with an absent, incompatible, or incomplete manifest is rejected before any report table or figure is constructed.

The publication calculations have substantial cost. Interruptions can be resumed from batches whose manifest matches the requested configuration. Results are deterministic with respect to the recorded seeds and do not depend on worker order.

## Tests

The tests under `tests/` cover the experiment code, notebook structure, generator domains, strict nesting, sample separation, fixed random-number allocation, candidate failures, reference allowances, selection bounds, inference, and output manifests. Run them with

```bash
make test
```

The Makefile fixes the BLAS thread counts before invoking the suite so that test runtime does not depend on thread oversubscription.

The Parquet persistence tests require the `data` optional dependency, including `pyarrow`. The remaining tests do not require a Parquet engine.
