# Generalized Riesz Regression experiments

This directory contains the three active experiments for the Generalized Riesz Regression manuscript, which targets Econometrica or the Journal of Econometrics: E-12, E-24, and E-25. The reusable estimators are in `../src/genriesz/`. Experiment designs, replication records, summaries, tables, and figures are implemented in `grrexp/`. The Python runner and the marimo apps use those same modules.

The experiment environment is managed with uv, and the interfaces are Python marimo apps. Jupyter notebooks are historical records and are not the execution path for the current programme. The earlier five-study Python files remain unchanged; use the versioned entry points below for the active programme.

## Environment and execution

Run these commands from `genriesz/experiments/`:

```bash
uv sync --locked
uv run --locked python run_registered_v2.py E-24 --n-jobs 2 --output /absolute/path/to/new-output
uv run --locked marimo edit apps/e24_v2.py
```

Python is pinned to 3.13.6. `pyproject.toml` and `uv.lock` preserve the numerical package versions and lock marimo and its dependencies. The local checkout supplies `genriesz`; another editable installation must not supply the estimators.

Use `E-12`, `E-24`, or `E-25` with the runner. The corresponding marimo apps are `apps/e12_v2.py`, `apps/e24_v2.py`, and `apps/e25_v2.py`. The runner uses the complete ordinary design by default. It exposes worker count and a new output destination, without a reduced experiment mode. Workers execute with one numerical thread each; one to four workers are supported.

Opening an app displays saved results. Its run button executes the complete specified experiment in a new output directory. A historical result displayed by the app remains a result from the earlier source version.

The default output is `results/programme_20261018_uv_marimo_v2/E<number>/run/`. An existing output directory is never overwritten. `--output` selects another new directory. Recorded outputs include source and lock hashes, package versions, task keys, folds, failed fits and warnings, raw replication records, summaries, LaTeX tables, and PDF figures.

## Statistical programme

| Experiment | Design | Replications and inference |
|---|---|---|
| E-12 | Two overlap conditions, Include and Omit, n = 1000, 4000; SQ, UKL1, EB, and LogitAIPW | 500 per cell; full-sample RW for GRR and five-fold ARW |
| E-24 | Nonlinear ATE; n = 1000; six common-dictionary procedures, DML-GBM, and RieszNet | 500 per cell; two-fold ARW; three-fold inner CV where specified |
| E-25 | ATT rare-control example; b0 = 0.2 and 0.6; n = 500, 2000, 8000; matched exponential odds class for UKL0 and SQ, with logistic likelihood as a comparator | 500 per cell; two-fold ARW |

E-12 and E-25 use their existing experiment modules without changes. E-24 uses `grrexp/e24_v2.py`; its ordinary design changes only the sample-size set and the number of outer folds. It retains all eight procedures, 500 repetitions, the three-fold inner CV and tuning grids, the same network architecture, and the complete RieszNet training schedule. Numerical optimization uses the established libraries and their default stopping settings. Failed calculations are recorded; a different estimator is not substituted.

In E-25, the weak-overlap SQ coefficient reference normalization includes the logarithmic factor, and the representer error is evaluated under the stated population law. The parent project's `docs/spec/02_data-and-experiments.md` records its correction and mathematical scope.

## Historical data and results

Existing data snapshots remain in `../notebooks/experiments/data/`. Their SHA-256 values are checked before use; missing data are not downloaded. Previous results and work records remain in `../notebooks/experiments/results/` and `../notebooks/experiments/CHANGES_2026-10-*.md`. Earlier outputs in `results/` are retained as history.

The active programme requires new full-design execution and result checks. Earlier outputs do not verify the current source. Manuscript figures and tables, `Generated/`, and the submission bundle must be reconciled with the accepted recorded results before the new submission package is complete. The submission copy is created from the accepted source and outputs and checked for matching contents; the same Monte Carlo experiment is not executed again merely because it was copied.

## Code verification

From this directory, run:

```bash
uv run --locked python -m pytest ../tests -q
```

Tests of the estimators and entry points are separate from full-design experimental execution. Neither a passing unit test nor a matching old manifest establishes that the current manuscript numbers have been reproduced.
