# Generalized Riesz Regression experiments

This directory contains the five current experiments for the Generalized Riesz Regression manuscript, which targets Econometrica or the Journal of Econometrics. The reusable estimators are in `../src/genriesz/`. Experiment designs, replication records, summaries, tables, and figures are implemented in `grrexp/`. The Python runner and the marimo apps use those same modules.

The experiment environment is managed with uv, and the interfaces are Python marimo apps. Jupyter notebooks are historical records and are not the execution path for the current programme.

## Environment and execution

Run these commands from `genriesz/experiments/`:

```bash
uv sync --locked
uv run --locked python run_registered.py E-19 --n-jobs 2 --output /absolute/path/to/new-output
uv run --locked marimo edit apps/e19.py
```

Python is pinned to 3.13.6. `pyproject.toml` and `uv.lock` preserve the numerical package versions used by the preceding programme and lock marimo and its dependencies. The local checkout supplies `genriesz`; another editable installation must not supply the estimators.

Use `E-12`, `E-19`, `E-22`, `E-24`, or `E-25` with the runner. The corresponding marimo apps are `apps/e12.py`, `apps/e19.py`, `apps/e22.py`, `apps/e24.py`, and `apps/e25.py`. The runner uses the complete design by default. It exposes worker count and a new output destination, without a reduced experiment mode. Workers execute with one numerical thread each; one to four workers are supported.

Opening an app displays saved results. Its run button executes the complete specified experiment in a new output directory. A historical result displayed by the app remains a result from the earlier source version.

The default output is `results/programme_20261018_uv_marimo/E<number>/run/`. An existing output directory is never overwritten. `--output` selects another new directory. Recorded outputs include source and lock hashes, package versions, task keys, folds, failed fits and warnings, raw replication records, summaries, LaTeX tables, and PDF figures.

## Statistical programme

| Experiment | Design | Replications and inference |
|---|---|---|
| E-25 | ATT rare-control example; b0 = 0.2 and 0.6; n = 500, 2000, 8000; matched exponential odds class for UKL0 and SQ, with logistic likelihood as a comparator | 500 per cell; two-fold ARW |
| E-24 | Nonlinear ATE; n = 1000, 4000; six common-dictionary procedures, DML-GBM, and RieszNet | 500 per cell; five-fold ARW; three-fold inner CV where specified |
| E-12 | Two overlap conditions, Include and Omit, n = 1000, 4000; SQ, UKL1, EB, and LogitAIPW | 500 per cell; full-sample RW for GRR and five-fold ARW |
| E-19 | DGP19Q; n = 1000, 4000; SQ and the specified seven-column outcome OLS | 500 per cell; five-fold ARW |
| E-22 | LaLonde; n = 614; D1 and D2 with SQ, UKL0, and LogitAIPW | 20 predetermined five-fold partitions; partition 0 is primary, the other 19 describe sensitivity |

The migration preserves these sample sizes, replication counts, tuning grids, seeds, and RieszNet training schedule. Numerical optimization uses the established libraries and their default stopping settings. Failed calculations are recorded; a different estimator is not substituted.

E-19 must use its outcome basis separately from its Riesz basis. In E-25, the weak-overlap SQ coefficient reference normalization includes the logarithmic factor, and the representer error is evaluated under the stated population law. In E-22, score concentration is reported for the primary partition and the sensitivity partitions. The parent project's `docs/spec/02_data-and-experiments.md` records these corrections and their mathematical scope.

## Historical data and results

Existing data snapshots remain in `../notebooks/experiments/data/`. Their SHA-256 values are checked before use; missing data are not downloaded. Previous results and work records remain in `../notebooks/experiments/results/` and `../notebooks/experiments/CHANGES_2026-10-*.md`. The earlier programme and its inspected results are retained as history.

The full programme has not yet been rerun after the migration and corrections. Earlier outputs do not verify the current code. Manuscript figures and tables, `Generated/`, and the submission bundle must be reconciled with recorded full-design results before the new submission package is complete.

## Code verification

From this directory, run:

```bash
uv run --locked python -m pytest ../tests -q
```

Tests of the estimators and entry points are separate from full-design experimental execution. Neither a passing unit test nor a matching old manifest establishes that the current manuscript numbers have been reproduced.
