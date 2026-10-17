# Historical GRR experiment notebooks

The active experiments have moved to `../../experiments/` and use uv and marimo. See [the current experiment README](../../experiments/README.md). Do not use these Jupyter notebooks as the current execution path. Existing data, results, and `CHANGES_2026-10-*.md` work records remain here so their recorded paths and provenance are preserved.

The instructions below describe earlier programmes. They do not verify the migrated code, and the historical results are not results from the corrected implementation.

## Previous paper experiment programme

The previous programme consisted of E-25 (weak-overlap ATT moments), E-24 (nonlinear ATE),
E-12 (balance and omitted directions), E-19 (average marginal effect), and E-22 (LaLonde).
The exact designs and implementation choices are in [CHANGES_2026-10-18.md](CHANGES_2026-10-18.md).

Run one experiment from the genriesz root:

```bash
python3 notebooks/experiments/run_registered.py E-25 --n-jobs 2
```

Each experiment has one execution stage. The default replication count is 500 for
simulations and 20 predetermined five-fold partitions for LaLonde. Parallel execution
uses `grrexp.parallel.run_tasks`, with one to four workers (default two). Set
`GRREXP_MAX_WORKERS_FILE` to a text file containing the desired worker count to adjust
concurrency during a run, up to the initial `--n-jobs` limit. Workers finish their current
task before retiring.

Results are written once to `results/programme_20261018/E<number>/run/`. An existing
output directory is never replaced; `--output` selects a new destination. Each run saves
source hashes, package versions, seeds, fold assignments, raw replication records including
failed fits, a manifest, CSV summaries, LaTeX tables and PDF figures. No pilot, bootstrap,
population fitting stage, prediction test family or parallel bit-identity prerequisite is run.
Only local data snapshots are read; no data are downloaded by this programme.

## Superseded notebooks (E-01 to E-11)

The notebooks `01` to `09` below are superseded (E-01 to E-11 in the parent repository's
`docs/spec/02_data-and-experiments.md`). They are kept as a record and are not referenced by
the manuscript. The rest of this README describes them.

These notebooks contain the publication-scale experiments for the Generalized Riesz Regression manuscript.

The exploratory fast-mode code, synthetic real-data fallbacks, and external plotting/table helper module have been removed. Each notebook contains its own table and plotting code so titles, axis labels, method labels, colors, line styles, and font sizes can be edited directly in the notebook. ATE and ATT results are displayed in separate tables and separate figures, not pooled into a single table or axis.

Run from the repository root:

```bash
pip install -e .
pip install -r notebooks/experiments/requirements.txt
jupyter notebook notebooks/experiments
```

The IHDP notebook uses the npci-format files under `notebooks/experiments/data/ihdp`. The Lalonde notebook downloads and caches the public MatchIt/Rdatasets Lalonde CSV if it is not already present. No fallback data are generated.

## After re-running a notebook

The Monte Carlo loops emit NumPy `RuntimeWarning`s, and Jupyter stores every one of them in the notebook as an `stderr` output. A single execution of `01_main_simulation_study.ipynb` writes 6508 of them and grows the file past GitHub's 100 MB blob limit, which makes the whole push fail. Before committing a re-run:

```bash
make notebooks-strip   # drop stderr outputs; figures, tables, and stdout are kept
```

`make verify` refuses to pass while an executed notebook still carries `stderr` output, and `make install-hooks` installs a pre-commit hook that enforces the same check plus a 50 MB file-size guard.

### The matmul warnings are spurious

On Apple silicon, the `divide by zero`, `overflow`, and `invalid value encountered in matmul` warnings do not indicate a numerical problem in this package. Apple's Accelerate BLAS raises bogus floating-point exception flags for any `@` product above roughly 16x16, and NumPy reports them. `np.eye(40) @ np.eye(40)` reproduces all three warnings while returning the exact identity, and the products agree with `np.einsum` to the last bit. See numpy issues [#28790](https://github.com/numpy/numpy/issues/28790) and [#29820](https://github.com/numpy/numpy/issues/29820); upgrading NumPy does not help, because the defect is in Accelerate.

Do not "harden" `basis.py` or `glm.py` against these. A NumPy built against OpenBLAS instead of Accelerate is silent. A broad `np.errstate(all="ignore")` would suppress the flags without changing any result, but it would also hide genuine warnings. Instead, each notebook's setup cell installs a **message-targeted** filter — `warnings.filterwarnings("ignore", message=r"(?:divide by zero|overflow|invalid value) encountered in matmul$", category=RuntimeWarning)` — placed right after the `filterwarnings("once")` catch-all so it takes precedence. It suppresses only these three spurious matmul messages; every non-matmul `RuntimeWarning` (e.g. `overflow encountered in exp`) stays visible in the live session. `make notebooks-strip` still removes any residual `stderr` from the committed notebook as a backstop. (A genuine `overflow encountered in matmul` shares the same text and would also be filtered; in practice the experiment notebooks cap the Riesz weights with `EXPERIMENT_ALPHA_CAP`, and any real blow-up still surfaces as non-finite estimates through the existing fit-failure checks.)


Display convention: ATE and ATT are never placed in the same table or the same figure. Each notebook filters by `estimand` and then displays a separate table or plot for each target.

Stability convention: KL-type branchwise generators use estimand-specific shift constants. ATE uses `C=1`, while ATT uses a smaller shift because the ATT control-branch Riesz representer can have magnitude below one. This avoids the artificial weight explosions that occur when an ATT experiment is forced into an ATE-style `|alpha|>1` domain.


Note on the score-guided balancing appendix: the current high-level `grr_att` wrapper targets the full ATT effect functional. A pure covariate-balancing ATT counterfactual-mean variant requires a separate functional, so that unsupported combination is skipped rather than plotted as a degenerate result.


The notebooks use positive regularization grids and a finite-overlap guard for the UKL, BKL, BP, and propensity-index fits. The guard is defined directly in each notebook so it can be edited together with the plotting code.

Recent numerical fixes:

- `06_appendix_model_variation.ipynb` now uses the defined `BASIS_KINDS` grid and includes an explicit notebook-local `fit_matching_ate` function for the ATE-only nearest-neighbor matching baseline.
- The score-guided appendix fixes the loss at UKL-Riesz and uses a separate regressor-basis outcome model. This avoids the degenerate `alpha=0` solution that occurs when an unconstrained SQ loss is combined with covariate-only ATE features.
- The Zhao/Kang--Schafer appendix now reports balance-path diagnostics rather than treatment-effect MSE for that subsection. This follows the purpose of Zhao's Figure 1 and avoids displaying unstable outcome estimates from deliberately misspecified early-step models.
- The weak-overlap synthetic DGP is kept nonlinear but no longer uses extremely heavy-tailed covariates. Propensities are clipped to `[0.05, 0.95]` to make the comparison about loss-link behavior rather than rare numerical outliers.
