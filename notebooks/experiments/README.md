# GRR publication experiment notebooks

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

## Reference-based loss--link selection (`09_*.ipynb` and `refsel/`)

This experiment is separate from the eight notebooks above. It validates the selection
theorems of the manuscript rather than comparing fixed specifications, so it lives in a
package (`refsel/`) with its own tests rather than in a single notebook cell.

The design is specified in `notebooks/experiments/REFERENCE_SELECTION_PLAN.md`, which
**supersedes** `reference_selection_experiment_details.md` and `simulation_coding_design_v10.md`
in the manuscript repository. Read it before changing anything here: each experiment maps to
one claim in Main.tex section 4, and an experiment that answers no referee question does not
belong in the run.

```text
refsel/
  dgp.py          designs, the drifting misspecification family, fold rotation, seeds
  candidates.py   dictionaries, the 90-candidate grid, ScaledGenerator, FoldLibrary
  reference.py    four references, their allowances, the pairwise check
  selection.py    multiplier bootstrap, the error budget, every selection rule
  audit.py        analytic bias/variance audit with a shared integration sample
  inference.py    the four intervals and Monte Carlo standard errors
  calibration.py  bias-to-standard-error calibration; writes calibration.json
  grids.py        publication and smoke grids, tier settings
  report.py       the manuscript tables
  rescaling.py    the generator-rescaling demonstration
```

### Tiers

Set `TIER` in the notebook. Only the replication count changes; the candidate library,
selection rules, designs, and seed construction are identical across tiers.

| Tier | Jobs | Cost |
|---|---|---|
| `smoke` | 4 | under a minute |
| `pilot` | 650 | about 1 core-hour |
| `publication` | 26,000 | about 99 core-hours (measured: 6.3 s at n=1000, 9.9 s at n=3000, 44 s high-dimensional) |

Unlike the other experiment notebooks, this one is safe to open and run: the default tier is
`smoke`. Results go to `notebooks/experiments/results/`, which is git-ignored.

For the publication tier, set `MAX_WORKERS`. Running from the notebook is fine. Running from a
**script** on macOS or Windows requires the standard entry-point guard, because the default
multiprocessing start method re-imports the calling module in every worker:

```python
if __name__ == "__main__":
    run_experiment(config, output_dir)
```

Without it the pool dies with `BrokenProcessPool` partway through the first batch.

### Things that are easy to get wrong here

- **`calibration.json` is committed and must not be regenerated casually.** The publication
  run reads it instead of re-calibrating, which is what makes the run deterministic.
  Regenerating it changes every grid-B scenario.
- **The hidden direction must stay unrepresentable by the candidates and representable by the
  `correct` reference.** Making it unrepresentable by everything breaks the reference's own
  allowance and the experiment stops measuring anything (plan section 18.1).
- **BKL fails on every ATE fold by design** (`domain_error`). It is kept in the library to show
  the admissibility screen removing an incompatible pair; that is why coverage is reported
  unconditionally, with failures in the denominator.
- **`bias_aware_pooled` has no supporting theorem.** Use `bias_aware_split` or
  `conservative_cf` when transcribing numbers into the manuscript.

Tests are in `tests/test_reference_selection_experiment.py` and run under `make verify`;
`make lint` covers `notebooks/experiments/refsel`.
