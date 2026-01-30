# Generalized Riesz Regression (GRR)

This repository contains a reference implementation for **generalized Riesz regression** (GRR): a unified framework for fitting Riesz representers via **Bregman-divergence** minimization, and using them for debiased / double machine learning style estimation.

The code in this repository was reorganized into a proper Python package (with minimal changes to the original research implementation).

## Contents

- `src/grr/` — the Python package
  - `grr.GRR_ATE` — cross-fitted ATE estimation (DM / IPW / AIPW)
  - `grr.RKHS_GRR` — RKHS-based Riesz representer learner (NumPy/SciPy)
  - `grr.NN_GRR` — neural-network Riesz representer learner (PyTorch; optional)
- `examples/` — small runnable numerical examples
- `notebooks/` — research notebooks (outputs cleared)
- `data/` — IHDP dataset files used by the notebooks (not required for the package)

## Installation

Create a virtual environment and install in editable mode:

```bash
pip install -U pip
pip install -e .
```

### Optional dependencies

- Install PyTorch support (for `NN_GRR`):

```bash
pip install -e .[torch]
```

- Install extra libraries used in the original experiments/notebooks:

```bash
pip install -e .[experiments]
```

- Development tooling (tests + lint):

```bash
pip install -e .[dev]
```

## Quick start: synthetic ATE experiment

Run the included synthetic experiment:

```bash
python examples/ate_synthetic.py --method RKHS_GRR
```

If you have PyTorch installed you can also run:

```bash
python examples/ate_synthetic.py --method NN_GRR
```

The script prints the true ATE and the GRR estimates (DM / IPW / AIPW) with 95% confidence intervals.

## Minimal API example

```python
import numpy as np
from grr import GRR_ATE

rng = np.random.default_rng(0)

n, d = 2000, 3
X = rng.normal(size=(n, d))

# Propensity
beta = np.array([0.8, -0.5, 0.2])
ps = 1 / (1 + np.exp(-(X @ beta)))
T = rng.binomial(1, ps, size=n)

# Outcome
true_tau = 1.0
Y = true_tau * T + X @ np.array([1.0, -1.0, 0.5]) + rng.normal(scale=1.0, size=n)

est = GRR_ATE()
(
    dm, ipw, aipw,
    dm_lo, dm_hi,
    ipw_lo, ipw_hi,
    aipw_lo, aipw_hi,
) = est.estimate(
    covariates=X,
    treatment=T,
    outcome=Y,
    method="RKHS_GRR",
    riesz_loss="SQ",
    riesz_with_D=True,
    riesz_link_name="Linear",
    folds=2,
    num_basis=50,
    sigma_list=np.array([0.5, 1.0]),
    lda_list=np.array([0.01, 0.1]),
)

print("True ATE:", true_tau)
print("AIPW:", aipw, (aipw_lo, aipw_hi))
```

## Notes on options

- `method`:
  - `"RKHS_GRR"`: kernel approximation + (sigma, lambda) selection by cross-validation
  - `"NN_GRR"`: neural network training (requires PyTorch)

- `riesz_loss`:
  - `"SQ"`: squared-distance-style objective
  - `"UKL"`: unnormalized-KL-style objective
  - `"BKL"`: binary-KL / MLE-style objective

- `riesz_link_name`:
  - `"Linear"`: direct parameterization
  - `"Logit"`: interpret the model output as a propensity logit and map it to the ATE Riesz representer

## Running tests and lint

```bash
ruff check .
pytest -q
```

## Continuous Integration (CI)

A GitHub Actions workflow is provided in `.github/workflows/ci.yml`.
It runs `ruff` and `pytest` on multiple Python versions.

## Citation

If you use this code, please cite the accompanying paper/manuscript:

- Masahiro Kato, *Riesz Representer Fitting under Bregman Divergence: A Unified Framework for Debiased Machine Learning*, January 2026.
