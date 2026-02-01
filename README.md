# genriesz — Generalized Riesz Regression (GRR)

This repository packages a small Python library for **Generalized Riesz Regression** under **Bregman divergences**.

The key idea is:

- you specify a **linear functional** `m(X, γ)` (the estimand),
- you specify a **basis** `φ(X)`,
- you specify a **Bregman generator** `g(X, α)`,

and the library:

1. builds the **automatic-covariate-balancing (ACB)** link function from `g`,
2. fits a **Riesz representer** `α̂(X)` via GRR,
3. optionally fits an **outcome model** `γ̂(X)`,
4. returns **DM / IPW / AIPW** estimates with **standard errors, confidence intervals, and p-values** (optionally with cross-fitting).

> Notation: in this library the *regressor* is called `X` and the outcome is called `Y`.
> If you prefer the paper's notation, you can think of `X` as the full regressor vector (often `X=[D,Z]`).

---

## Installation

From PyPI:

```bash
pip install genriesz
```

Optional extras:

```bash
# scikit-learn integrations (random forest leaf basis)
pip install "genriesz[sklearn]"

# PyTorch integrations (neural-network feature maps)
pip install "genriesz[torch]"
```

From a local checkout (editable install):

```bash
python -m pip install -U pip
pip install -e .
```

---

## Quickstart: ATE (Average Treatment Effect)

The ATE can be estimated as a special case of `grr_functional`.

```python
import numpy as np
from genriesz import (
    grr_ate,
    UKLGenerator,
    PolynomialBasis,
    TreatmentInteractionBasis,
)

# Example layout: X = [D, Z]
#   D: treatment (0/1)
#   Z: covariates
n, d_z = 1000, 5
rng = np.random.default_rng(0)
Z = rng.normal(size=(n, d_z))
D = (rng.normal(size=n) > 0).astype(float)
Y = 2.0 * D + Z[:, 0] + rng.normal(size=n)

X = np.column_stack([D, Z])

# Base basis on Z (or on all of X if you prefer).
psi = PolynomialBasis(degree=2)

# ATE-friendly basis: interact the base basis with treatment.
phi = TreatmentInteractionBasis(base_basis=psi)

# A common generator choice for ATE-style balancing.
# The branch function chooses the sign of alpha depending on the treatment.
# Here: positive for treated (D=1), negative for control (D=0).
gen = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1.0)).as_generator()

res = grr_ate(
    X=X,
    Y=Y,
    basis=phi,
    generator=gen,
    cross_fit=True,
    folds=5,
    riesz_penalty="l2",
    riesz_lam=1e-3,
    estimators=("dm", "ipw", "aipw"),
)

print(res.summary_text())
```

---

## General API: `grr_functional`

`grr_functional` is the most general entry point.

You provide:

- `m(X, gamma)` — the estimand,
- a `basis(X)` — feature map,
- a Bregman generator `g(X, alpha)` (or a pre-built `generator`).

Example skeleton:

```python
import numpy as np
from genriesz import grr_functional, BregmanGenerator

def m(x, gamma):
    # x is a single row (1D array)
    # gamma is a callable gamma(w)
    # return a scalar
    return gamma(x)

def g(x, alpha):
    # x is a single row; alpha is a scalar
    # return g(x, alpha)
    return 0.5 * alpha**2

def basis(X):
    # X is (n,d); return (n,p)
    return np.c_[np.ones(len(X)), X]

X = np.random.randn(200, 3)
Y = np.random.randn(200)

generator = BregmanGenerator(g=g)  # gradients/inverse-grad can be auto-derived numerically

res = grr_functional(
    X=X,
    Y=Y,
    m=m,
    basis=basis,
    generator=generator,
    estimators=("ipw",),
)

print(res.summary_text())
```

### Providing `g'` and `(g')^{-1}`

If you can analytically implement the derivative `g_grad(X_i, alpha)` and the inverse derivative
`g_inv_grad(X_i, v)`, pass them to `BregmanGenerator(g=..., grad=..., inv_grad=...)`.

If you omit them, the library falls back to:

- **finite differences** for `g'`, and
- **scalar root-finding** for `(g')^{-1}`.

---

## Basis functions

### Polynomial basis

```python
from genriesz import PolynomialBasis

psi = PolynomialBasis(degree=3)
Phi = psi(X)  # (n,p)
```

### RKHS-style bases

You can approximate an RBF kernel either with **random Fourier features** or a **Nyström** basis.

```python
from genriesz import RBFRandomFourierBasis, RBFNystromBasis

rff = RBFRandomFourierBasis(n_features=500, sigma=1.0, standardize=True, random_state=0)
Phi_rff = rff(X)

nys = RBFNystromBasis(n_centers=500, sigma=1.0, standardize=True, random_state=0)
Phi_nys = nys(X)
```

### Nearest-neighbor matching (kNN catchment-area basis)

Nearest-neighbor matching can be expressed using a *catchment-area* indicator basis

\(\phi_j(z) = \mathbf{1}\{c_j \in \mathrm{NN}_k(z)\}\),

where \(\{c_j\}\) are a set of centers and \(\mathrm{NN}_k(z)\) is the set of k nearest
centers of \(z\).

This library provides `KNNCatchmentBasis`:

```python
from genriesz import KNNCatchmentBasis

basis = KNNCatchmentBasis(n_neighbors=3).fit(centers)
Phi = basis(queries)  # dense (n_queries, n_centers)
```

See `examples/ate_synthetic_nn_matching.py` for an end-to-end matching-style ATE estimate.

### Random forest leaves (scikit-learn)

If you have `scikit-learn` installed, you can use a random forest as a feature map by encoding
leaf indices.

```python
from sklearn.ensemble import RandomForestRegressor
from genriesz.sklearn_basis import RandomForestLeafBasis

rf = RandomForestRegressor(n_estimators=200, random_state=0)
leaf_basis = RandomForestLeafBasis(rf)
Phi_rf = leaf_basis(X)
```

### Neural network features (PyTorch)

If you have PyTorch installed, you can use a neural network as a **fixed feature map**.

See `src/genriesz/torch_basis.py` for a minimal wrapper.

---

## Included estimands

- **ATE**: `grr_ate`, or `m=ATEFunctional(...)`
- **AME** (average marginal effect / average derivative): `grr_ame`, or `m=AverageDerivativeFunctional(...)`
- **Average policy effect**: `grr_policy_effect`, or `m=PolicyEffectFunctional(...)`

---

## Documentation

The documentation is written in Sphinx and configured for **Read the Docs** via
`.readthedocs.yaml`.

Build locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`.

### Publishing to PyPI

This repository is ready for PyPI via **either**:

1. **manual upload** (``python -m build`` + ``twine upload``), or
2. **GitHub Actions + PyPI Trusted Publishing** (recommended).

The release workflow lives at:

- `.github/workflows/release.yml`

#### Manual upload (recommended first: TestPyPI)

```bash
python -m pip install -U pip build twine

# Build sdist + wheel
python -m build

# Sanity check the distributions
python -m twine check dist/*

# Upload to TestPyPI (recommended for the very first trial)
python -m twine upload -r testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

If you prefer to store credentials in a config file, create ``~/.pypirc``.

#### GitHub Actions (Trusted Publishing)

1. Create your first release tag (example):

   ```bash
   git tag v0.1.9
   git push --tags
   ```

2. On PyPI, add a **Trusted Publisher** pointing to your GitHub repository,
   using the workflow file ``release.yml``.

After that, pushing a tag ``vX.Y.Z`` will automatically:

- build + check your package,
- upload it to PyPI, and
- create a GitHub Release with the artifacts attached.

See `docs/releasing.rst` for a more detailed checklist.

---

## Jupyter notebook

An end-to-end notebook with runnable examples is provided at:

- `notebooks/GRR_end_to_end_examples.ipynb`

---

## Development

Run tests:

```bash
pytest -q
```

---

## License

MIT
