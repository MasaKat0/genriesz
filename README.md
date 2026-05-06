# genriesz — Generalized Riesz Regression (GRR)

A Python library for **Generalized Riesz Regression** (GRR) under **Bregman divergences** — a unified way to fit **Riesz representers** with **automatic regressor balancing** and then report **RA, RW, and ARW** estimates with inference (optionally via cross fitting).

- **Docs**: https://genriesz.readthedocs.io/en/latest/
- **Paper**: [A Unified Framework for Debiased Machine Learning: Riesz Representer Fitting under Bregman Divergence (arXiv:2601.07752)](https://arxiv.org/abs/2601.07752)

---

## Contents

- [Contents](#contents)
- [Installation](#installation)
- [Core idea](#core-idea)
- [Quickstart: ATE (Average Treatment Effect)](#quickstart-ate-average-treatment-effect)
- [Choosing a Bregman generator (Table 1 from the paper)](#choosing-a-bregman-generator-table-1-from-the-paper)
  - [Built-in generator classes](#built-in-generator-classes)
- [General API: `grr_functional`](#general-api-grr_functional)
  - [Providing $g'$ and $(g')^{-1}$](#providing-g-and-g-1)
- [Built-in estimands](#built-in-estimands)
- [Basis functions](#basis-functions)
  - [Polynomial basis](#polynomial-basis)
  - [RKHS bases](#rkhs-bases)
  - [Nearest-neighbor matching (kNN indicator basis)](#nearest-neighbor-matching-knn-indicator-basis)
  - [Random forest leaf encodings (scikit-learn)](#random-forest-leaf-encodings-scikit-learn)
  - [Neural network features (PyTorch)](#neural-network-features-pytorch)
- [Jupyter notebook](#jupyter-notebook)
- [References](#references)
- [License](#license)

---

## Installation

Python **>= 3.10**.

From PyPI:

```bash
pip install genriesz
```

Optional extras:

```bash
# scikit-learn integrations (tree-based feature maps)
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

## Core idea

You specify:

- a linear functional **`m(X, γ)`** (the estimand),
- a feature map (basis) **`φ(X)`**,
- a Bregman generator **`g(X, α)`** (or one of the built-in generator classes),

and the library will:

1. build the **link function** induced by `g`,
2. fit a **Riesz representer** `α̂(X)` via GRR,
3. optionally fit an outcome model `γ̂(X)` (for RA, ARW, and TMLE),
4. return **RA, RW, ARW, and TMLE** point estimates and inference (SE, CI, and p-value), optionally with **cross fitting**.

> **Notation in this library**: the regressor is `X` (shape `(n, d)`) and the outcome is `Y` (shape `(n,)`).
> If you prefer the paper’s notation, you can think of `X` as the full regressor vector (often `X = [D, Z]`).

---

## Quickstart: ATE (Average Treatment Effect)

The ATE is available as a convenient wrapper `grr_ate`.

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

# Unnormalized KL generator with a branch function:
#   + branch for treated (D=1), - branch for control (D=0)
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
    estimators=("ra", "rw", "arw", "tmle"),
)

print(res.summary_text())
```

---

## Choosing a Bregman generator (Table 1 from the paper)

The generator `g` determines the GRR objective, and (through the induced link) the *shape* of the fitted representer and weights.
The paper’s **Table 1** summarizes how common choices relate to well-known **density-ratio estimation** and **Riesz representer and balancing-weight** methods.

> **Note on citations:** GitHub README rendering does **not** resolve LaTeX bibliography commands like `\citep{...}`.
> The table below uses clickable author–year links. For full bibliography entries (author lists, venues), see [CITATIONS.md](CITATIONS.md).

> Convention: `C ∈ ℝ` is a problem-dependent constant; `ω ∈ (0, ∞)`.

<summary><strong>Table 1 — Correspondence among Bregman generators, density-ratio estimation, and Riesz representer estimation</strong></summary>

| Bregman generator $g(\alpha)$ | Density-ratio (DR) estimation view | Riesz representer (RR) estimation view |
|---|---|---|
| $(\alpha - C)^2$ | LSIF ([Kanamori et al., 2009](https://jmlr.org/papers/v10/kanamori09a.html)) and KuLSIF ([Kanamori et al., 2012](https://link.springer.com/article/10.1007/s10994-011-5266-3)) | **SQ-Riesz regression** (this library); RieszNet and ForestRiesz ([Chernozhukov et al., 2022](https://arxiv.org/abs/2110.03031)); RieszBoost ([Lee & Schuler, 2025](https://arxiv.org/abs/2501.04871)); KRRR ([Singh, 2021](https://arxiv.org/abs/2102.11076)); nearest-neighbor matching ([Lin et al., 2023](https://arxiv.org/abs/2112.13506)); causal tree and causal forest ([Wager & Athey, 2018](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1319839)) |
| **Dual solution (linear link)** | Kernel mean matching ([Gretton et al., 2009](https://www.gatsby.ucl.ac.uk/~gretton/papers/covariateShiftChapter.pdf)) | Sieve Riesz representer ([Chen & Christensen, 2015](https://www.jstor.org/stable/43616960)); stable balancing weights ([Zubizarreta, 2015](https://www.tandfonline.com/doi/abs/10.1080/01621459.2015.1023805); [Bruns-Smith et al., 2025](https://arxiv.org/abs/2304.14545)); approximate residual balancing ([Athey et al., 2018](https://arxiv.org/abs/1604.07125)); covariate balancing by SVM ([Tarr & Imai, 2025](https://imai.fas.harvard.edu/research/files/causalsvm.pdf)) |
| $(\lvert\alpha\rvert - C)\log(\lvert\alpha\rvert - C) - \lvert\alpha\rvert$ | UKL divergence minimization ([Nguyen et al., 2010](https://arxiv.org/abs/0809.0853)) | **UKL-Riesz regression** (this library); tailored loss minimization ($\alpha=\beta=-1$; [Zhao, 2019](https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-2/Covariate-balancing-propensity-score-by-tailored-loss-functions/10.1214/18-AOS1698.full)); calibrated estimation ([Tan, 2020](https://academic.oup.com/biomet/article-abstract/107/1/137/5658668)) |
| **Dual solution (logistic or log link)** | KLIEP ([Sugiyama et al., 2008](https://www.ism.ac.jp/editsec/aism/60/699.pdf)) | Entropy balancing weights ([Hainmueller, 2012](https://www.cambridge.org/core/journals/political-analysis/article/entropy-balancing-for-causal-effects-a-multivariate-reweighting-method-to-produce-balanced-samples-in-observational-studies/220E4FC838066552B53128E647E4FAA7)) |
| $(\lvert\alpha\rvert - C)\log(\lvert\alpha\rvert - C) - (\lvert\alpha\rvert + C)\log(\lvert\alpha\rvert + C)$ | BKL divergence minimization ([Qin, 1998](https://academic.oup.com/biomet/article-abstract/85/3/619/229087)); TRE ([Rhodes et al., 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/33d3b157ddc0896addfb22fa2a519097-Abstract.html)) | **BKL-Riesz regression** (this library); logistic MLE propensity-score fit (standard approach); tailored loss minimization ($\alpha=\beta=0$; [Zhao, 2019](https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-2/Covariate-balancing-propensity-score-by-tailored-loss-functions/10.1214/18-AOS1698.full)) |
| $\frac{(\lvert\alpha\rvert - C)^{1+\omega} - (\lvert\alpha\rvert - C)}{\omega} - (\lvert\alpha\rvert - C)$, $\omega>0$ | Basu's Power (BP) divergence minimization ([Sugiyama et al., 2012](https://www.cambridge.org/core/books/density-ratio-estimation-in-machine-learning/BCBEA6AEAADD66569B1E85DDDEAA7648)) | **BP-Riesz regression** (this library) |
| $C\log(1-\lvert\alpha\rvert) + C\lvert\alpha\rvert\bigl(\log\lvert\alpha\rvert - \log(1-\lvert\alpha\rvert)\bigr)$, $\alpha\in(0,1)$ | PU learning and nonnegative PU learning ([du Plessis et al., 2015](https://proceedings.mlr.press/v37/plessis15.html); [Kiryo et al., 2017](https://arxiv.org/abs/1703.00593)) | PU-Riesz regression (this library) |
| General Bregman divergence minimization | Density-ratio matching ([Sugiyama et al., 2012](https://www.cambridge.org/core/books/density-ratio-estimation-in-machine-learning/BCBEA6AEAADD66569B1E85DDDEAA7648)); D3RE ([Kato & Teshima, 2021](https://proceedings.mlr.press/v139/kato21a.html)) | **Generalized Riesz regression** (this library via custom `BregmanGenerator`) |

Full bibliography: see [CITATIONS.md](CITATIONS.md).

### Built-in generator classes

For most use-cases you can start from one of the built-ins:

- `SquaredGenerator`  → squared distance ("SQ-Riesz")
- `UKLGenerator`      → unnormalized KL divergence ("UKL-Riesz")
- `BKLGenerator`      → binary KL divergence ("BKL-Riesz")
- `BPGenerator`       → Basu's power divergence ("BP-Riesz")
- `PUGenerator`       → bounded-weights generator ("PU-Riesz")
- `BregmanGenerator`  → bring your own `g`, optionally with `grad` and `inv_grad`

---

## General API: `grr_functional`

`grr_functional` is the most general entry point.

You provide:

- `m(x_row, gamma)` — the estimand (a **linear** functional),
- a basis `basis(X)` — feature map returning an `(n, p)` design matrix,
- a Bregman `generator`.

`m` can be either:

- a built-in `LinearFunctional` (recommended), or
- a plain Python callable (wrapped as `CallableFunctional`).

Example skeleton:

```python
import numpy as np
from genriesz import grr_functional, BregmanGenerator

def m(x, gamma):
    # x is a single row (1D array)
    # gamma is a callable gamma(x_row)
    return gamma(x)

def g(x, alpha):
    # x is a single row; alpha is a scalar
    return 0.5 * alpha**2

def basis(X):
    # X is (n,d) -> (n,p)
    return np.c_[np.ones(len(X)), X]

X = np.random.randn(200, 3)
Y = np.random.randn(200)

generator = BregmanGenerator(g=g)  # grad/inv_grad can be derived numerically if omitted

res = grr_functional(
    X=X,
    Y=Y,
    m=m,
    basis=basis,
    generator=generator,
    estimators=("rw",),
)

print(res.summary_text())
```

Notes:

- In the example above, ``m`` is a plain callable. Internally, ``grr_functional`` wraps
  it as `CallableFunctional`.
- The callable must be **linear in** the function argument ``gamma``. If you need
  performance or advanced control, implement a custom subclass of
  `LinearFunctional` instead.
- If you want a custom name in the summary output, wrap explicitly:
  ``m = CallableFunctional(m, name="MyEstimand")``.
- Bernoulli TMLE (``outcome_link="logit"``) is implemented for the built-in
  treatment-type functionals (ATE/ATT/DID). If you represent those estimands via a
  custom callable ``m``, prefer the built-in wrappers (e.g. ``grr_ate``).

### Providing $g'$ and $(g')^{-1}$

If you can implement the derivative `grad(W_i, alpha)` and inverse-derivative
`inv_grad(W_i, v)` analytically, pass them to:

```python
BregmanGenerator(g=..., grad=..., inv_grad=...)
```

If you omit them, the library falls back to:

- finite differences for $g'$, and
- scalar root-finding for $(g')^{-1}$.

---

## Built-in estimands

The following convenience wrappers are included:

- **ATE** (average treatment effect): `grr_ate` or `ATEFunctional(...)`
- **ATT** (average treatment effect on the treated): `grr_att` or `ATTFunctional(...)`
- **DID** (panel DID as ATT on ΔY): `grr_did` or `DIDFunctional(...)`
- **AME** (average marginal effect, i.e. average derivative): `grr_ame` or `AMEFunctional(...)`

For covariate-shift *density ratio* estimation via generalized Bregman divergences, see `fit_density_ratio`.

---

## Basis functions

### Polynomial basis

```python
from genriesz import PolynomialBasis

psi = PolynomialBasis(degree=3)
Phi = psi(X)  # (n,p)
```

### RKHS bases

Approximate an RBF kernel with either **random Fourier features** or a **Nyström** basis:

```python
from genriesz import RBFRandomFourierBasis, RBFNystromBasis

rff = RBFRandomFourierBasis(n_features=500, sigma=1.0, standardize=True, random_state=0)
Phi_rff = rff(X)

nys = RBFNystromBasis(n_centers=500, sigma=1.0, standardize=True, random_state=0)
Phi_nys = nys(X)
```

### Nearest-neighbor matching (kNN indicator basis)

Nearest-neighbor matching can be expressed using a *nearest-neighbor indicator* basis

$\phi_j(z) = \mathbf{1}\{c_j \in \mathrm{NN}_k(z)\}$,

where $\{c_j\}$ are centers and $\mathrm{NN}_k(z)$ is the set of *k* nearest centers of $z$.

```python
from genriesz import KNNCatchmentBasis

centers = X[:200]                  # example
queries = X[200:]                  # example

basis = KNNCatchmentBasis(n_neighbors=3).fit(centers)
Phi = basis(queries)               # dense (n_queries, n_centers)
```

See `examples/ate_synthetic_nn_matching.py` for an end-to-end matching-style ATE estimate.

### Random forest leaf encodings (scikit-learn)

You can use random forest leaf encodings as basis functions:

```python
from sklearn.ensemble import RandomForestRegressor
from genriesz import TreatmentInteractionBasis
from genriesz.sklearn_basis import RandomForestLeafBasis

rf = RandomForestRegressor(n_estimators=30, max_depth=3, random_state=0)
psi = RandomForestLeafBasis(rf)
phi = TreatmentInteractionBasis(base_basis=psi)
```

### Neural network features (PyTorch)

If you have PyTorch installed, you can use a neural network as a **basis function**.

See `src/genriesz/torch_basis.py` for a minimal wrapper.

---

## Jupyter notebook

An end-to-end notebook with runnable examples is provided at:

- `notebooks/ATE_example.ipynb`
- `notebooks/ATT_example.ipynb`
- `notebooks/AME_example.ipynb`
- `notebooks/DID_example.ipynb`


---

## References

If you use **genriesz** in academic work, please cite:

- [A Unified Framework for Debiased Machine Learning: Riesz Representer Fitting under Bregman Divergence (arXiv:2601.07752)](https://arxiv.org/abs/2601.07752)  
  - Consolidates earlier related drafts: [arXiv:2509.22122](https://arxiv.org/abs/2509.22122), [arXiv:2510.26783](https://arxiv.org/abs/2510.26783), [arXiv:2510.23534](https://arxiv.org/abs/2510.23534).
  - Bibtex-entry:

    ```
    @misc{Kato2026unifiedframework,
      title={A Unified Framework for Debiased Machine Learning: Riesz Representer Fitting under Bregman Divergence}, 
      author={Masahiro Kato},
      year={2026},
      note={{a}rXiv: 2601.07752},
    }
    ```
- [Direct Bias-Correction Term Estimation for Propensity Scores and Average Treatment Effect Estimation (arXiv:2509.22122)](https://arxiv.org/abs/2509.22122)
- [Nearest Neighbor Matching as Least Squares Density Ratio Estimation and Riesz Regression (arXiv:2510.24433)](https://arxiv.org/abs/2510.24433)

For full bibliography entries (author lists, venues), see [CITATIONS.md](CITATIONS.md).

---

## License

GNU General Public License v3.0 (GPL-3.0).
