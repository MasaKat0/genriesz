"""Synthetic ATE example using a random-forest leaf-indicator basis.

Run:

    python examples/ate_synthetic_glm_rf_leaf_basis.py

This example illustrates how to use a tree-based basis with the high-level
:func:`genriesz.grr_ate` interface.

We:

1) fit a random forest to predict D from Z,
2) map each sample to its leaf index in each tree,
3) use those leaf indices to obtain psi(Z),
4) form phi(X) = [D*psi(Z), (1-D)*psi(Z)] via TreatmentInteractionBasis,
5) run generalized Riesz regression with an automatically constructed link from
   a Bregman generator.

This keeps the GRR optimization convex in beta while still using a flexible
nonparametric representation.

Notes
-----
This example relies on scikit-learn. Install the optional extra:

    pip install "genriesz[sklearn]"
"""

from __future__ import annotations

import numpy as np

from genriesz import SquaredGenerator, TreatmentInteractionBasis, grr_ate
from genriesz.sklearn_basis import RandomForestLeafBasis


def make_synthetic_data(n: int = 3000, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))

    logits = 0.6 * Z[:, 0] - 0.4 * Z[:, 1] + 0.2 * Z[:, 2]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n)

    tau = 1.0
    mu0 = np.sin(Z[:, 0]) + 0.25 * Z[:, 1] ** 2
    Y = mu0 + tau * D + rng.normal(scale=1.0, size=n)

    X = np.concatenate([D.reshape(-1, 1), Z], axis=1)
    return X, Z, D, Y, tau


def main() -> None:
    X, Z, D, Y, tau = make_synthetic_data(n=800, d=5, seed=0)

    # Fit a modest forest (limit depth/leaves to keep the basis size reasonable).
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=10,
        max_depth=2,
        min_samples_leaf=50,
        random_state=0,
    )

    psi = RandomForestLeafBasis(rf)
    phi = TreatmentInteractionBasis(base_basis=psi)
    gen = SquaredGenerator(C=0.0).as_generator()

    res = grr_ate(
        X=X,
        Y=Y,
        basis=phi,
        generator=gen,
        cross_fit=True,
        folds=2,
        random_state=0,
        estimators=("rw", "arw"),
        outcome_models="shared",
        riesz_penalty="l2",
        riesz_lam=1e-2,
        max_iter=120,
        tol=1e-6,
    )

    print("True ATE (by construction):", tau)
    print(res.summary_text())


if __name__ == "__main__":
    main()
