"""Synthetic ATE example using a random-forest leaf-indicator basis.

Run:

    python examples/ate_synthetic_glm_rf_leaf_basis.py

This example illustrates how to use a tree-based basis with the *GLM-style*
GRR solver (:class:`genriesz.glm.GRR`).

We:

1) fit a random forest to predict D from Z,
2) map each sample to its leaf index in each tree,
3) one-hot encode those indices to obtain psi(Z),
4) form phi(W) = [1, D, psi(Z), D*psi(Z)] via TreatmentInteractionBasis,
5) run GRR with an automatically constructed link from a Bregman generator.

This keeps the GRR optimization convex in beta while still using a flexible
nonparametric representation.

Requires:
    pip install genriesz[sklearn]
"""

from __future__ import annotations

import numpy as np

from genriesz import ATEFunctional, GRR, TreatmentInteractionBasis, UKLGenerator
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
    X, Z, D, Y, tau = make_synthetic_data(n=3000, d=5, seed=0)

    # Fit a modest forest (limit depth/leaves to keep the basis size reasonable).
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=20,
        random_state=0,
    )

    psi = RandomForestLeafBasis(rf)
    psi.fit(Z, D)

    phi = TreatmentInteractionBasis(base_basis=psi)
    m = ATEFunctional(treatment_index=0)
    gen = UKLGenerator(C=1.0, branch_fn=lambda w: int(w[0] == 1)).as_generator()

    est = GRR(basis=phi, m=m, generator=gen, penalty="l2", lam=1e-3)
    est.fit(X, max_iter=400, tol=1e-9)

    ate_hat = est.estimate_linear_functional(Y, X)

    print("True ATE (by construction):", tau)
    print("Estimated ATE:", float(ate_hat))


if __name__ == "__main__":
    main()
