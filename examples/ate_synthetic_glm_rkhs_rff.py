"""Synthetic ATE example with an RKHS-like RBF basis (random Fourier features).

Run:

    python examples/ate_synthetic_glm_rkhs_rff.py

This script uses :class:`genriesz.basis.RBFRandomFourierBasis` to approximate an RBF
kernel (an RKHS-style basis) and then estimates the ATE via :func:`genriesz.grr_ate`.

The link function is *not* hand-coded; it is induced automatically by the chosen
Bregman generator via its inverse derivative.
"""

from __future__ import annotations

import numpy as np

from genriesz import RBFRandomFourierBasis, TreatmentInteractionBasis, UKLGenerator, grr_ate


def make_synthetic_data(n: int = 2500, d: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))

    logits = 0.6 * Z[:, 0] - 0.4 * Z[:, 1] + 0.2 * Z[:, 2]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n)

    tau = 1.0
    mu0 = np.sin(Z[:, 0]) + 0.25 * Z[:, 1] ** 2
    Y = mu0 + tau * D + rng.normal(scale=1.0, size=n)

    X = np.concatenate([D.reshape(-1, 1), Z], axis=1)
    return X, Y, tau


def main() -> None:
    X, Y, tau = make_synthetic_data(n=2500, d=5, seed=0)

    # --- Basis: RKHS-like RBF features on Z, plus treatment interactions ---
    # psi(Z) is high-dimensional, but m provides a vectorized basis_matrix()
    # implementation so GRR remains fast.
    psi = RBFRandomFourierBasis(n_features=300, sigma=1.0, standardize=True, random_state=0)
    phi = TreatmentInteractionBasis(base_basis=psi)

    # --- Generator: UKL (automatic link) ---
    gen = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1)).as_generator()

    res = grr_ate(
        X=X,
        Y=Y,
        basis=phi,
        generator=gen,
        cross_fit=True,
        folds=5,
        random_state=0,
        estimators=("ra", "rw", "arw", "tmle"),
        outcome_models="shared",
        riesz_penalty="l2",
        riesz_lam=1e-3,
        max_iter=300,
        tol=1e-9,
    )

    print("True ATE (by construction):", tau)
    print(res.summary_text())


if __name__ == "__main__":
    main()
