"""Synthetic ATE example with an RKHS-like RBF basis (random Fourier features).

Run:

    python examples/ate_synthetic_glm_rkhs_rff.py

This script uses :class:`genriesz.basis.RBFRandomFourierBasis` to approximate an RBF
kernel (an RKHS basis) and then fits :class:`genriesz.genriesz.GRR`.

The link function is *not* hand-coded; it is induced automatically by the chosen
Bregman generator via its inverse derivative.
"""

from __future__ import annotations

import numpy as np

from genriesz import (
    ATEFunctional,
    GRR,
    RBFRandomFourierBasis,
    TreatmentInteractionBasis,
    UKLGenerator,
)


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

    # --- Linear functional: ATE (vectorized) ---
    m = ATEFunctional(treatment_index=0)

    # --- Generator: UKL (automatic link) ---
    gen = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1)).as_generator()

    est = GRR(basis=phi, m=m, generator=gen, penalty="l2", lam=1e-3)
    est.fit(X, max_iter=300, tol=1e-9)

    ate_hat = est.estimate_linear_functional(Y, X)
    resid = est.covariate_balance_residual()

    print("True ATE (by construction):", tau)
    print("Estimated ATE:", float(ate_hat))
    print("Balance residual: mean(|resid|) =", float(np.mean(np.abs(resid))))


if __name__ == "__main__":
    main()
