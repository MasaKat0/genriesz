"""Synthetic ATE example with a polynomial basis.

Run:

    python examples/ate_synthetic_glm_polynomial.py

This demonstrates how to plug a polynomial basis into the GLM-style GRR solver
while keeping the link function automatic (induced by the chosen Bregman generator).
"""

from __future__ import annotations

import numpy as np

from genriesz import GRR, ATEFunctional, PolynomialBasis, TreatmentInteractionBasis, UKLGenerator


def make_synthetic_data(n: int = 3000, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))

    # Propensity depends on Z
    logits = 0.6 * Z[:, 0] - 0.4 * Z[:, 1]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n)

    # Outcome: baseline + constant treatment effect
    tau = 1.0
    mu0 = Z[:, 0] + 0.5 * Z[:, 1] ** 2
    Y = mu0 + tau * D + rng.normal(scale=1.0, size=n)

    X = np.concatenate([D.reshape(-1, 1), Z], axis=1)
    return X, Y, tau


def main() -> None:
    X, Y, tau = make_synthetic_data(n=3000, d=3, seed=0)

    # --- Basis: polynomial on Z, plus treatment interactions ---
    psi = PolynomialBasis(degree=2, include_bias=False)
    phi = TreatmentInteractionBasis(base_basis=psi)

    # --- Linear functional: ATE ---
    m = ATEFunctional(treatment_index=0)

    # --- Generator: UKL (automatic link via inverse derivative) ---
    gen = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1)).as_generator()

    est = GRR(basis=phi, m=m, generator=gen, penalty="l2", lam=1e-3)
    est.fit(X, max_iter=500, tol=1e-10)

    ate_hat = est.estimate_linear_functional(Y, X)
    resid = est.covariate_balance_residual()

    print("True ATE (by construction):", tau)
    print("Estimated ATE:", float(ate_hat))
    print("Balance residual: mean(|resid|) =", float(np.mean(np.abs(resid))))


if __name__ == "__main__":
    main()
