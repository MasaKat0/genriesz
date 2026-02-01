"""Synthetic ATE example using the GRR solver.

Run:

    python examples/ate_synthetic_glm.py

This script generates a simple binary-treatment dataset and estimates the ATE.
"""

from __future__ import annotations

import numpy as np

from genriesz import ATEFunctional, GRR, SquaredGenerator, UKLGenerator, grr_ate


def make_synthetic_data(n: int = 2000, d: int = 3, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))

    # Propensity depends on Z
    logits = 0.5 * Z[:, 0] - 0.25 * Z[:, 1]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n)

    # Outcome with heterogeneous baseline + constant treatment effect
    tau = 1.0
    mu0 = Z[:, 0] + 0.5 * Z[:, 1] ** 2
    Y = mu0 + tau * D + rng.normal(scale=1.0, size=n)

    X = np.concatenate([D.reshape(-1, 1), Z], axis=1)
    return X, Y


def phi(X: np.ndarray) -> np.ndarray:
    """A simple basis on X=(D,Z): [1, D, Z, D*Z]."""

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        d = X[0]
        z = X[1:]
        return np.concatenate([[1.0], [d], z, d * z])

    d = X[:, [0]]
    z = X[:, 1:]
    return np.concatenate([np.ones((len(X), 1)), d, z, d * z], axis=1)


def main() -> None:
    X, Y = make_synthetic_data(n=2000, d=3, seed=0)

    print("True ATE is approximately 1.0 (by construction).\n")

    # --- Using the high-level interface (recommended) ---
    sq = SquaredGenerator(C=0.0).as_generator()
    res_sq = grr_ate(X=X, Y=Y, basis=phi, generator=sq)
    print("[SQ]  grr_ate output")
    print(res_sq.summary_text())

    ukl = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1)).as_generator()
    res_ukl = grr_ate(X=X, Y=Y, basis=phi, generator=ukl)
    print("\n[UKL] grr_ate output")
    print(res_ukl.summary_text())

    # --- If you only want the Riesz representer and the pure IPW plug-in ---
    m = ATEFunctional(treatment_index=0)
    model = GRR(basis=phi, m=m, generator=ukl, penalty="l2", lam=1e-3)
    model.fit(X)
    ate_ipw = model.estimate_linear_functional(Y, X)
    print("\n[UKL] IPW (no outcome model):", float(ate_ipw))
    print("[UKL] Balance residual (mean):", np.round(model.covariate_balance_residual(), 6))


if __name__ == "__main__":
    main()
