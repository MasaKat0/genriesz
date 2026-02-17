"""Synthetic ATT example using the high-level GRR functional API.

Run:

    python examples/att_synthetic_glm.py

This script generates a binary-treatment dataset with heterogeneous treatment
effects and estimates the ATT.
"""

from __future__ import annotations

import numpy as np

from genriesz import SquaredGenerator, grr_att


def make_synthetic_data(n: int = 2000, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))

    logits = 0.7 * Z[:, 0] - 0.3 * Z[:, 1]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n).astype(int)

    # Heterogeneous treatment effect => ATE != ATT in general.
    tau = 1.0 + 0.5 * Z[:, 0]
    mu0 = 0.5 * Z[:, 0] + 0.25 * Z[:, 1] ** 2
    Y0 = mu0 + rng.normal(scale=1.0, size=n)
    Y1 = mu0 + tau + rng.normal(scale=1.0, size=n)
    Y = D * Y1 + (1 - D) * Y0

    X = np.concatenate([D.reshape(-1, 1), Z], axis=1)
    return X, Y, Y0, Y1, D


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
    X, Y, Y0, Y1, D = make_synthetic_data(n=3000, d=3, seed=0)

    # "True" ATT via Monte Carlo on the same simulated population.
    true_att = float(np.mean((Y1 - Y0)[D == 1]))
    print(f"Approx. true ATT (Monte Carlo): {true_att:.4f}\n")

    gen = SquaredGenerator(C=0.0).as_generator()
    res = grr_att(
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
        tol=1e-8,
    )

    print(res.summary_text())


if __name__ == "__main__":
    main()
