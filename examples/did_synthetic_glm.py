"""Synthetic panel DID example (implemented as ATT on ΔY).

Run:

    python examples/did_synthetic_glm.py

This script generates a panel dataset with a pre/post outcome for each unit and
estimates the DID effect using genriesz.grr_did.
"""

from __future__ import annotations

import numpy as np

from genriesz import SquaredGenerator, grr_did


def make_panel_data(n: int = 3000, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))

    logits = 0.6 * Z[:, 0] - 0.25 * Z[:, 1]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n).astype(int)

    # Baseline level + common trend + treatment effect in post period.
    tau = 1.0
    mu = 0.5 * Z[:, 0] - 0.2 * Z[:, 1] ** 2
    trend = 0.5 + 0.1 * Z[:, 0]

    u = rng.normal(scale=1.0, size=n)  # unit fixed effect

    Y0 = mu + u + rng.normal(scale=1.0, size=n)
    Y1 = mu + trend + u + tau * D + rng.normal(scale=1.0, size=n)

    X = np.concatenate([D.reshape(-1, 1), Z], axis=1)
    return X, Y0, Y1, tau


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
    X, Y0, Y1, tau = make_panel_data(n=3000, d=3, seed=0)
    print(f"True DID effect (by construction): {tau:.3f}\n")

    gen = SquaredGenerator(C=0.0).as_generator()
    res = grr_did(
        X=X,
        Y0=Y0,
        Y1=Y1,
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
