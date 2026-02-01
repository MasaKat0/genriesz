"""Average marginal effect (average derivative) example.

Run:

    python examples/ame_synthetic_glm.py

We generate a continuous-treatment dataset with confounding and estimate the
average marginal effect (AME)

    theta = E[ d/dD mu(D, Z) ]

using the GLM-style GRR solver.

This is a good example of a linear functional that is not a simple difference
between two treatment levels.
"""

from __future__ import annotations

import numpy as np

from genriesz import AverageDerivativeFunctional, GRR, PolynomialBasis, SquaredGenerator


def make_synthetic_data(n: int = 4000, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))

    # Continuous treatment with confounding
    D = 0.7 * Z[:, 0] - 0.3 * Z[:, 1] + rng.normal(scale=1.0, size=n)

    # Outcome: nonlinear baseline + linear treatment effect
    # mu(D,Z) = sin(Z0) + 0.25 Z1^2 + (1 + 0.5 Z1) * D
    mu = np.sin(Z[:, 0]) + 0.25 * Z[:, 1] ** 2 + (1.0 + 0.5 * Z[:, 1]) * D
    Y = mu + rng.normal(scale=1.0, size=n)

    # True AME = E[1 + 0.5 Z1] = 1 (since E[Z1]=0)
    true_ame = 1.0

    X = np.concatenate([D.reshape(-1, 1), Z], axis=1)
    return X, Y, true_ame


def main() -> None:
    X, Y, true_ame = make_synthetic_data(n=4000, d=3, seed=0)

    # Basis: polynomial on the full W = [D, Z]
    phi = PolynomialBasis(degree=2, include_bias=True)

    # Linear functional: average derivative w.r.t. the treatment coordinate (index 0)
    m = AverageDerivativeFunctional(coordinate=0, eps=1e-4)

    # Generator: squared (linear inverse link)
    gen = SquaredGenerator(C=0.0).as_generator()

    est = GRR(basis=phi, m=m, generator=gen, penalty="l2", lam=1e-3)
    est.fit(X, max_iter=300, tol=1e-9)

    ame_hat = est.estimate_linear_functional(Y, X)
    resid = est.covariate_balance_residual()

    print("True AME (by construction):", true_ame)
    print("Estimated AME:", float(ame_hat))
    print("Balance residual: mean(|resid|) =", float(np.mean(np.abs(resid))))


if __name__ == "__main__":
    main()
