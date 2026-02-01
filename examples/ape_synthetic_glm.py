"""Average policy effect example.

Run:

    python examples/ape_synthetic_glm.py

We generate a binary-treatment dataset with heterogeneous treatment effects and
estimate the *average policy effect* between two policies:

    theta = E[ mu(pi1(Z), Z) - mu(pi0(Z), Z) ]

where mu(d,z) = E[Y | D=d, Z=z].

This reduces to the ATE when pi1(z)=1 and pi0(z)=0.
"""

from __future__ import annotations

import numpy as np

from genriesz import (
    GRR,
    PolicyEffectFunctional,
    PolynomialBasis,
    TreatmentInteractionBasis,
    UKLGenerator,
)


def make_synthetic_data(n: int = 4000, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))

    # Treatment assignment (propensity depends on Z)
    logits = 0.6 * Z[:, 0] - 0.4 * Z[:, 1]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n)

    # Heterogeneous treatment effect tau(Z)
    tau = 1.0 + 0.5 * Z[:, 0]
    mu0 = Z[:, 0] + 0.25 * Z[:, 1] ** 2
    Y = mu0 + tau * D + rng.normal(scale=1.0, size=n)

    X = np.concatenate([D.reshape(-1, 1), Z], axis=1)
    return X, Y


def pi1(z: np.ndarray) -> float:
    """Treat if the first covariate is positive."""
    return float(z[0] > 0.0)


def pi0(_z: np.ndarray) -> float:
    """Never treat."""
    return 0.0


def estimate_true_policy_effect(n_mc: int = 200_000, seed: int = 123) -> float:
    """Monte Carlo approximation of the policy effect under the DGP."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_mc, 3))
    tau = 1.0 + 0.5 * Z[:, 0]
    return float(np.mean(tau * (Z[:, 0] > 0.0)))


def main() -> None:
    X, Y = make_synthetic_data(n=4000, d=3, seed=0)
    true_theta = estimate_true_policy_effect()

    # Basis: polynomial on Z, plus treatment interactions.
    psi = PolynomialBasis(degree=2, include_bias=False)
    phi = TreatmentInteractionBasis(base_basis=psi)

    # Linear functional: policy effect between pi1 and pi0.
    m = PolicyEffectFunctional(policy_1=pi1, policy_0=pi0, treatment_index=0)

    # Generator: UKL (automatic link).
    gen = UKLGenerator(C=1.0, branch_fn=lambda x: int(x[0] == 1)).as_generator()

    est = GRR(basis=phi, m=m, generator=gen, penalty="l2", lam=1e-3)
    est.fit(X, max_iter=400, tol=1e-9)

    theta_hat = est.estimate_linear_functional(Y, X)
    resid = est.covariate_balance_residual()

    print("True policy effect (Monte Carlo):", true_theta)
    print("Estimated policy effect:", float(theta_hat))
    print("Balance residual: mean(|resid|) =", float(np.mean(np.abs(resid))))


if __name__ == "__main__":
    main()
