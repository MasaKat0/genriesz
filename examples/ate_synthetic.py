"""Synthetic ATE example for Generalized Riesz Regression (GRR).

This script generates a simple semi-synthetic causal inference dataset:

- Covariates: X ~ N(0, I)
- Treatment:  T ~ Bernoulli(sigmoid(X @ beta))
- Outcome:    Y = tau * T + f(X) + noise

Then it estimates the ATE using GRR with either:

- RKHS_GRR (NumPy/SciPy only)
- NN_GRR (requires PyTorch)

Run:
    python examples/ate_synthetic.py --method RKHS_GRR
    python examples/ate_synthetic.py --method NN_GRR
"""

from __future__ import annotations

import argparse

import numpy as np

from grr import GRR_ATE


def make_synthetic_data(n: int, d: int, tau: float, seed: int):
    """Generate a synthetic dataset with a known ATE."""

    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n, d))

    # Propensity model
    beta = rng.normal(scale=0.6, size=d)
    ps = 1.0 / (1.0 + np.exp(-(X @ beta)))
    T = rng.binomial(1, ps, size=n)

    # Outcome model
    f_x = X @ rng.normal(size=d) + 0.25 * (X[:, 0] ** 2)
    Y = tau * T + f_x + rng.normal(scale=1.0, size=n)

    return X, T, Y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="RKHS_GRR", choices=["RKHS_GRR", "NN_GRR"])
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    X, T, Y = make_synthetic_data(n=args.n, d=args.d, tau=args.tau, seed=args.seed)

    est = GRR_ATE()

    # Keep the RKHS example light by using a small hyperparameter grid.
    sigma_list = np.array([0.5, 1.0])
    lda_list = np.array([0.01, 0.1])

    # Keep the NN example reasonably fast by reducing epochs.
    nn_kwargs = {}
    if args.method == "NN_GRR":
        nn_kwargs = {
            "riesz_hidden_dim": 100,
            "riesz_max_iter": 500,
            "reg_hidden_dim": 100,
            "reg_max_iter": 500,
            "batch_size": 256,
        }

    (
        dm,
        ipw,
        aipw,
        dm_lo,
        dm_hi,
        ipw_lo,
        ipw_hi,
        aipw_lo,
        aipw_hi,
    ) = est.estimate(
        covariates=X,
        treatment=T,
        outcome=Y,
        method=args.method,
        riesz_loss="SQ",
        riesz_with_D=True,
        riesz_link_name="Linear",
        folds=2,
        num_basis=50,
        sigma_list=sigma_list,
        lda_list=lda_list,
        random_state=args.seed,
        verbose=args.verbose,
        **nn_kwargs,
    )

    print("True ATE:", args.tau)
    print("DM  :", dm, "95% CI:", (dm_lo, dm_hi))
    print("IPW :", ipw, "95% CI:", (ipw_lo, ipw_hi))
    print("AIPW:", aipw, "95% CI:", (aipw_lo, aipw_hi))


if __name__ == "__main__":
    main()
