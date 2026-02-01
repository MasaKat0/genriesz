"""Synthetic ATE example: nearest-neighbor matching (as a Riesz/LSIF special case).

Run:

    python examples/ate_synthetic_nn_matching.py

This script illustrates the connection (proved in the paper) between
nearest-neighbor (NN) matching and squared-loss Riesz regression / LSIF.

Key idea
--------
In many matching estimators, the ATE can be written in a weighted form

    \hat\tau = (1/n) * \sum_i (2D_i - 1) * \hat w_i * Y_i,

with weights

    \hat w_i = 1 + K_M(i) / M,

where K_M(i) is the "matched-times" count: how many times unit i is selected as
one of the M nearest neighbors for units in the opposite treatment arm.

The matched-times counts can be written using the *catchment-area* indicator
basis

    phi_j(z) = 1[ c_j \in NN_M(z) ],

implemented here as :class:`genriesz.KNNCatchmentBasis`.

Notes
-----
This example computes the classical matching-style IPW estimate and a naive
Wald-style standard error. (The Abadie-Imbens variance correction is not
implemented here.)
"""

from __future__ import annotations

import numpy as np

from genriesz import KNNCatchmentBasis


def make_synthetic_data(n: int = 2000, d: int = 5, seed: int = 0):
    """Simple synthetic binary-treatment setup with a constant ATE."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))

    logits = 0.6 * Z[:, 0] - 0.4 * Z[:, 1] + 0.2 * Z[:, 2]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n).astype(int)

    tau = 1.0
    mu0 = np.sin(Z[:, 0]) + 0.25 * Z[:, 1] ** 2
    Y = mu0 + tau * D + rng.normal(scale=1.0, size=n)
    return Z, D, Y, tau


def nn_matching_weights(Z: np.ndarray, D: np.ndarray, *, M: int = 1) -> np.ndarray:
    """Compute matching weights w_i = 1 + K_M(i)/M.

    For treated units, K_M(i) counts how many *control* units select i among
    their M nearest treated neighbors. For control units, K_M(i) counts how
    many treated units select i among their M nearest control neighbors.
    """

    Z = np.asarray(Z, dtype=float)
    D = np.asarray(D, dtype=int).reshape(-1)
    if Z.ndim != 2:
        raise ValueError("Z must be 2D (n,d).")
    n = len(Z)
    if len(D) != n:
        raise ValueError("Z and D must have the same number of rows.")

    M = int(M)
    if M <= 0:
        raise ValueError("M must be >= 1")

    idx_t = np.flatnonzero(D == 1)
    idx_c = np.flatnonzero(D == 0)
    if len(idx_t) == 0 or len(idx_c) == 0:
        raise ValueError("Both treatment arms must be nonempty.")
    if M > len(idx_t) or M > len(idx_c):
        raise ValueError(
            f"M={M} is too large for the group sizes: n_treated={len(idx_t)}, n_control={len(idx_c)}."
        )

    w = np.empty(n, dtype=float)

    # --- Treated weights: count how many controls match to each treated unit.
    # Centers are treated; queries are controls.
    basis_t = KNNCatchmentBasis(n_neighbors=M).fit(Z[idx_t])
    Phi_ct = basis_t(Z[idx_c])  # (n_control, n_treated), each row has M ones
    K_t = Phi_ct.sum(axis=0)
    w[idx_t] = 1.0 + K_t / float(M)

    # --- Control weights: count how many treated match to each control unit.
    basis_c = KNNCatchmentBasis(n_neighbors=M).fit(Z[idx_c])
    Phi_tc = basis_c(Z[idx_t])  # (n_treated, n_control)
    K_c = Phi_tc.sum(axis=0)
    w[idx_c] = 1.0 + K_c / float(M)

    return w


def main() -> None:
    Z, D, Y, tau = make_synthetic_data(n=2000, d=5, seed=0)
    M = 1

    w = nn_matching_weights(Z, D, M=M)
    scores = (2.0 * D - 1.0) * w * Y
    ate_hat = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))

    print("True ATE (by construction):", tau)
    print(f"NN matching (M={M}) ATE:", ate_hat)
    print("Naive SE (treating weights as fixed):", se)
    print("Naive 95% CI:", (ate_hat - 1.96 * se, ate_hat + 1.96 * se))


if __name__ == "__main__":
    main()
