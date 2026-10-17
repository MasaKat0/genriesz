"""E-19: five-fold ARW for the average marginal effect under DGP19Q."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import study
from .seeds import CONFIRMATORY_ENTROPY, Seeds, fold_ids

EXP = 19
R, K = 500, 5
SQRT3 = np.sqrt(3.0)
N_VALUES = (1000, 4000)
CELLS = [[n] for n in N_VALUES]
ARM_LABELS = ("SQ|ARW_cf",)


def gr():
    import genriesz

    return genriesz


def mu(Z):
    return 0.5 * Z[:, 0] + 0.3 * np.sin(Z[:, 1])


def gamma0(X):
    d, z1, z2 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(d) + 0.5 * d * z1 + z2**2


def dgamma0(X):
    """``m(W, gamma_0) = d gamma_0 / d d``."""
    return np.cos(X[:, 0]) + 0.5 * X[:, 1]


def alpha0_q(X):
    """DGP19Q: ``-d/dd log f(d | z) = (d - mu) + 0.4 d^3``."""
    d = X[:, 0]
    return d + 0.4 * d**3 - 0.5 * X[:, 1] - 0.3 * np.sin(X[:, 2])


def draw_q(rng, n):
    """DGP19Q. Z ~ Unif(-sqrt3, sqrt3)^2; D | Z by rejection: a proposal ``N(mu, 1)`` is
    accepted with probability ``exp(-0.1 d^4)``; every pending observation draws one
    proposal and one uniform per round (in index order) until all are accepted; then
    ``eps ~ N(0, 1)``."""
    Z = rng.uniform(-SQRT3, SQRT3, size=(n, 2))
    m = mu(Z)
    D = np.empty(n)
    pending = np.arange(n)
    while pending.size:
        prop = m[pending] + rng.standard_normal(pending.size)
        accept = rng.uniform(size=pending.size) < np.exp(-0.1 * prop**4)
        D[pending[accept]] = prop[accept]
        pending = pending[~accept]
    X = np.column_stack([D, Z])
    Y = gamma0(X) + rng.standard_normal(n)
    return X, Y


def feat_alpha(X):
    X = np.atleast_2d(X)
    d = X[:, 0]
    return np.column_stack([np.ones(len(X)), d, d**3, X[:, 1], np.sin(X[:, 2])])


def dfeat_alpha(X, coordinate=0):
    if coordinate != 0:
        raise ValueError("E-19 differentiates in d (coordinate 0) only")
    X = np.atleast_2d(X)
    d = X[:, 0]
    z = np.zeros(len(X))
    return np.column_stack([z, np.ones(len(X)), 3.0 * d**2, z, z])


def feat_gamma(X):
    X = np.atleast_2d(X)
    d, z1, z2 = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([np.ones(len(X)), np.sin(d), d, d * z1, z1, z2, z2**2])


def dfeat_gamma(X):
    X = np.atleast_2d(X)
    d, z1 = X[:, 0], X[:, 1]
    z = np.zeros(len(X))
    return np.column_stack([z, np.cos(d), np.ones(len(X)), z1, z, z, z])


def alpha_basis():
    return gr().CallableBasis(feat_alpha, derivative=dfeat_alpha)


def _gl(a, b, points):
    x, w = np.polynomial.legendre.leggauss(points)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def quadrature_q(z_points, d_nodes):
    """Population rows of DGP19Q on a tensor rule: Gauss-Legendre in each z (uniform
    density) times the given d nodes and weights, with the normalized conditional
    density of D. Returns ``X`` and the probability weights."""
    zx, zw = _gl(-SQRT3, SQRT3, z_points)
    zw = zw / (2.0 * SQRT3)
    dx, dw = d_nodes
    Z1, Z2 = np.meshgrid(zx, zx, indexing="ij")
    Z = np.column_stack([Z1.ravel(), Z2.ravel()])
    wz = np.outer(zw, zw).ravel()
    m = mu(Z)
    logk = -((dx[None, :] - m[:, None]) ** 2) / 2.0 - 0.1 * dx[None, :] ** 4
    k = np.exp(logk) * dw[None, :]
    k = k / k.sum(axis=1, keepdims=True)  # f(d | z) dd, normalized per z
    w = (wz[:, None] * k).ravel()
    X = np.column_stack([np.repeat(dx[None, :], len(Z), 0).ravel(), np.repeat(Z, len(dx), 0)])
    return X, w


@lru_cache(maxsize=1)
def true_target():
    # Only the target integral is needed; no fitted population nuisance or gate.
    nodes, weights = _gl(-10.0, 10.0, 160)
    X, w = quadrature_q(24, (nodes, weights))
    return float(w @ dgamma0(X))


def replicate(task):
    cell, rep, entropy = task
    n = CELLS[cell][0]
    seeds = Seeds(EXP, entropy)
    X, Y = draw_q(seeds.data(cell, rep), n)
    folds = fold_ids(n, K, seeds.folds(cell, rep))
    log = study.FitLog()
    g = gr()
    res = g.grr_functional(
        X=X,
        Y=Y,
        m=g.AMEFunctional(0),
        basis=alpha_basis(),
        generator=g.SquaredGenerator(C=0.0),
        riesz_penalty=None,
        riesz_lam=0.0,
        outcome_basis=g.CallableBasis(
            feat_gamma, derivative=lambda X, coordinate=0: dfeat_gamma(X)
        ),
        outcome_models="shared",
        outcome_link="identity",
        outcome_penalty=None,
        outcome_lam=0.0,
        fold_ids=folds,
        estimators=("arw",),
        expose_alpha_values=True,
        evaluate_m_alpha=False,
        fit_hook=log.hook,
    )
    record = study.functional_record(res, log)
    record.pop("alpha", None)
    return {"arms": {ARM_LABELS[0]: record}, "folds": folds.tolist()}


def tasks_for(entropy=CONFIRMATORY_ENTROPY, reps=None, cells=None):
    import sys

    return study.tasks_for(sys.modules[__name__], entropy, reps, cells)


def summarise(raw):
    return study.summarise(raw, true_target())
