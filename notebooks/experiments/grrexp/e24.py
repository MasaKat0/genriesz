"""E-24: controlled nonlinear benchmark (B type; H24-Base, H24-GRR; U19).

Design: doc/2026-10-07_experiment_registration.md (parent repository), E-24, §1.2,
§1.4 (ARW, TMLE, AutoDML-lasso), §1.5 and §1.9 (bootstrap family 7). U28: the
document is a design, not a binding registration; the choices it leaves open are
recorded in ``docs/spec/02_data-and-experiments.md`` §2.1 ("E-24 の" rows) and in
the docstrings below.

There are no Stage 0 predictions (B type). Stage 0 checks the design constants by
quadrature and records ``E[alpha0^2]`` and the efficiency bound.

Arms (all cross-fitted with ``K = 5``; one common outcome model except DML-GBM):

- GRR: SQ, UKL(C=1), BKL(C=1), BP(0.5, C=0), offset ``alpha_ref = +-2``, ridge
  ``lambda`` tuned on pilot data, Nystrom features with treatment interaction;
- RieszNet: SQ Riesz loss, one hidden ReLU layer of 100 units on the raw ``(D, Z)``;
- AutoDML: the CNS lasso procedure of §1.4 on the degree-2 dictionary (``p = 56``);
- EB: UKL(C=0), ``lambda = 0``, degree-2 polynomials with treatment interaction;
- LogitAIPW: l2-penalised logistic propensity on the Nystrom features;
- DML-GBM: gradient boosting for the propensity (Z) and the outcome (D, Z).

Each arm reports the cross-fitted ARW (the primary estimator, compared in the
families) and the cross-fitted TMLE (descriptive).
"""

from __future__ import annotations

import collections
import itertools
import json

import numpy as np

from . import baselines, metrics
from .seeds import Seeds, fold_ids

EXP = 24
THETA0 = 1.0
N_VALUES = (1000, 2000, 4000, 8000)
CELLS = [[n] for n in N_VALUES]  # registered order
R = 1000
K = 5
CV_FOLDS = 5
PZ = 6
SQRT3 = np.sqrt(3.0)

GRR_ARMS = ("SQ", "UKL1", "BKL1", "BP_C0")
BASELINE_ARMS = ("RieszNet", "AutoDML", "EB", "LogitAIPW", "DML-GBM")
ARMS = GRR_ARMS + BASELINE_ARMS
ESTIMATORS = ("ARW_cf", "TMLE_cf")
ARM_LABELS = [f"{a}|{e}" for a in ARMS for e in ESTIMATORS]

#: registered design constants (outward-rounded)
ETA_BOUND = 3.8784610
E_RANGE = (0.0202635, 0.9797365)
ALPHA0_BOUND = 49.349747

# tuning grids (registration E-24)
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)  # GRR ridge and the common outcome ridge
WD_GRID = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)  # RieszNet weight decay
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)  # LogitAIPW inverse l2 strength
N_CENTRES = 100
EIG_FLOOR = 1e-8  # relative eigenvalue floor of K(C, C)
SAMPLE_TOL = 1e-8  # §1.3 C, ridge: rho <= 1e-8; lambda = 0 (EB): scaled 1e-8

# RieszNet (registration E-24)
NN_HIDDEN = 100
NN_LR = 1e-3
NN_EPOCHS = 300
NN_BATCH = 256

# AutoDML-lasso (§1.4)
CNS_C = (1.0, 0.1, 0.1)
CNS_LOADING_SHIFT = 0.2
CNS_OUTER = 10
CNS_OUTER_TOL = 1e-8
CNS_INNER_TOL = 1e-10
CNS_INNER_SWEEPS = 10000

# comparisons (registration E-24)
COMMON_SUCCESS_MIN = 0.95
FAMILY_LEVEL = 0.05
COVERAGE_BAND = 0.0069  # 2 MCSE at c = 0.95 and R = 1000


def gr():
    import genriesz

    return genriesz


def sign(A):
    return np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)


sign.vectorized = True


def alpha_ref(X):
    return np.where(np.atleast_2d(X)[:, 0] == 1, 2.0, -2.0)


# ---------------------------------------------------------------- DGP


def eta(Z):
    Z = np.atleast_2d(Z)
    return (
        0.7 * Z[:, 0] - 0.5 * Z[:, 1] + 0.6 * np.sin(np.pi * Z[:, 2] / SQRT3)
        + 0.4 * Z[:, 0] * Z[:, 3]
    )  # fmt: skip


def propensity(Z):
    return 1.0 / (1.0 + np.exp(-eta(Z)))


def tau(Z):
    Z = np.atleast_2d(Z)
    return 1.0 + 0.5 * np.tanh(Z[:, 0] + Z[:, 3])


def gamma0(D, Z):
    Z = np.atleast_2d(Z)
    return np.sin(Z[:, 0]) + 0.5 * Z[:, 1] ** 2 + 0.3 * Z[:, 2] * Z[:, 4] + D * tau(Z)


def draw(rng, n):
    """Z ~ Unif(-sqrt3, sqrt3)^6, D ~ Bern(e(Z)), Y = gamma0(D, Z) + N(0, 1)."""
    Z = rng.uniform(-SQRT3, SQRT3, size=(n, PZ))
    D = (rng.uniform(size=n) < propensity(Z)).astype(float)
    Y = gamma0(D, Z) + rng.standard_normal(n)
    return D, Z, Y


def _arms_rows(Z):
    n = len(Z)
    return np.column_stack([np.ones(n), Z]), np.column_stack([np.zeros(n), Z])


# ---------------------------------------------------------------- Stage 0


def stage0(points=(48, 64)):
    """Design checks by tensor Gauss-Legendre quadrature in ``(z1, z2, z3, z4)``.

    ``alpha0`` depends on ``Z`` only through ``eta(z1, ..., z4)``; ``tau`` only
    through ``z1 + z4``. Returns the quadrature values at both grid sizes, their
    differences, and the checks of the registered bounds (the bound of ``|eta|``
    is the exact maximum ``1.2 sqrt3 + 1.8`` of its terms, attained at a vertex).
    """
    out = {"points": {}}
    for m in points:
        x, w = np.polynomial.legendre.leggauss(m)
        z, wz = SQRT3 * x, w / 2.0
        g = np.meshgrid(z, z, z, z, indexing="ij")
        W = np.einsum("i,j,k,l->ijkl", wz, wz, wz, wz).ravel()
        Zq = np.zeros((W.size, PZ))
        for j in range(4):
            Zq[:, j] = g[j].ravel()
        e = propensity(Zq)
        t = tau(Zq)
        e_alpha2 = float(np.sum(W * (1.0 / e + 1.0 / (1.0 - e))))
        theta = float(np.sum(W * t))
        var_tau = float(np.sum(W * (t - theta) ** 2))
        out["points"][str(m)] = {
            "E_alpha0_sq": e_alpha2,
            "theta0": theta,
            "var_tau": var_tau,
            # efficiency bound of the ATE with Var(Y | X) = 1
            "V_star": e_alpha2 + var_tau,
        }
    a, b = (out["points"][str(m)] for m in points)
    out["differences"] = {k: abs(a[k] - b[k]) for k in a}
    eta_max = 1.2 * SQRT3 + 1.8
    e_lo = 1.0 / (1.0 + np.exp(eta_max))
    out["eta_max_exact"] = eta_max
    out["checks"] = {
        "eta_bound_outward": bool(ETA_BOUND >= eta_max and ETA_BOUND - eta_max < 1e-6),
        "e_range_outward": bool(E_RANGE[0] <= e_lo and E_RANGE[1] >= 1.0 - e_lo),
        "alpha0_bound_outward": bool(ALPHA0_BOUND >= 1.0 / e_lo),
        "theta0": bool(abs(b["theta0"] - THETA0) <= 1e-10),
        "quadrature_agreement": bool(all(v <= 1e-6 for v in out["differences"].values())),
    }
    out["ok"] = bool(all(out["checks"].values()))
    return out


# ---------------------------------------------------------------- features


def poly2(Z):
    """Non-constant monomials of total degree <= 2 in z1..z6, in degree-then-lexicographic
    order (z1, ..., z6, z1^2, z1 z2, ..., z6^2): 27 columns (§1.4, E-24 dictionary)."""
    Z = np.atleast_2d(Z)
    cols = [Z[:, j] for j in range(PZ)]
    cols += [Z[:, i] * Z[:, j] for i, j in itertools.combinations_with_replacement(range(PZ), 2)]
    return np.column_stack(cols)


def cns_dictionary(X):
    """b = (1, D, monomials, D x monomials), p = 56 (§1.4)."""
    X = np.atleast_2d(X)
    D, Z = X[:, 0], X[:, 1:]
    mono = poly2(Z)
    return np.column_stack([np.ones(len(X)), D, mono, D[:, None] * mono])


class Nystrom:
    """Nystrom features ``K(z, C) K(C, C)^{-1/2}`` of the Gaussian kernel.

    Fixed per ``n`` from the pilot sample: 100 centres drawn uniformly without
    replacement, ``Z`` standardised by the pilot mean and SD, bandwidth the median
    pairwise distance of the standardised pilot ``Z``, kernel
    ``exp(-|z - z'|^2 / (2 h^2))``; eigenvalues below ``1e-8`` times the largest
    are dropped (pseudo inverse square root).
    """

    def __init__(self, Z_pilot, rng):
        Z_pilot = np.asarray(Z_pilot, dtype=float)
        self.mean = Z_pilot.mean(axis=0)
        self.sd = Z_pilot.std(axis=0, ddof=1)
        S = (Z_pilot - self.mean) / self.sd
        idx = np.sort(rng.choice(len(S), size=N_CENTRES, replace=False))
        self.centres = S[idx]
        self.bandwidth = float(_median_pairwise_distance(S))
        Kcc = self._kernel(self.centres, self.centres)
        lam, U = np.linalg.eigh(Kcc)
        keep = lam >= EIG_FLOOR * lam.max()
        self.map = U[:, keep] / np.sqrt(lam[keep])
        self.dim = int(keep.sum())

    def _kernel(self, A, B):
        d2 = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2.0 * A @ B.T
        return np.exp(-np.maximum(d2, 0.0) / (2.0 * self.bandwidth**2))

    def __call__(self, Z):
        S = (np.atleast_2d(Z) - self.mean) / self.sd
        return self._kernel(S, self.centres) @ self.map

    def base(self, Z):
        """(1, Nystrom features): the per-arm features of the GRR and outcome models."""
        F = self(Z)
        return np.column_stack([np.ones(len(F)), F])


def _median_pairwise_distance(S, chunk=1000):
    """Exact median of the ``n(n-1)/2`` pairwise Euclidean distances."""
    n = len(S)
    parts = []
    for a in range(0, n, chunk):
        A = S[a : a + chunk]
        d2 = np.sum(A**2, 1)[:, None] + np.sum(S**2, 1)[None, :] - 2.0 * A @ S.T
        rows = np.arange(a, a + len(A))[:, None]
        mask = np.arange(n)[None, :] > rows
        parts.append(np.sqrt(np.maximum(d2[mask], 0.0)))
    return float(np.median(np.concatenate(parts)))


def interaction(base, X):
    X = np.atleast_2d(X)
    D = X[:, :1]
    B = base(X[:, 1:])
    return np.concatenate([D * B, (1.0 - D) * B], axis=1)


# ---------------------------------------------------------------- representers


def generator(arm):
    g = gr()
    return {
        "SQ": lambda: g.SquaredGenerator(C=0.0),
        "UKL1": lambda: g.UKLGenerator(C=1.0, branch_fn=sign),
        "BKL1": lambda: g.BKLGenerator(C=1.0, branch_fn=sign),
        "BP_C0": lambda: g.BPGenerator(omega=0.5, C=0.0, branch_fn=sign),
        "EB": lambda: g.UKLGenerator(C=0.0, branch_fn=sign),
    }[arm]()


def glm_model(arm, base, lam):
    """GRRGLM of a GRR arm (ridge ``lam``) or of EB (``lam = 0``, degree-2 basis)."""
    g = gr()
    gen = generator(arm)
    basis = g.TreatmentInteractionBasis(base_basis=g.CallableBasis(base))
    return g.GRRGLM(
        basis=basis, generator=gen, functional=g.ATEFunctional(0),
        penalty=None if lam == 0.0 else "l2", lam=float(lam),
        offset=g.offset_from_alpha(gen, alpha_ref),
    )  # fmt: skip


def poly_base(Z):
    Z = np.atleast_2d(Z)
    return np.column_stack([np.ones(len(Z)), poly2(Z)])


class Rep:
    """A fitted representer: ``alpha`` at observed rows and at both counterfactual arms."""

    def __init__(self, status, fn=None, info=None):
        self.status, self.fn, self.info = status, fn, info or {}

    def at(self, X):
        """``(alpha(X), alpha(1, Z), alpha(0, Z), status)``; a domain violation or a
        non-finite value at any of these rows is a failure (§1.3 B)."""
        X1, X0 = _arms_rows(X[:, 1:])
        return self.fn(X, X1, X0)


def fit_glm(arm, base, lam, X_tr):
    if not (np.any(X_tr[:, 0] == 1) and np.any(X_tr[:, 0] == 0)):
        return Rep("degenerate_functional")
    mdl = glm_model(arm, base, lam)
    fr = mdl.fit(X_tr, tol=SAMPLE_TOL)
    if fr.status != "ok":
        return Rep(str(fr.status))

    def fn(X, X1, X0):
        a, outside, nonfinite = mdl.classify(np.vstack([X, X1, X0]))
        if np.any(nonfinite):
            return None, None, None, "nonfinite"
        if np.any(outside):
            return None, None, None, "domain_prediction"
        n = len(X)
        return a[:n], a[n : 2 * n], a[2 * n :], "ok"

    return Rep("ok", fn, {"n_iter": int(fr.n_iter)})


def fit_autodml(X_tr):
    """CNS Appendix A.1.1/A.2 (registration §1.4), on the degree-2 dictionary."""
    b = cns_dictionary(X_tr)
    X1, X0 = _arms_rows(X_tr[:, 1:])
    mb = cns_dictionary(X1) - cns_dictionary(X0)  # m(W, b_j) for the ATE
    n, p = b.shape
    G = b.T @ b / n
    M = mb.mean(axis=0)
    low = max(2, int(np.ceil(p / 40)))
    rho = np.zeros(p)
    rho[:low] = np.linalg.solve(G[:low, :low], M[:low])
    from scipy.stats import norm

    c1, c2, c3 = CNS_C
    r_l = c1 / np.sqrt(n) * norm.ppf(1.0 - c2 / (2.0 * p))
    outer_stops, sweeps_max = 0, 0
    for it in range(CNS_OUTER):
        resid = b * (b @ rho)[:, None] - mb  # b_j(X) b(X)' rho - m(W, b_j)
        Dl = np.sqrt(np.mean(resid**2, axis=0)) + CNS_LOADING_SHIFT
        w = 2.0 * r_l * Dl  # objective rho'G rho - 2 rho'M + sum_j w_j |rho_j|
        w[0] *= c3
        new, sweeps = _cns_coordinate_descent(G, M, w, rho.copy())
        sweeps_max = max(sweeps_max, sweeps)
        done = np.max(np.abs(new - rho)) <= CNS_OUTER_TOL
        rho = new
        outer_stops = it + 1
        if done:
            break

    def fn(X, X1, X0):
        a = cns_dictionary(X) @ rho
        a1 = cns_dictionary(X1) @ rho
        a0 = cns_dictionary(X0) @ rho
        if not np.all(np.isfinite(np.concatenate([a, a1, a0]))):
            return None, None, None, "nonfinite"
        return a, a1, a0, "ok"

    info = {
        "outer_iterations": outer_stops,
        "outer_cap_reached": int(outer_stops == CNS_OUTER and not done),
        "inner_sweeps_max": sweeps_max,
    }
    return Rep("ok", fn, info)


def _cns_coordinate_descent(G, M, w, rho):
    """min rho'G rho - 2 rho'M + sum_j w_j |rho_j| by cyclic soft-thresholding (CNS A.2).

    Stops when one sweep changes no coefficient by more than 1e-10, or after 10000
    sweeps (§1.4).
    """
    p = len(M)
    Grho = G @ rho
    diag = np.diag(G).copy()
    for sweep in range(1, CNS_INNER_SWEEPS + 1):
        change = 0.0
        for j in range(p):
            old = rho[j]
            z = M[j] - (Grho[j] - diag[j] * old)
            new = np.sign(z) * max(abs(z) - w[j] / 2.0, 0.0) / diag[j]
            if new != old:
                Grho += G[:, j] * (new - old)
                rho[j] = new
                change = max(change, abs(new - old))
        if change <= CNS_INNER_TOL:
            return rho, sweep
    return rho, CNS_INNER_SWEEPS


def fit_logit(base_nys, C, X_tr):
    """l2-penalised logistic propensity on the Nystrom features (``C`` tuned)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=float(C), penalty="l2", solver="lbfgs", max_iter=10000, tol=1e-10)
    clf.fit(base_nys(X_tr[:, 1:]), X_tr[:, 0])

    def fn(X, X1, X0):
        e = clf.predict_proba(base_nys(X[:, 1:]))[:, 1]
        if not np.all((e > 1e-12) & (e < 1 - 1e-12)):
            return None, None, None, "propensity_range"
        D = X[:, 0]
        return D / e - (1 - D) / (1 - e), 1.0 / e, -1.0 / (1.0 - e), "ok"

    return Rep("ok", fn)


def fit_gbm_propensity(params, seed, X_tr):
    clf = baselines._gbm_classifier(params, seed).fit(X_tr[:, 1:], X_tr[:, 0])

    def fn(X, X1, X0):
        e = clf.predict_proba(X[:, 1:])[:, 1]
        if not np.all((e > 0) & (e < 1)):
            return None, None, None, "propensity_range"
        D = X[:, 0]
        return D / e - (1 - D) / (1 - e), 1.0 / e, -1.0 / (1.0 - e), "ok"

    return Rep("ok", fn)


def _torch():
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def fit_riesznet(wd, seed, X_tr):
    """SQ Riesz loss ``mean(alpha(X)^2 - 2 (alpha(1, Z) - alpha(0, Z)))`` with a one-hidden-
    layer ReLU network (100 units) on the raw ``(D, Z)``; Adam (lr 1e-3, weight decay
    ``wd``), 300 epochs, mini-batches of 256 in a seeded random order, float64."""
    torch = _torch()
    gen = torch.Generator().manual_seed(int(seed))
    net = torch.nn.Sequential(
        torch.nn.Linear(1 + PZ, NN_HIDDEN), torch.nn.ReLU(), torch.nn.Linear(NN_HIDDEN, 1)
    ).double()
    with torch.no_grad():
        for prm in net.parameters():  # default PyTorch init, drawn from the seeded generator
            if prm.dim() == 2:
                bound = 1.0 / np.sqrt(prm.shape[1])
                prm.uniform_(-bound, bound, generator=gen)
            else:
                prm.uniform_(-bound, bound, generator=gen)
    opt = torch.optim.Adam(net.parameters(), lr=NN_LR, weight_decay=float(wd))
    X = torch.as_tensor(X_tr, dtype=torch.float64)
    X1 = X.clone()
    X1[:, 0] = 1.0
    X0 = X.clone()
    X0[:, 0] = 0.0
    n = len(X)
    for _ in range(NN_EPOCHS):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, NN_BATCH):
            idx = perm[s : s + NN_BATCH]
            loss = (
                net(X[idx]).squeeze(1) ** 2 - 2.0 * (net(X1[idx]) - net(X0[idx])).squeeze(1)
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    def fn(Xe, Xe1, Xe0):
        with torch.no_grad():
            a = [
                net(torch.as_tensor(r, dtype=torch.float64)).squeeze(1).numpy()
                for r in (Xe, Xe1, Xe0)
            ]
        if not all(np.all(np.isfinite(v)) for v in a):
            return None, None, None, "nonfinite"
        return a[0], a[1], a[2], "ok"

    return Rep("ok", fn)


# ---------------------------------------------------------------- outcome models


class Ridge:
    """Least squares on the Nystrom features with treatment interaction, ridge ``lam``
    on every coefficient except the two arm intercepts: ``mean (y - F b)^2 + lam |b|^2``."""

    def __init__(self, base, lam, X_tr, Y_tr):
        F = interaction(base, X_tr)
        n, q = F.shape
        pen = np.full(q, float(lam))
        half = q // 2
        pen[0] = pen[half] = 0.0  # D * 1 and (1 - D) * 1
        A = F.T @ F / n + np.diag(pen)
        self.base = base
        self.coef = np.linalg.solve(A, F.T @ Y_tr / n)

    def __call__(self, X):
        return interaction(self.base, X) @ self.coef


class GBMOutcome:
    def __init__(self, params, seed, X_tr, Y_tr):
        self.reg = baselines._gbm_regressor(params, seed).fit(X_tr, Y_tr)

    def __call__(self, X):
        return self.reg.predict(X)


# ---------------------------------------------------------------- tuning on pilot data


def riesz_criterion(a, a1, a0):
    """Held-out squared Riesz criterion ``sum alpha(X)^2 - 2 m(W, alpha)`` (sum, not mean)."""
    return float(np.sum(a**2 - 2.0 * (a1 - a0)))


def _cv_select(grid, score_fn, tie):
    """``score_fn(value) -> (total held-out loss, failures)``. A value with any failed CV
    fit or evaluation (a status other than ok, any recorded warning, a domain or
    non-finite prediction) is not eligible; ``failures`` lists ``[fold, status,
    warnings]`` of each. ``tie``: ``"larger"`` or ``"smaller"`` grid value wins exact
    ties. Returns ``(choice or None, record)``."""
    record = {}
    for v in grid:
        score, failures = score_fn(v)
        record[repr(v)] = {"score": score, "failures": failures}
    ok = {v: record[repr(v)]["score"] for v in grid if not record[repr(v)]["failures"]}
    ok = {v: s for v, s in ok.items() if s is not None and np.isfinite(s)}
    if not ok:
        return None, record
    best = min(ok.values())
    winners = [v for v, s in ok.items() if s == best]
    return (max(winners) if tie == "larger" else min(winners)), record


def _cv_failures(record):
    return sum(len(r["failures"]) for r in record.values())


def tune(cell, entropy):
    """Pilot data (stream 5, size ``n``), its 5-fold CV split and the Nystrom centres all
    come from the stream-5 generator, in that order; the RieszNet and GBM seeds are
    the next draws of the same generator. Returns the frozen per-``n`` choices with the
    held-out score and the failed fits of every candidate."""
    from sklearn.metrics import log_loss

    (n,) = CELLS[cell]
    rng = Seeds(EXP, entropy=entropy).pilot(cell)
    D, Z, Y = draw(rng, n)
    X = np.column_stack([D, Z])
    cv = fold_ids(n, CV_FOLDS, rng)
    nys = Nystrom(Z, rng)
    nn_seed = int(rng.integers(2**31 - 1))
    gbm_seed = int(rng.integers(2**31 - 1))
    base = nys.base
    folds = [(cv != k, cv == k) for k in range(CV_FOLDS)]
    warn = collections.Counter()

    def rec(fn):
        res, counts, _ = baselines.run_recording_warnings(fn)
        warn.update(counts)
        return res, counts

    out = {"n": n, "nystrom_dim": nys.dim, "bandwidth": nys.bandwidth, "lambda": {}, "scores": {}}

    def cv_score(fold_loss):
        """``fold_loss(tr, te) -> (status, loss)``, run under warning recording."""
        total, failures = 0.0, []
        for k, (tr, te) in enumerate(folds):
            (status, loss), counts = rec(lambda tr=tr, te=te: fold_loss(tr, te))
            if status == "ok" and counts:
                status = "warning"
            if status != "ok":
                failures.append([k, status, counts])
                continue
            total += loss
        return (None if failures else total), failures

    def riesz_loss(make):
        def fold_loss(tr, te):
            rep = make(X[tr])
            if rep.status != "ok":
                return rep.status, None
            a, a1, a0, st = rep.at(X[te])
            return (st, None) if st != "ok" else ("ok", riesz_criterion(a, a1, a0))

        return cv_score(fold_loss)

    for arm in GRR_ARMS:
        lam, sc = _cv_select(
            LAMBDA_GRID,
            lambda lam, arm=arm: riesz_loss(lambda Xt: fit_glm(arm, base, lam, Xt)),
            "larger",
        )
        out["scores"][arm] = sc
        if lam is None:
            raise RuntimeError(f"E-24 tuning: no eligible lambda for {arm} at n = {n}: {sc}")
        out["lambda"][arm] = lam

    wd, sc = _cv_select(
        WD_GRID, lambda wd: riesz_loss(lambda Xt: fit_riesznet(wd, nn_seed, Xt)), "larger"
    )
    out["scores"]["RieszNet"] = sc
    if wd is None:
        raise RuntimeError(f"E-24 tuning: no eligible weight decay at n = {n}: {sc}")
    out["weight_decay"] = wd

    def logit_loss(C):
        from sklearn.linear_model import LogisticRegression

        def fold_loss(tr, te):
            clf = LogisticRegression(
                C=float(C), penalty="l2", solver="lbfgs", max_iter=10000, tol=1e-10
            )
            clf.fit(nys(Z[tr]), D[tr])
            p = clf.predict_proba(nys(Z[te]))[:, 1]
            return "ok", float(log_loss(D[te], p, labels=[0, 1]) * te.sum())

        return cv_score(fold_loss)

    c, sc = _cv_select(C_GRID, logit_loss, "smaller")
    out["scores"]["LogitAIPW"] = sc
    if c is None:
        raise RuntimeError(f"E-24 tuning: no eligible C at n = {n}: {sc}")
    out["logit_C"] = c

    def outcome_loss(lam):
        def fold_loss(tr, te):
            pred = Ridge(base, lam, X[tr], Y[tr])(X[te])
            if not np.all(np.isfinite(pred)):
                return "nonfinite", None
            return "ok", float(np.sum((pred - Y[te]) ** 2))

        return cv_score(fold_loss)

    lam, sc = _cv_select(LAMBDA_GRID, outcome_loss, "larger")
    out["scores"]["outcome"] = sc
    if lam is None:
        raise RuntimeError(f"E-24 tuning: no eligible outcome lambda at n = {n}: {sc}")
    out["outcome_lambda"] = lam

    # GBM (the E-12 grid, ties to the earlier grid point), each model, candidate and fold
    # under the same rule as above: a warned or failed CV fit makes the candidate ineligible
    DZ = np.column_stack([D, Z])
    order = {params: i for i, params in enumerate(baselines.GBM_GRID)}

    def gbm_prop_loss(params):
        def fold_loss(tr, te):
            clf = baselines._gbm_classifier(params, gbm_seed).fit(Z[tr], D[tr])
            p = clf.predict_proba(Z[te])[:, 1]
            return "ok", float(log_loss(D[te], p, labels=[0, 1]) * te.sum())

        return cv_score(fold_loss)

    def gbm_out_loss(params):
        def fold_loss(tr, te):
            pred = baselines._gbm_regressor(params, gbm_seed).fit(DZ[tr], Y[tr]).predict(DZ[te])
            return "ok", float(np.sum((pred - Y[te]) ** 2))

        return cv_score(fold_loss)

    def earliest(grid, loss):
        choice, record = _cv_select(grid, loss, "smaller")
        if choice is None:
            return None, record
        best = record[repr(choice)]["score"]
        tied = [
            v for v in grid if not record[repr(v)]["failures"] and record[repr(v)]["score"] == best
        ]
        return min(tied, key=order.get), record

    prop, sc = earliest(baselines.GBM_GRID, gbm_prop_loss)
    out["scores"]["GBM-propensity"] = sc
    outc, sc2 = earliest(baselines.GBM_GRID, gbm_out_loss)
    out["scores"]["GBM-outcome"] = sc2
    if prop is None or outc is None:
        raise RuntimeError(f"E-24 tuning: no eligible GBM candidate at n = {n}")
    out["gbm"] = {"propensity": list(prop), "outcome": list(outc), "seed": gbm_seed}
    out["nn_seed"] = nn_seed
    out["warnings"] = dict(warn)
    out["cv_failures"] = int(sum(_cv_failures(s) for s in out["scores"].values()))
    out["nystrom"] = {
        "mean": nys.mean.tolist(), "sd": nys.sd.tolist(), "centres": nys.centres.tolist(),
        "bandwidth": nys.bandwidth,
    }  # fmt: skip
    return out


class Frozen:
    """The per-``n`` tuning choices, rebuilt in a worker from the JSON of :func:`tune`."""

    def __init__(self, t):
        self.t = t
        nys = Nystrom.__new__(Nystrom)
        nys.mean = np.asarray(t["nystrom"]["mean"])
        nys.sd = np.asarray(t["nystrom"]["sd"])
        nys.centres = np.asarray(t["nystrom"]["centres"])
        nys.bandwidth = float(t["nystrom"]["bandwidth"])
        Kcc = nys._kernel(nys.centres, nys.centres)
        lam, U = np.linalg.eigh(Kcc)
        keep = lam >= EIG_FLOOR * lam.max()
        nys.map = U[:, keep] / np.sqrt(lam[keep])
        nys.dim = int(keep.sum())
        self.nys = nys


# ---------------------------------------------------------------- one replication


def _failed(status, **extra):
    return {"estimate": float("nan"), "se": float("nan"), "status": status, **extra}


def _scores(D, Y, a, a1, a0, g, g1, g0):
    """ARW and TMLE (§1.4) from cross-fitted alpha and gamma (each length n)."""
    n = len(Y)
    psi = g1 - g0 + a * (Y - g)
    theta = float(np.mean(psi))
    arw = (theta, float(np.std(psi - theta) / np.sqrt(n)))
    eps = float(np.sum(a * (Y - g)) / np.sum(a**2))
    gs, gs1, gs0 = g + eps * a, g1 + eps * a1, g0 + eps * a0
    th_t = float(np.mean(gs1 - gs0))
    psi_t = gs1 - gs0 + a * (Y - gs) - th_t
    tmle = (th_t, float(np.std(psi_t) / np.sqrt(n)))
    return arw, tmle


def replicate(task):
    """One replication of one cell, every arm. ``task = (cell, rep, entropy, tuning_json)``."""
    cell, rep, entropy, tj = task
    (n,) = CELLS[cell]
    fz = Frozen(json.loads(tj))
    t = fz.t
    seeds = Seeds(EXP, entropy=entropy)
    D, Z, Y = draw(seeds.data(cell, rep), n)
    X = np.column_stack([D, Z])
    folds = fold_ids(n, K, seeds.folds(cell, rep))
    brng = seeds.baseline(cell, rep)
    nn_seeds = [int(s) for s in brng.integers(2**31 - 1, size=K)]
    gbm_seed = int(brng.integers(2**31 - 1))
    base = fz.nys.base
    nys = fz.nys
    gbm_p, gbm_o = tuple(t["gbm"]["propensity"]), tuple(t["gbm"]["outcome"])
    makers = {
        **{
            arm: (lambda Xt, k, arm=arm: fit_glm(arm, base, t["lambda"][arm], Xt))
            for arm in GRR_ARMS
        },
        "RieszNet": lambda Xt, k: fit_riesznet(t["weight_decay"], nn_seeds[k], Xt),
        "AutoDML": lambda Xt, k: fit_autodml(Xt),
        "EB": lambda Xt, k: fit_glm("EB", poly_base, 0.0, Xt),
        "LogitAIPW": lambda Xt, k: fit_logit(nys, t["logit_C"], Xt),
        "DML-GBM": lambda Xt, k: fit_gbm_propensity(gbm_p, gbm_seed, Xt),
    }
    A = {arm: np.full((3, n), np.nan) for arm in ARMS}
    G = {key: np.full((3, n), np.nan) for key in ("common", "gbm")}
    status = {arm: "ok" for arm in ARMS}
    fold_status = {arm: [] for arm in ARMS}
    warns = {arm: collections.Counter() for arm in ARMS}
    info = {arm: collections.defaultdict(int) for arm in ARMS}
    out_warn = collections.Counter()
    X1, X0 = _arms_rows(Z)
    out_status = {"common": "ok", "gbm": "ok"}
    for k in range(K):
        tr, te = folds != k, folds == k
        if not (np.any(D[tr] == 1) and np.any(D[tr] == 0)):
            for arm in ARMS:
                if status[arm] == "ok":
                    fold_status[arm].append([k, "degenerate_functional"])
                    status[arm] = "degenerate_functional"
            break
        for key, make in (
            ("common", lambda tr=tr: Ridge(base, t["outcome_lambda"], X[tr], Y[tr])),
            ("gbm", lambda tr=tr: GBMOutcome(gbm_o, gbm_seed, X[tr], Y[tr])),
        ):
            if out_status[key] != "ok":
                continue

            def predict(make=make, te=te):
                mdl = make()
                return [mdl(X[te]), mdl(X1[te]), mdl(X0[te])]

            preds, counts, converged = baselines.run_recording_warnings(predict)
            out_warn.update(counts)
            if not converged:
                out_status[key] = "convergence_warning"
            elif not all(np.all(np.isfinite(p)) for p in preds):
                out_status[key] = "outcome_nonfinite"
            else:
                G[key][:, te] = preds
                continue
            for arm in ARMS:
                if status[arm] == "ok" and (key == "gbm") == (arm == "DML-GBM"):
                    fold_status[arm].append([k, f"outcome {out_status[key]}"])
                    status[arm] = out_status[key]
        for arm in ARMS:
            if status[arm] != "ok":
                continue
            rep_, counts, converged = baselines.run_recording_warnings(
                lambda arm=arm, tr=tr, k=k: makers[arm](X[tr], k)
            )
            warns[arm].update(counts)
            st = rep_.status
            if st == "ok" and not converged:
                st = "convergence_warning"
            if st == "ok":
                (a, a1, a0, st), counts, conv2 = baselines.run_recording_warnings(
                    lambda rep_=rep_, te=te: rep_.at(X[te])
                )
                warns[arm].update(counts)
                if st == "ok" and not conv2:
                    st = "convergence_warning"
                if st == "ok":
                    A[arm][:, te] = [a, a1, a0]
                for key, v in rep_.info.items():
                    info[arm][key] = (
                        max(info[arm][key], v) if key != "outer_cap_reached" else info[arm][key] + v
                    )
            fold_status[arm].append([k, st])
            if st != "ok":
                status[arm] = st
    out = {}
    for arm in ARMS:
        common = {
            "fold_status": json.dumps(fold_status[arm]),
            "warnings": json.dumps({"fit": dict(warns[arm]), "outcome": dict(out_warn)}),
            "n_warnings": int(sum(warns[arm].values()) + sum(out_warn.values())),
            "info": json.dumps(dict(info[arm])),
        }
        if status[arm] != "ok":
            for e in ESTIMATORS:
                out[f"{arm}|{e}"] = _failed(status[arm], **common)
            continue
        g = G["gbm"] if arm == "DML-GBM" else G["common"]
        a, a1, a0 = A[arm]
        (arw, tmle), counts, _ = baselines.run_recording_warnings(
            lambda a=a, a1=a1, a0=a0, g=g: _scores(D, Y, a, a1, a0, *g)
        )
        warns[arm].update(counts)
        common["warnings"] = json.dumps({"fit": dict(warns[arm]), "outcome": dict(out_warn)})
        common["n_warnings"] = int(sum(warns[arm].values()) + sum(out_warn.values()))
        extra = {"max_abs_alpha": float(np.max(np.abs(a))), "ess": metrics.ess(a)}
        for e, (th, se) in (("ARW_cf", arw), ("TMLE_cf", tmle)):
            if np.isfinite(th) and np.isfinite(se):
                out[f"{arm}|{e}"] = {"estimate": th, "se": se, "status": "ok", **extra, **common}
            else:
                out[f"{arm}|{e}"] = _failed("nonfinite", **common)
    return out


def tasks_for(entropy, reps, tunings, cells=None):
    cells = range(len(CELLS)) if cells is None else cells
    return [(ci, r, entropy, tunings[ci]) for ci in cells for r in range(reps)]


# ---------------------------------------------------------------- aggregation

NUMERIC_FIELDS = ("estimate", "se", "max_abs_alpha", "ess", "n_warnings")
TEXT_FIELDS = ("status", "warnings", "fold_status", "info")


def raw_frame(tasks, results):
    import pandas as pd

    rows = []
    for (ci, rep, _, _), res in zip(tasks, results, strict=True):
        (n,) = CELLS[ci]
        for label in ARM_LABELS:
            r = res[label]
            rows.append(
                {
                    "cell": ci, "n": n, "rep": rep, "arm": label,
                    **{k: float(r.get(k, np.nan)) for k in NUMERIC_FIELDS},
                    **{k: str(r.get(k, "")) for k in TEXT_FIELDS},
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


def total_warnings(raw):
    # one count per arm record; the outcome warnings are repeated across arms, so count
    # them from the ARW rows of one arm per replication only once
    fit = 0
    outcome = 0
    for lab, g in raw.groupby("arm"):
        if not lab.endswith("ARW_cf"):
            continue
        for w in g["warnings"]:
            d = json.loads(w)
            fit += sum(d["fit"].values())
        if lab == f"{ARMS[0]}|ARW_cf":
            outcome = sum(sum(json.loads(w)["outcome"].values()) for w in g["warnings"])
    return int(fit + outcome)


def summarise(raw):
    """§1.5 metrics per cell x arm|estimator, in the registered order."""
    import pandas as pd

    from .e12 import POSITIVE_VARIANCE, moment_status

    rows = []
    for ci, (n,) in enumerate(CELLS):
        for label in ARM_LABELS:
            g = raw[(raw["cell"] == ci) & (raw["arm"] == label)].sort_values("rep")
            if g.empty:
                continue
            ok = (g["status"] == "ok").to_numpy()
            est, se = g["estimate"].to_numpy(), g["se"].to_numpy()
            arm, _, estimator = label.partition("|")
            row = {"cell": ci, "n": n, "arm": label, "method": arm, "estimator": estimator}
            row.update(metrics.failure_rate(ok))
            row.update(metrics.coverage(est, se, ok, THETA0))
            row["coverage_close_to_nominal"] = bool(abs(row["coverage"] - 0.95) <= COVERAGE_BAND)
            row["status_counts"] = json.dumps(
                dict(sorted(collections.Counter(g["status"]).items()))
            )
            infos = [json.loads(s) for s in g["info"] if s]
            if any("outer_cap_reached" in d for d in infos):
                row["cns_outer_cap_folds"] = int(sum(d.get("outer_cap_reached", 0) for d in infos))
                row["cns_inner_sweeps_max"] = int(max(d.get("inner_sweeps_max", 0) for d in infos))
            if ok.sum() >= 1:
                row.update(metrics.ci_length(se, ok))
                row.update(metrics.max_weight_summary(g["max_abs_alpha"].to_numpy(), ok))
                row["ess_median"] = float(g.loc[ok, "ess"].median())
            if ok.sum() >= 2:
                row.update(metrics.bias(est, ok, THETA0))
                row["rmse"] = metrics.rmse(est, ok, THETA0)
                row["sd"] = float(est[ok].std(ddof=1))
                row["root_n_sd"] = float(np.sqrt(n) * row["sd"])
                ms = row["moment_status"] = moment_status(est[ok])
                if ms in POSITIVE_VARIANCE:
                    row["se_ratio"] = metrics.se_ratio(est, se, ok)
                if ms == "ok":
                    row.update(metrics.root_n_sd(est, ok, n))
            rows.append(row)
    return pd.DataFrame(rows)


def comparisons():
    """The registered comparisons in call order: H24-Base (for each n, GRR arm, baseline),
    then H24-GRR (for each n, generator pair in GRR order)."""
    base = [("H24-Base", n, a, b) for n in N_VALUES for a in GRR_ARMS for b in BASELINE_ARMS]
    pairs = [("H24-GRR", n, a, b) for n in N_VALUES for a, b in itertools.combinations(GRR_ARMS, 2)]
    return base + pairs


def compare(raw, seeds):
    """Paired bootstrap tests of ``log(RMSE_A / RMSE_B)`` of the ARW estimators (family 7).

    A comparison whose common-success rate is below 95% is not evaluable: it is not
    tested, consumes no draws, and stays in its family as a non-rejecting entry
    (``p = 1``) so that the Holm multiplicity does not depend on the outcome.
    Returns a list of dicts in :func:`comparisons` order.
    """
    from .bootstrap import FamilyBootstrap, centred_pvalue, log_rmse_ratio_counts

    boot = FamilyBootstrap(seeds, 7)
    arw = raw[raw["arm"].str.endswith("|ARW_cf")]
    present = set(int(v) for v in raw["n"].unique())
    out = []
    for fam, n, a, b in comparisons():
        if n not in present:  # the per-cell aggregation of Stage 0.5
            continue
        ga = arw[(arw["n"] == n) & (arw["arm"] == f"{a}|ARW_cf")].sort_values("rep")
        gb = arw[(arw["n"] == n) & (arw["arm"] == f"{b}|ARW_cf")].sort_values("rep")
        both = (ga["status"].to_numpy() == "ok") & (gb["status"].to_numpy() == "ok")
        rate = float(both.mean())
        rec = {"family": fam, "n": n, "A": a, "B": b, "common_success": rate}
        if rate < COMMON_SUCCESS_MIN:
            rec.update({"evaluable": False, "T": float("nan"), "p": 1.0})
        else:
            ea = ga["estimate"].to_numpy()[both] - THETA0
            eb = gb["estimate"].to_numpy()[both] - THETA0
            t_hat = 0.5 * (np.log(np.sum(ea**2)) - np.log(np.sum(eb**2)))
            draws = boot.replicate(log_rmse_ratio_counts(ea, eb), len(ea), mode="counts")
            rec.update(
                {
                    "evaluable": True,
                    "T": float(t_hat),
                    "rmse_A": float(np.sqrt(np.mean(ea**2))),
                    "rmse_B": float(np.sqrt(np.mean(eb**2))),
                    "p": centred_pvalue(draws, t_hat),
                }
            )
        out.append(rec)
    from .inference import holm

    for fam in ("H24-Base", "H24-GRR"):
        idx = [i for i, r in enumerate(out) if r["family"] == fam]
        if not idx:
            continue
        rej = holm([out[i]["p"] for i in idx], FAMILY_LEVEL)
        for i, r in zip(idx, rej, strict=True):
            out[i]["rejected"] = bool(r)
    return out


def family_sentences(comps):
    res = {}
    for fam, none in (
        ("H24-Base", "no difference from the registered baselines was detected"),
        ("H24-GRR", "no difference between generators was detected"),
    ):
        rows = [r for r in comps if r["family"] == fam]
        rej = [r for r in rows if r["rejected"]]
        res[fam] = {
            "tests": len(rows),
            "evaluable": int(sum(r["evaluable"] for r in rows)),
            "rejected": len(rej),
            "sentence": none if not rej else f"{len(rej)} of {len(rows)} comparisons rejected",
        }
    return res


# ---------------------------------------------------------------- outputs

LABEL = {
    "SQ": "SQ", "UKL1": "UKL ($C=1$)", "BKL1": "BKL ($C=1$)", "BP_C0": "BP ($C=0$)",
    "RieszNet": "RieszNet-type", "AutoDML": "AutoDML-lasso", "EB": "EB",
    "LogitAIPW": "LogitAIPW", "DML-GBM": "DML-GBM",
}  # fmt: skip
END = " \\\\"


def _fmt(x, d=3):
    return "--" if x is None or x != x else f"{float(x):.{d}f}"


def tables(S, comps):
    """``tab_E24`` (metrics) and ``tab_E24_comparisons``: longtable bodies (the head is
    repeated with ``\\endfirsthead``/``\\endhead``; the manuscript adds the caption)."""
    heads = [
        "Method",
        "Estimator",
        "$n$",
        "Bias",
        "SD",
        "RMSE",
        "Coverage",
        "CI length",
        "Fail \\%",
    ]
    head = ["\\hline", " & ".join(heads) + END, "\\hline"]
    lines = ["\\begin{tabular}{llrrrrrrr}", *head, "\\endfirsthead", *head, "\\endhead"]
    for arm in ARMS:
        for e in ESTIMATORS:
            for n in N_VALUES:
                r = S[(S["arm"] == f"{arm}|{e}") & (S["n"] == n)].iloc[0]
                lines.append(
                    " & ".join(
                        [
                            LABEL[arm], "ARW" if e == "ARW_cf" else "TMLE", str(n),
                            _fmt(r.get("bias")), _fmt(r.get("sd")), _fmt(r.get("rmse")),
                            _fmt(r.get("coverage")), _fmt(r.get("ci_length_mean")),
                            f"{100 * r['failure_rate']:.1f}",
                        ]
                    )
                    + END
                )  # fmt: skip
    tab = "\n".join(lines + ["\\hline", "\\end{tabular}"]) + "\n"

    heads = ["Family", "$n$", "A", "B", "Common success", "RMSE$_A$/RMSE$_B$", "$p$", "Rejected"]
    head = ["\\hline", " & ".join(heads) + END, "\\hline"]
    lines = ["\\begin{tabular}{lrllrrrl}", *head, "\\endfirsthead", *head, "\\endhead"]
    for r in comps:
        ratio = np.exp(r["T"]) if r["evaluable"] else float("nan")
        lines.append(
            " & ".join(
                [
                    r["family"], str(r["n"]), LABEL[r["A"]], LABEL[r["B"]],
                    _fmt(r["common_success"]), _fmt(ratio),
                    "--" if not r["evaluable"] else f"{r['p']:.2e}",
                    "yes" if r["rejected"] else ("not evaluable" if not r["evaluable"] else "no"),
                ]
            )
            + END
        )  # fmt: skip
    comp = "\n".join(lines + ["\\hline", "\\end{tabular}"]) + "\n"
    return {"tab_E24": tab, "tab_E24_comparisons": comp}


def macros(S, comps):
    fam = family_sentences(comps)
    return {
        "EXXIVR": str(R),
        "EXXIVBaseTests": str(fam["H24-Base"]["tests"]),
        "EXXIVBaseEvaluable": str(fam["H24-Base"]["evaluable"]),
        "EXXIVBaseRejected": str(fam["H24-Base"]["rejected"]),
        "EXXIVGRRTests": str(fam["H24-GRR"]["tests"]),
        "EXXIVGRREvaluable": str(fam["H24-GRR"]["evaluable"]),
        "EXXIVGRRRejected": str(fam["H24-GRR"]["rejected"]),
    }


def figure(S):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for arm in ARMS:
        g = S[S["arm"] == f"{arm}|ARW_cf"].sort_values("n")
        ax.plot(g["n"], g["rmse"], marker="o", label=LABEL[arm].replace("$", ""),
                linestyle="-" if arm in GRR_ARMS else "--")  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(list(N_VALUES))
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.set_xlabel("$n$")
    ax.set_ylabel("RMSE of the ARW estimator")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def tune_task(task):
    """``(cell, entropy)`` -> the JSON of :func:`tune` (importable for process parallelism)."""
    cell, entropy = task
    return json.dumps(tune(cell, entropy), sort_keys=True)


def tuning_table(tunings):
    """``tab_E24_tuning``: the frozen per-``n`` choices (pilot data, 5-fold CV)."""
    heads = ["$n$"] + [f"$\\lambda$ {LABEL[a]}" for a in GRR_ARMS] + [
        "Weight decay", "LogitAIPW $C$", "Outcome $\\lambda$", "Nystr\\\"om dim.",
    ]  # fmt: skip
    spec = "r" * len(heads)
    lines = ["\\begin{tabular}{" + spec + "}", "\\hline", " & ".join(heads) + END, "\\hline"]
    for tj in tunings:
        t = json.loads(tj)
        cells = [str(t["n"])] + [f"{t['lambda'][a]:g}" for a in GRR_ARMS] + [
            f"{t['weight_decay']:g}", f"{t['logit_C']:g}", f"{t['outcome_lambda']:g}",
            str(t["nystrom_dim"]),
        ]  # fmt: skip
        lines.append(" & ".join(cells) + END)
    return "\n".join(lines + ["\\hline", "\\end{tabular}"]) + "\n"
