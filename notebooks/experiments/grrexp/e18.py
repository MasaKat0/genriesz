"""E-18: selection by the held-out squared-loss Riesz criterion (``thm:selection_oracle``,
``cor:selection_inference``).

Design document: doc/2026-10-07_experiment_registration.md (parent repository), E-18 and
§1.2–§1.9. The design is not a binding registration (author instruction U28); the choices
it leaves open are recorded in ``docs/spec/02_data-and-experiments.md`` §2.1 ("E-18 の").
This module holds the two designs, the 65 candidates, the Stage 0 computation, the
replication function (importable for process parallelism) and the aggregation shared by
the pilot and Stage 1. The notebook ``12_E18_heldout_riesz_selection.ipynb`` runs the
stages; its last cell makes the table, the figure and the macros from ``summary.csv``.

Candidates (``M = 65``, both designs): 4 generators (SQ, UKL(C=1), BKL(C=1),
BP(omega=0.5, C=0)) x total Legendre degree 1-4 x ridge ``lam in {0, 1e-3, 1e-2, 1e-1}``
(``lam/2 ||beta||^2``), in that order, then the constant ``a0 = alpha_ref = +-2``. Each
treatment arm has its own coefficients ``beta_d`` (treatment interaction, §1.4) with the
dual coordinate ``u = u_ref + psi(z)^T beta_d`` and the restriction ``||beta_d||_1 <= R_g``
per arm. Because every feature satisfies ``|psi_j| <= 1``, this restriction gives
``|u - u_ref| <= R_g`` and hence ``|alpha| <= 50`` on the whole support. The ATE objective
is a sum of one objective per arm, so the two arms are fitted separately (exactly the
minimizer over the product of the two balls).

Constructions of the cross-fitted selector (``K = 5`` outer folds):
(a) the complement of fold k is split 50/50 into ``T_k`` and ``V_k`` (stream 2); the
candidates are fitted on ``T_k`` and selected on ``V_k``; (b) the candidates are fitted
on the complement and selected on the covariates of fold k.
"""

from __future__ import annotations

import collections
import itertools
import json

import numpy as np

from . import baselines, inference, metrics
from .bootstrap import FamilyBootstrap, percentile_interval
from .seeds import Seeds, fold_ids

EXP = 18
THETA0 = 1.0
SQRT3 = np.sqrt(3.0)
DESIGNS = ("18A", "18B")
N_VALUES = (1000, 2000, 4000, 8000)
CELLS = [[d, n] for d in DESIGNS for n in N_VALUES]  # registered order
R = {"18A": 500, "18B": 1000}
K = 5
CONSTRUCTIONS = ("a", "b")

GENERATORS = ("SQ", "UKL1", "BKL1", "BP0")
DEGREES = (1, 2, 3, 4)
LAMBDAS = (0.0, 1e-3, 1e-2, 1e-1)
CANDIDATES = [(g, d, lam) for g in GENERATORS for d in DEGREES for lam in LAMBDAS]
A0 = len(CANDIDATES)  # index of the constant candidate alpha_ref (last: ties go to fitted ones)
M = A0 + 1
CONTROL = CANDIDATES.index(("SQ", 1, 0.0))  # 18B descriptive control (fixed candidate)
#: l1 radius per arm; the values of the design document (rounded toward the guarantee)
RADIUS = {"SQ": 96.0, "UKL1": 3.8918202, "BKL1": 1.0586069, "BP0": 4.1426406}
A_ALPHA = 50.0
B_M = 2.0
#: design constants (outward-rounded bounds of the design document)
ETA_BOUND = {"18A": 2.9660255, "18B": 3.4516661}
#: C_m^2 = 2 / min(e_min, 1 - e_max) with e_min from ETA_BOUND, rounded up at 1e-6. The
#: design document lists 40.829202 and 65.105835, which round the exact values
#: 40.8292054... and 65.1058377... down; an upper bound must round up.
CM2 = {d: float(np.ceil(2.0 * (1.0 + np.exp(b)) * 1e6) / 1e6) for d, b in ETA_BOUND.items()}
E_RANGE = {"18A": (0.0489845, 0.9510155), "18B": (0.0307192, 0.9692808)}
MAX_ALPHA0 = {"18A": 20.414601, "18B": 32.552918}
E_ALPHA0_SQ = {"18A": 4.9630924, "18B": 5.3668883}
UKL_L1_18B = 3.4516661  # l1 norm of the Legendre coefficients of u0 = -eta per arm (18B)
DGP_TOL = 1e-6
N_EVAL = 100_000
T_FOLD = float(np.log(2 * M / 0.01))
EB_DELTA = 0.001 / (2 * M * K)  # one side of one interval (H18-Thm)
THM_P0 = 0.05 + 0.001
#: the solver works on the ball of radius R_g (1 - 1e-9): its ``"ok"`` solutions satisfy
#: ||beta||_1 <= radius (1 + 1e-12) (polishing allowance), so the stated R_g holds exactly.
SOLVER_RADIUS_FACTOR = 1.0 - 1e-9
SAMPLE_TOL = 1e-8  # §1.3 C (lam = 0: scaled by max(1, ||P_n m||_inf); ridge/ball: 1e-8)
INF_N = (2000, 4000, 8000)
EMP_N = (1000, 8000)

# Total-degree-ordered exponents of the tensor Legendre features (degree <= 4, 35 columns).
EXPONENTS = [m for d in range(5) for m in itertools.product(range(5), repeat=3) if sum(m) == d]
N_COLS = {d: sum(1 for m in EXPONENTS if sum(m) <= d) for d in range(5)}


def gr():
    import genriesz

    return genriesz


def c_sel(design):
    return 36.0 * (A_ALPHA**2 + CM2[design]) + 16.0 * A_ALPHA * (A_ALPHA + B_M)


def sign(A):
    return np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)


sign.vectorized = True  # one call on all rows (genriesz BregmanGenerator.branch_fn)


def generator(name):
    g = gr()
    return {
        "SQ": lambda: g.SquaredGenerator(C=0.0),
        "UKL1": lambda: g.UKLGenerator(C=1.0, branch_fn=sign),
        "BKL1": lambda: g.BKLGenerator(C=1.0, branch_fn=sign),
        "BP0": lambda: g.BPGenerator(C=0.0, omega=0.5, branch_fn=sign),
    }[name]()


def alpha_ref(X):
    return np.where(np.atleast_2d(X)[:, 0] == 1, 2.0, -2.0)


def _legendre(k, t):
    c = np.zeros(k + 1)
    c[k] = 1.0
    return np.polynomial.legendre.legval(t, c)


def legendre_features(Z, degree=4):
    """Tensor Legendre polynomials of ``t = z / sqrt3`` with total degree <= ``degree``
    (|value| <= 1 on the support), ordered by total degree."""
    T = np.atleast_2d(Z) / SQRT3
    P = [[_legendre(k, T[:, j]) for k in range(5)] for j in range(3)]
    cols = [P[0][a] * P[1][b] * P[2][c] for (a, b, c) in EXPONENTS[: N_COLS[degree]]]
    return np.column_stack(cols)


def eta(Z, design):
    Z = np.atleast_2d(Z)
    if design == "18A":
        return 0.9 * np.sin(np.pi * Z[:, 0] / SQRT3) - 0.5 * Z[:, 1] + 0.4 * Z[:, 0] * Z[:, 2]
    return 0.8 * Z[:, 0] - 0.5 * Z[:, 1] + 0.4 * Z[:, 0] * Z[:, 2]


def propensity(Z, design):
    return 1.0 / (1.0 + np.exp(-eta(Z, design)))


def gamma0(D, Z):
    """18B: gamma0 = z1 + 0.5 z2^2 + d (1 + 0.5 z3) + 0.3 z1 z2 (theta0 = 1)."""
    return Z[:, 0] + 0.5 * Z[:, 1] ** 2 + D * (1.0 + 0.5 * Z[:, 2]) + 0.3 * Z[:, 0] * Z[:, 1]


def draw(rng, n, design):
    """Z ~ Unif(-sqrt3, sqrt3)^3, D ~ Bern(e(Z)); 18B also Y = gamma0 + N(0, 1)."""
    Z = rng.uniform(-SQRT3, SQRT3, size=(n, 3))
    D = (rng.uniform(size=n) < propensity(Z, design)).astype(float)
    Y = gamma0(D, Z) + rng.standard_normal(n) if design == "18B" else None
    return D, Z, Y


def draw_eval(rng):
    return rng.uniform(-SQRT3, SQRT3, size=(N_EVAL, 3))


def quadratic(Z):
    """Arm-specific quadratic regressors of the 18B outcome regression (10 columns)."""
    z1, z2, z3 = Z[:, 0], Z[:, 1], Z[:, 2]
    return np.column_stack(
        [np.ones(len(Z)), z1, z2, z3, z1 * z1, z2 * z2, z3 * z3, z1 * z2, z1 * z3, z2 * z3]
    )


# ------------------------------------------------------------------ Stage 0


def gauss_legendre_mean(fn, points):
    """``E fn(Z)`` for ``Z ~ Unif(-sqrt3, sqrt3)^3`` by a tensor Gauss-Legendre rule."""
    x, w = np.polynomial.legendre.leggauss(points)
    x, w = x * SQRT3, w / 2.0
    G = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    W = np.einsum("i,j,k->ijk", w, w, w).reshape(-1)
    return float(np.sum(W * fn(G)))


def eta_extremes(design, grid=201):
    """Max |eta| over the support box: the coordinate grid plus the analytic interior
    critical points of each design (eta is multilinear in (z2, z3) given z1)."""
    z = np.linspace(-SQRT3, SQRT3, grid)
    if design == "18A":
        # sin attains +-1 at z1 = +-sqrt3/2; the other terms are linear in z2 and z3
        z1 = np.union1d(z, [SQRT3 / 2, -SQRT3 / 2])
    else:
        z1 = z
    G = np.stack(np.meshgrid(z1, [-SQRT3, SQRT3], [-SQRT3, SQRT3], indexing="ij"), -1)
    return float(np.max(np.abs(eta(G.reshape(-1, 3), design))))


def radius_check():
    """The l1 radii guarantee |alpha| <= 50 at u_ref +- R_g in both arms and stay in the
    domain; returns per generator the extreme |alpha| and the domain status."""
    g = gr()
    out = {}
    for name in GENERATORS:
        gen = generator(name)
        rows = {}
        for d in (1.0, 0.0):
            X = np.array([[d, 0.0, 0.0, 0.0]])
            u_ref = float(gen.grad(X, alpha_ref(X))[0])
            for side in (-1.0, 1.0):
                v = np.array([u_ref + side * RADIUS[name]])
                a, outside, nonfinite = g.glm.classify_predictions(gen, X, v)
                rows[f"d={int(d)},side={int(side)}"] = {
                    "u_ref": u_ref,
                    "alpha": float(a[0]),
                    "in_domain": bool(not outside[0] and not nonfinite[0]),
                }
        max_abs = max(abs(r["alpha"]) for r in rows.values())
        out[name] = {
            "rows": rows,
            "max_abs_alpha": max_abs,
            "ok": bool(max_abs <= A_ALPHA and all(r["in_domain"] for r in rows.values())),
        }
    return out


def ukl_18b_coefficients():
    """18B: u0 = -eta in both arms (UKL(C=1), u_ref = 0). Its Legendre coefficients on the
    degree-2 features and their l1 norm; the representation is checked on points."""
    t = np.zeros(N_COLS[2])
    idx = {m: i for i, m in enumerate(EXPONENTS[: N_COLS[2]])}
    # eta = 0.8 sqrt3 t1 - 0.5 sqrt3 t2 + 1.2 t1 t3 with P1(t) = t
    t[idx[(1, 0, 0)]] = -0.8 * SQRT3
    t[idx[(0, 1, 0)]] = 0.5 * SQRT3
    t[idx[(1, 0, 1)]] = -1.2
    rng = np.random.default_rng(0)
    Z = rng.uniform(-SQRT3, SQRT3, size=(1000, 3))
    resid = float(np.max(np.abs(legendre_features(Z, 2) @ t + eta(Z, "18B"))))
    return {"coefficients": t.tolist(), "l1": float(np.sum(np.abs(t))), "max_residual": resid}


def stage0():
    """Design checks of E-18 (Stage 0): the moments and bounds of both designs, the
    selection constants, the radius guarantee, and the 18B correct specification."""
    designs = {}
    for d in DESIGNS:
        e_lo = 1.0 / (1.0 + np.exp(ETA_BOUND[d]))
        ea2 = {}
        for pts in (48, 64):

            def f(Z, d=d):
                e = propensity(Z, d)
                return 1.0 / e + 1.0 / (1.0 - e)

            ea2[pts] = gauss_legendre_mean(f, pts)
        eta_max = eta_extremes(d)
        cm2 = 2.0 / min(e_lo, 1.0 - (1.0 - e_lo))
        rec = {
            "eta_max_grid": eta_max,
            "eta_bound": ETA_BOUND[d],
            "e_range_from_bound": [e_lo, 1.0 - e_lo],
            "max_abs_alpha0_from_bound": 1.0 + float(np.exp(ETA_BOUND[d])),
            "E_alpha0_sq": {str(k): v for k, v in ea2.items()},
            "Cm2_from_bound": cm2,
            "C_sel": c_sel(d),
            "t_fold": T_FOLD,
        }
        rec["checks"] = {
            "eta_bound_holds": eta_max <= ETA_BOUND[d],
            "e_range": abs(e_lo - E_RANGE[d][0]) <= DGP_TOL
            and abs((1.0 - e_lo) - E_RANGE[d][1]) <= DGP_TOL,
            "max_abs_alpha0": abs(rec["max_abs_alpha0_from_bound"] - MAX_ALPHA0[d]) <= 1e-5,
            "E_alpha0_sq": abs(ea2[64] - E_ALPHA0_SQ[d]) <= DGP_TOL
            and abs(ea2[48] - ea2[64]) <= 1e-10,
            "Cm2": cm2 <= CM2[d] <= cm2 + 1e-6,
        }
        rec["checks"] = {k: bool(v) for k, v in rec["checks"].items()}
        designs[d] = rec
    radius = radius_check()
    ukl = ukl_18b_coefficients()
    ukl["checks"] = {
        "l1": abs(ukl["l1"] - UKL_L1_18B) <= 1e-6 and ukl["l1"] < RADIUS["UKL1"],
        "represents_u0": ukl["max_residual"] <= 1e-12,
    }
    ukl["checks"] = {k: bool(v) for k, v in ukl["checks"].items()}
    ok = (
        all(all(r["checks"].values()) for r in designs.values())
        and all(r["ok"] for r in radius.values())
        and all(ukl["checks"].values())
    )
    return {
        "designs": designs,
        "radius": radius,
        "ukl_18b": ukl,
        "M": M,
        "predicted_coverage_18B": 0.95,
        "ok": bool(ok),
    }


# ------------------------------------------------------------------ candidates


class _Fit:
    """A fitted candidate: per arm the coefficients, the generator and the offset."""

    def __init__(self, gen_name, degree, betas):
        self.gen_name, self.degree, self.betas = gen_name, degree, betas
        self.gen = generator(gen_name)
        self.u_ref = {
            d: float(
                self.gen.grad(np.array([[d, 0.0, 0.0, 0.0]]), np.array([2.0 * (2 * d - 1)]))[0]
            )
            for d in (1.0, 0.0)
        }

    def u(self, d, Psi4):
        return self.u_ref[d] + Psi4[:, : N_COLS[self.degree]] @ self.betas[d]

    def alpha(self, d, Psi4):
        """Closed-form inverse link of §1.4 (tested against genriesz's classification).
        The l1 ball keeps ``|u - u_ref| <= R_g`` inside the domain on the support; a row
        outside it is flagged."""
        u = self.u(d, Psi4)
        s = 2.0 * d - 1.0
        su = s * u
        out = np.abs(u - self.u_ref[d]) > RADIUS[self.gen_name]
        name = self.gen_name
        if name == "SQ":
            a = u / 2.0
        elif name == "UKL1":
            a = s * (1.0 + np.exp(su))
        elif name == "BKL1":
            out = out | (su >= 0.0)
            e = np.exp(np.minimum(su, 0.0))
            a = s * (1.0 + e) / (1.0 - e)
        else:  # BP(omega = 0.5, C = 0): |alpha| = (1 + s u / k)^(1/omega), k = 3
            out = out | (su <= -3.0)
            a = s * (1.0 + su / 3.0) ** 2
        return a, out | ~np.isfinite(a) | (np.abs(a) > A_ALPHA)

    def conj(self, d, Psi4):
        X = np.zeros((len(Psi4), 4))
        X[:, 0] = d
        return np.asarray(self.gen.conjugate(X, self.u(d, Psi4))[0], dtype=float)


def _arm_basis(arm, degree):
    def f(X):
        X = np.atleast_2d(X)
        return (X[:, 0] == arm)[:, None] * legendre_features(X[:, 1:], degree)

    return gr().CallableBasis(f)


def fit_candidate(index, X_fit):
    """Fit candidate ``index`` (< A0) on ``X_fit``; returns ``(status, _Fit | None)``.
    Each arm is fitted on its own block (the product of the two l1 balls)."""
    g = gr()
    gen_name, degree, lam = CANDIDATES[index]
    D = X_fit[:, 0]
    if not (np.any(D == 1) and np.any(D == 0)):
        return "degenerate_functional", None
    betas = {}
    for arm in (1.0, 0.0):
        gen = generator(gen_name)
        mdl = g.GRRGLM(
            basis=_arm_basis(arm, degree), generator=gen, functional=g.ATEFunctional(0),
            penalty=None if lam == 0.0 else "l2", lam=lam,
            offset=g.offset_from_alpha(gen, alpha_ref),
            l1_radius=RADIUS[gen_name] * SOLVER_RADIUS_FACTOR,
        )  # fmt: skip
        fr = mdl.fit(X_fit, tol=SAMPLE_TOL)
        if fr.status != "ok":
            return str(fr.status), None
        beta = np.asarray(mdl.beta_, dtype=float)
        if not float(np.sum(np.abs(beta))) <= RADIUS[gen_name]:
            # the solver radius leaves room for its polishing allowance (1e-12, relative);
            # a coefficient outside the stated ball would break the |alpha| <= 50 envelope
            raise RuntimeError(f"candidate {index}: ||beta||_1 exceeds R_g after the fit")
        betas[arm] = beta
    return "ok", _Fit(gen_name, degree, betas)


# ------------------------------------------------------------------ one replication


def _risk_values(alpha1, alpha0_, e):
    """Pointwise ``e (alpha(1,z) - 1/e)^2 + (1 - e)(alpha(0,z) + 1/(1-e))^2``."""
    return e * (alpha1 - 1.0 / e) ** 2 + (1.0 - e) * (alpha0_ + 1.0 / (1.0 - e)) ** 2


def bernstein_interval(values, b, delta):
    """Two-sided empirical Bernstein interval (Maurer and Pontil 2009, Thm 4) for the mean
    of i.i.d. values in ``[0, b]``; each side holds with probability ``>= 1 - delta``."""
    x = np.asarray(values, dtype=float)
    N = x.size
    if np.any(x < 0) or np.any(x > b):
        raise ValueError("risk integrand outside [0, b]")
    mean = float(x.mean())
    var = float(x.var(ddof=1))
    lg = np.log(2.0 / delta)
    w = np.sqrt(2.0 * var * lg / N) + 7.0 * b * lg / (3.0 * (N - 1))
    return mean, max(0.0, mean - w), mean + w


def rhs(r, design, n_v):
    c = c_sel(design)
    return r + 2.0 * np.sqrt(c * r * T_FOLD / n_v) + 2.0 * c * T_FOLD / n_v


def _ols(Q_fit, y_fit):
    coef, *_ = np.linalg.lstsq(Q_fit, y_fit, rcond=None)
    return coef


def _fold_selection(design, X, Z, fit_idx, sel_idx, Psi4, Psi_eval, e_eval, b_range):
    """Fit all candidates on ``fit_idx`` and select on ``sel_idx``.

    Returns the per-candidate records and the fitted models (``None`` if failed)."""
    fits, statuses, warn = [], [], collections.Counter()
    for a in range(A0):
        (status, f), counts, converged = baselines.run_recording_warnings(
            lambda a=a: fit_candidate(a, X[fit_idx])
        )
        warn.update(counts)
        if status == "ok" and not converged:
            status, f = "convergence_warning", None
        statuses.append(status)
        fits.append(f)
    Pv = Psi4[sel_idx]
    Dv = X[sel_idx, 0]
    crit = np.full(M, np.nan)
    raw = np.full(A0, np.nan)
    risk = np.full(M, np.nan)
    lo = np.full(M, np.nan)
    hi = np.full(M, np.nan)
    for a in range(M):
        if a == A0:
            a1v = np.full(len(Pv), 2.0)
            a0v = np.full(len(Pv), -2.0)
            ae1 = np.full(len(Psi_eval), 2.0)
            ae0 = np.full(len(Psi_eval), -2.0)
        else:
            f = fits[a]
            if f is None:
                continue
            a1v, o1 = f.alpha(1.0, Pv)
            a0v, o0 = f.alpha(0.0, Pv)
            ae1, oe1 = f.alpha(1.0, Psi_eval)
            ae0, oe0 = f.alpha(0.0, Psi_eval)
            if np.any(o1) or np.any(o0) or np.any(oe1) or np.any(oe0):
                # the ball keeps |u - u_ref| <= R_g inside the domain on the support
                raise RuntimeError(f"candidate {a} left the domain at a selection point")
            # raw Bregman objective of the candidate's own generator on the selection rows
            u1, u0 = f.u(1.0, Pv), f.u(0.0, Pv)
            gstar = np.where(Dv == 1, f.conj(1.0, Pv), f.conj(0.0, Pv))
            raw[a] = float(np.mean(gstar - (u1 - u0)))
        alpha_obs = np.where(Dv == 1, a1v, a0v)
        crit[a] = float(np.mean(alpha_obs**2 - 2.0 * (a1v - a0v)))
        vals = _risk_values(ae1, ae0, e_eval)
        risk[a], lo[a], hi[a] = bernstein_interval(vals, b_range, EB_DELTA)
    avail = np.isfinite(crit)
    sel = int(np.flatnonzero(avail)[np.argmin(crit[avail])])  # first minimiser: registered order
    rawsel = A0 if not np.any(np.isfinite(raw)) else int(np.nanargmin(raw))
    rstar = float(np.nanmin(risk))
    rstar_u = float(np.nanmin(hi))
    n_v = len(sel_idx)
    return {
        "sel": sel,
        "risk_sel": float(risk[sel]),
        "risk_star": rstar,
        "risk_raw_sel": float(risk[rawsel]),
        "raw_sel": rawsel,
        "risk_control": float(risk[CONTROL]),
        "violation": bool(lo[sel] > rhs(rstar_u, design, n_v)),
        "n_v": n_v,
        "n_failed": int(sum(s != "ok" for s in statuses)),
        "control_status": statuses[CONTROL],
        "statuses": collections.Counter(statuses),
        "warnings": warn,
    }, fits


def _arm_ols(Q, X, Y, comp):
    """Arm-specific quadratic OLS on the rows ``comp``; ``(status, coef1, coef0)``. An arm
    without enough rows for a full-rank fit is a failure, never a zero regression."""
    coefs = {}
    for arm in (1.0, 0.0):
        rows = comp[X[comp, 0] == arm]
        if len(rows) < Q.shape[1]:
            return "degenerate_functional", None, None
        coef, _, rank, _ = np.linalg.lstsq(Q[rows], Y[rows], rcond=None)
        if rank < Q.shape[1]:
            return "singular", None, None
        coefs[arm] = coef
    return "ok", coefs[1.0], coefs[0.0]


def _arw_fold(fitted, X, Y, Psi4, Q, comp, te):
    """ARW score on fold ``te`` with the representer ``fitted`` (a _Fit, or ``None`` for the
    constant a0) and the arm-specific quadratic OLS fitted on ``comp``.

    Returns ``(status, psi, alpha)``; the observed and counterfactual predictions of the
    representer on the fold are checked (§1.3 B ``domain_prediction``)."""
    status, c1, c0 = _arm_ols(Q, X, Y, comp)
    if status != "ok":
        return status, None, None
    Qt, Dt = Q[te], X[te, 0]
    g1, g0 = Qt @ c1, Qt @ c0
    if fitted is None:
        a1 = np.full(len(Qt), 2.0)
        a0 = np.full(len(Qt), -2.0)
    else:
        a1, o1 = fitted.alpha(1.0, Psi4[te])
        a0, o0 = fitted.alpha(0.0, Psi4[te])
        if np.any(o1) or np.any(o0):
            return "domain_prediction", None, None
    a = np.where(Dt == 1, a1, a0)
    gx = np.where(Dt == 1, g1, g0)
    psi = g1 - g0 + a * (Y[te] - gx)
    if not np.all(np.isfinite(psi)):
        return "nonfinite", None, None
    return "ok", psi, a


def replicate(task):
    """One replication of one cell. ``task = (cell, rep, entropy)``."""
    cell_index, rep, entropy = task
    design, n = CELLS[cell_index]
    seeds = Seeds(EXP, entropy=entropy)
    D, Z, Y = draw(seeds.data(cell_index, rep), n, design)
    X = np.column_stack([D, Z])
    folds = fold_ids(n, K, seeds.folds(cell_index, rep))
    split_rng = seeds.inner_cv(cell_index, rep)
    Z_eval = draw_eval(seeds.evaluation(rep))
    e_eval = propensity(Z_eval, design)
    b_range = (A_ALPHA + MAX_ALPHA0[design]) ** 2
    Psi4 = legendre_features(Z, 4)
    Psi_eval = legendre_features(Z_eval, 4)
    Q = quadratic(Z) if design == "18B" else None
    out = {}
    for con in CONSTRUCTIONS:
        recs, statuses, warns = [], collections.Counter(), collections.Counter()
        psi = {"SEL": np.empty(n), "CTRL": np.empty(n)}
        est_status = {"SEL": "ok", "CTRL": "ok"}
        alpha_sel = np.empty(n)
        for k in range(K):
            te = np.flatnonzero(folds == k)
            comp = np.flatnonzero(folds != k)
            if con == "a":
                perm = split_rng.permutation(comp)
                half = len(perm) // 2
                fit_idx, sel_idx = np.sort(perm[:half]), np.sort(perm[half:])
            else:
                fit_idx, sel_idx = comp, te
            rec, fits = _fold_selection(
                design, X, Z, fit_idx, sel_idx, Psi4, Psi_eval, e_eval, b_range
            )
            statuses.update(rec.pop("statuses"))
            warns.update(rec.pop("warnings"))
            recs.append(rec)
            if design == "18B":
                chosen = None if rec["sel"] == A0 else fits[rec["sel"]]
                for name, fitted, failed in (
                    ("SEL", chosen, None),
                    (
                        "CTRL",
                        fits[CONTROL],
                        None if fits[CONTROL] is not None else rec["control_status"],
                    ),
                ):
                    if est_status[name] != "ok":
                        continue  # the first failed fold fails the replication (§1.3 D)
                    if failed is not None:
                        est_status[name] = failed
                        continue
                    st, p_k, a_k = _arw_fold(fitted, X, Y, Psi4, Q, comp, te)
                    est_status[name] = st
                    if st == "ok":
                        psi[name][te] = p_k
                        if name == "SEL":
                            alpha_sel[te] = a_k
        res = {
            "folds": recs,
            "statuses": dict(statuses),
            "warnings": dict(warns),
            "violation_any": bool(any(r["violation"] for r in recs)),
        }
        if design == "18B":
            for name in ("SEL", "CTRL"):
                if est_status[name] == "ok":
                    th = float(np.mean(psi[name]))
                    res[name] = {
                        "estimate": th,
                        "se": float(np.std(psi[name] - th) / np.sqrt(n)),  # §1.4: P_n psi^2
                        "status": "ok",
                    }
                else:
                    res[name] = {"estimate": float("nan"), "se": float("nan"),
                                 "status": est_status[name]}  # fmt: skip
            res["SEL"]["max_abs_alpha"] = (
                float(np.max(np.abs(alpha_sel))) if est_status["SEL"] == "ok" else float("nan")
            )
        out[con] = res
    return out


def tasks_for(entropy, reps=None, cells=None):
    """Tasks in registered order; ``reps`` overrides the per-design R (the pilot)."""
    cells = range(len(CELLS)) if cells is None else cells
    return [
        (ci, r, entropy)
        for ci in cells
        for r in range(reps if reps is not None else R[CELLS[ci][0]])
    ]


# ------------------------------------------------------------------ aggregation

FOLD_FIELDS = ("sel", "risk_sel", "risk_star", "risk_raw_sel", "raw_sel", "risk_control",
               "violation", "n_v", "n_failed")  # fmt: skip


def raw_frame(tasks, results):
    """One row per replication x construction; per-fold values as JSON lists (§1.7)."""
    import pandas as pd

    rows = []
    for (ci, rep, _), res in zip(tasks, results, strict=True):
        design, n = CELLS[ci]
        for con in CONSTRUCTIONS:
            r = res[con]
            row = {"cell": ci, "design": design, "n": n, "rep": rep, "construction": con,
                   "violation_any": bool(r["violation_any"]),
                   "statuses": json.dumps(r["statuses"], sort_keys=True),
                   "warnings": json.dumps(r["warnings"], sort_keys=True),
                   "n_warnings": int(sum(r["warnings"].values()))}  # fmt: skip
            for f in FOLD_FIELDS:
                row[f] = json.dumps([fr[f] for fr in r["folds"]])
            for est in ("SEL", "CTRL"):
                e = r.get(est, {})
                row[f"{est}_estimate"] = float(e.get("estimate", np.nan))
                row[f"{est}_se"] = float(e.get("se", np.nan))
                row[f"{est}_status"] = str(e.get("status", ""))
            row["SEL_max_abs_alpha"] = float(r.get("SEL", {}).get("max_abs_alpha", np.nan))
            rows.append(row)
    return pd.DataFrame(rows)


def _folds(raw, field):
    return np.array([json.loads(v) for v in raw[field]], dtype=float)  # (reps, K)


def total_warnings(raw):
    return int(raw["n_warnings"].sum())


def total_failures(raw):
    """Failed candidate fits plus failed estimator records."""
    fits = sum(sum(v for k, v in json.loads(s).items() if k != "ok") for s in raw["statuses"])
    est = int(((raw["design"] == "18B") & (raw["SEL_status"] != "ok")).sum()) + int(
        ((raw["design"] == "18B") & (raw["CTRL_status"] != "ok")).sum()
    )
    return int(fits + est)


def candidate_label(a):
    if a == A0:
        return "a0"
    g, d, lam = CANDIDATES[a]
    return f"{g}-d{d}-l{lam:g}"


def summarise(raw):
    """One row per cell x construction."""
    import pandas as pd

    rows = []
    for (ci, con), g in raw.groupby(["cell", "construction"], sort=True):
        design, n = CELLS[ci]
        Rn = len(g)
        rs, rstar = _folds(g, "risk_sel"), _folds(g, "risk_star")
        nv = _folds(g, "n_v")
        sel = _folds(g, "sel").astype(int)
        gens = collections.Counter("a0" if a == A0 else CANDIDATES[a][0] for a in sel.reshape(-1))
        st = collections.Counter()
        for s in g["statuses"]:
            st.update(json.loads(s))
        row = {
            "cell": ci, "design": design, "n": n, "construction": con, "R": Rn,
            "violations": int(g["violation_any"].sum()),
            "violation_rate": float(g["violation_any"].mean()),
            "q90_excess": float(np.quantile(np.sqrt(nv) * (rs - rstar), 0.9)),
            "median_ratio": float(np.median(rs / rstar)),
            "median_risk_sel": float(np.median(rs)),
            "median_risk_star": float(np.median(rstar)),
            "median_ratio_raw": float(np.median(_folds(g, "risk_raw_sel") / rstar)),
            # the control candidate can fail in a fold: median over the folds where it was fitted
            "median_ratio_control": float(np.nanmedian(_folds(g, "risk_control") / rstar)),
            "mean_failed_candidates": float(_folds(g, "n_failed").mean()),
            "selected_generators": json.dumps(dict(sorted(gens.items()))),
            "candidate_statuses": json.dumps(dict(sorted(st.items()))),
        }  # fmt: skip
        if design == "18B":
            for est in ("SEL", "CTRL"):
                ok = (g[f"{est}_status"] == "ok").to_numpy()
                row[f"{est}_R_s"] = int(ok.sum())
                row[f"{est}_failure_rate"] = float(1.0 - ok.mean())
                row[f"{est}_status_counts"] = json.dumps(
                    dict(sorted(collections.Counter(g[f"{est}_status"]).items()))
                )
                if ok.all():
                    row[f"{est}_failure_rate_cp_upper"] = metrics.clopper_pearson_upper(0, len(ok))
                if not ok.any():
                    row[f"{est}_covered"] = 0
                    row[f"{est}_coverage"] = 0.0
                    continue
                est_v = g[f"{est}_estimate"].to_numpy()
                se_v = g[f"{est}_se"].to_numpy()
                x = est_v[ok]
                row[f"{est}_bias"] = float(x.mean() - THETA0)
                row[f"{est}_bias_mcse"] = (
                    float(x.std(ddof=1) / np.sqrt(x.size)) if x.size >= 2 else float("nan")
                )
                row[f"{est}_rmse"] = float(np.sqrt(np.mean((x - THETA0) ** 2)))
                # the SD MCSE (fourth moment) is not needed by any E-18 test; with the few
                # pilot replications its radicand can be negative, so it is not computed
                row[f"{est}_root_n_sd"] = (
                    float(np.sqrt(n) * x.std(ddof=1)) if x.size >= 2 else float("nan")
                )
                row[f"{est}_se_ratio"] = (
                    metrics.se_ratio(est_v, se_v, ok) if x.size >= 2 else float("nan")
                )
                row.update(
                    {
                        f"{est}_{k}": float(v)
                        for k, v in metrics.coverage(est_v, se_v, ok, THETA0).items()
                    }
                )
                row.update({f"{est}_{k}": float(v)
                            for k, v in metrics.ci_length(se_v, ok).items()})  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


def family_thm(summ):
    """H18-Thm: 18A, (a)(b) x 4 n; ``H0: p <= 0.051`` by the one-sided exact binomial test."""
    tests = {}
    for _, r in summ[summ["design"] == "18A"].iterrows():
        tests[f"{r['construction']}|n={r['n']}"] = inference.failure_rate_pvalue(
            int(r["violations"]), int(r["R"]), THM_P0
        )
    return (inference.judge_family("H18-Thm", tests) if tests else None), tests


def family_inf(summ):
    """H18-Inf: 18B, (a)(b) x n in {2000, 4000, 8000}; ``|c - 0.95| <= 0.02``."""
    tests = {}
    for _, r in summ[(summ["design"] == "18B") & summ["n"].isin(INF_N)].iterrows():
        tests[f"{r['construction']}|n={r['n']}"] = inference.coverage_pvalue(
            int(r["SEL_covered"]), int(r["R"]), 0.95
        )
    return (inference.judge_family("H18-Inf", tests) if tests else None), tests


def emp_intervals(raw, seeds):
    """H18-Emp (family 5, one generator): per construction, the statistics (b) the 90%
    point of sqrt(n_V)(rho_sel - rho*) and (c) the median of rho_sel/rho* on 18A, then (e)
    the median of rho_sel on 18B; within a statistic the n in increasing order. The
    replications are resampled with their five folds."""
    fb = FamilyBootstrap(seeds, 5)
    out = []
    specs = [("18A", "b_q90_excess"), ("18A", "c_median_ratio"), ("18B", "e_median_risk")]
    for con in CONSTRUCTIONS:
        for design, stat in specs:
            for n in N_VALUES:
                g = raw[(raw["design"] == design) & (raw["n"] == n) & (raw["construction"] == con)]
                if len(g) < 2:
                    continue  # a cell absent from this aggregation (pilot) draws nothing
                g = g.sort_values("rep")
                rs, rstar, nv = _folds(g, "risk_sel"), _folds(g, "risk_star"), _folds(g, "n_v")
                if stat == "b_q90_excess":
                    v = np.sqrt(nv) * (rs - rstar)

                    def fn(idx, v=v):
                        return np.quantile(v[idx].reshape(idx.shape[0], -1), 0.9, axis=1)

                    point = float(np.quantile(v, 0.9))
                elif stat == "c_median_ratio":
                    v = rs / rstar

                    def fn(idx, v=v):
                        return np.median(v[idx].reshape(idx.shape[0], -1), axis=1)

                    point = float(np.median(v))
                else:
                    v = rs

                    def fn(idx, v=v):
                        return np.median(v[idx].reshape(idx.shape[0], -1), axis=1)

                    point = float(np.median(v))
                boot = fb.replicate(fn, len(g))
                lo, hi = percentile_interval(boot, 5)
                out.append({"construction": con, "design": design, "statistic": stat, "n": n,
                            "point": point, "lower": lo, "upper": hi})  # fmt: skip
    return out


def emp_verdicts(intervals):
    """(b): upper(8000) <= 2 lower(1000); (c), (e): upper(8000) < lower(1000)."""
    res = {}
    for con in CONSTRUCTIONS:
        for stat in ("b_q90_excess", "c_median_ratio", "e_median_risk"):
            rows = {
                r["n"]: r for r in intervals if r["construction"] == con and r["statistic"] == stat
            }
            if not all(n in rows for n in EMP_N):
                continue
            lo, hi = rows[EMP_N[0]]["lower"], rows[EMP_N[1]]["upper"]
            ok = hi <= 2.0 * lo if stat == "b_q90_excess" else hi < lo
            res[f"{con}|{stat}"] = bool(ok)
    return res


# ------------------------------------------------------------------ outputs

END = " \\\\"


def _fmt(x, d=3):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def tables(S, intervals):
    """tab_E18_inference (18B estimators and the selection summary of both designs) and the
    macros, from ``summary.csv`` and the H18-Emp intervals only (§4)."""
    head = ("Design & Constr. & $n$ & Viol. & med. $\\varrho_{\\widehat a}/\\varrho^*$ & "
            "med. raw$/\\varrho^*$ & Bias (SEL) & $\\sqrt n$SD & SE ratio & Coverage & "
            "Coverage (SQ-1) & Fail \\% (SEL) & Fail \\% (SQ-1) & Fail.\\ cand." + END)  # fmt: skip
    lines = ["\\begin{tabular}{llrrrrrrrrrrrr}", "\\hline", head, "\\hline", "\\endfirsthead",
             "\\hline", head, "\\hline", "\\endhead"]  # fmt: skip
    for _, r in S.sort_values(["design", "construction", "n"]).iterrows():
        b = r["design"] == "18B"
        lines.append(" & ".join([
            r["design"], f"({r['construction']})", str(int(r["n"])),
            f"{int(r['violations'])}/{int(r['R'])}",
            _fmt(r["median_ratio"]), _fmt(r["median_ratio_raw"]),
            _fmt(r.get("SEL_bias")) if b else "--",
            _fmt(r.get("SEL_root_n_sd"), 2) if b else "--",
            _fmt(r.get("SEL_se_ratio"), 2) if b else "--",
            _fmt(r.get("SEL_coverage")) if b else "--",
            _fmt(r.get("CTRL_coverage")) if b else "--",
            f"{100 * r['SEL_failure_rate']:.1f}" if b else "--",
            f"{100 * r['CTRL_failure_rate']:.1f}" if b else "--",
            _fmt(r["mean_failed_candidates"], 2),
        ]) + END)  # fmt: skip
    lines += ["\\hline", "\\end{tabular}"]
    return {"tab_E18_inference": "\n".join(lines) + "\n"}


def figure(S, intervals):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for con, mk in (("a", "o"), ("b", "s")):
        for j, (_design, stat, ylab) in enumerate(
            [("18A", "c_median_ratio", r"median $\varrho_{\hat a}/\varrho^*$ (18A)"),
             ("18B", "e_median_risk", r"median $\varrho_{\hat a}$ (18B)")]
        ):  # fmt: skip
            rows = sorted(
                (r for r in intervals if r["construction"] == con and r["statistic"] == stat),
                key=lambda r: r["n"],
            )
            x = [r["n"] for r in rows]
            y = [r["point"] for r in rows]
            err = [[r["point"] - r["lower"] for r in rows], [r["upper"] - r["point"] for r in rows]]
            ax[j].errorbar(x, y, yerr=err, marker=mk, capsize=3, label=f"construction ({con})")
            ax[j].set_xscale("log")
            ax[j].set_xlabel("$n$")
            ax[j].set_ylabel(ylab)
    ax[0].legend()
    fig.tight_layout()
    return fig


def aggregate(raw, seeds):
    """The workflow after the replications (pilot and Stage 1): the summary, H18-Thm,
    H18-Inf and the H18-Emp intervals (family 5), stored as JSON columns of the summary."""
    summ = summarise(raw)
    thm, thm_tests = family_thm(summ)
    inf, inf_tests = family_inf(summ)
    intervals = emp_intervals(raw, seeds)
    emp = emp_verdicts(intervals)
    summ["thm_verdict"] = json.dumps(thm.sentence() if thm else None)
    summ["thm_rejected"] = json.dumps(list(thm.rejected) if thm else [])
    summ["thm_tests"] = json.dumps(thm_tests)
    summ["inf_verdict"] = json.dumps(inf.sentence() if inf else None)
    summ["inf_rejected"] = json.dumps(list(inf.rejected) if inf else [])
    summ["inf_tests"] = json.dumps(inf_tests)
    summ["emp_intervals"] = json.dumps(intervals)
    summ["emp_verdicts"] = json.dumps(emp)
    return summ


def outputs_from_summary(S):
    """Table, figure and macros from ``summary.csv`` only (§4)."""
    intervals = json.loads(S["emp_intervals"].iloc[0])
    emp = json.loads(S["emp_verdicts"].iloc[0])
    thm_neg = bool(json.loads(S["thm_rejected"].iloc[0]))
    inf_neg = bool(json.loads(S["inf_rejected"].iloc[0]))
    return tables(S, intervals), figure(S, intervals), macros(S, thm_neg, inf_neg, emp)


def macros(S, thm_negative, inf_negative, emp):
    def word(neg):
        return "negative" if neg else "no departure detected"

    m = {
        "EXVIIIThmVerdict": word(thm_negative),
        "EXVIIIInfVerdict": word(inf_negative),
        "EXVIIIM": str(M),
        "EXVIIIRA": str(R["18A"]),
        "EXVIIIRB": str(R["18B"]),
        "EXVIIIThmViolations": str(int(S[S["design"] == "18A"]["violations"].sum())),
    }
    for key, ok in emp.items():
        con, stat = key.split("|")
        name = (
            "EXVIIIEmp"
            + {"a": "A", "b": "B"}[con]
            + {"b_q90_excess": "Excess", "c_median_ratio": "Ratio", "e_median_risk": "Risk"}[stat]
        )
        m[name] = "satisfied" if ok else "not satisfied"
    return m
