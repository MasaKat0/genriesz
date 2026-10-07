"""E-13: compatibility and tangent-direction balance (H13, ``prop:arbitrary_pair_foc``).

Registration: doc/2026-10-07_experiment_registration.md, E-13, §1.3 A-4, §1.4–§1.9
(including "E-13 の incompatible な pair の規則" (i')–(iv')). This module holds the
DGP, the six loss-link pairs, the Stage 0 population computation, the replication
function (importable for process parallelism) and the aggregation shared by the
pilot and the confirmatory stage. The notebook
``11_E13_compatibility_tangent_balance.ipynb`` runs the stages and writes the
outputs; its last cell makes the table and the figure from ``summary.csv`` only (§4).

Pairs (all ``lambda = 0``; ``alpha_ref = +-2``):

- compatible (``GRRGLM``, dual offset ``u_ref = g'(alpha_ref)``): C-SQ, C-UKL(C=1),
  C-BKL(C=1);
- incompatible (``GRRGeneralLink``, index offset ``v_ref = link^{-1}(alpha_ref)``):
  I-SQexp (SQ + ``s(1 + e^{sv})``), I-BKLexp (BKL(C=1) + the same link), I-UKLlin
  (UKL(C=1) + ``s(1 + v)``, ``v > 0``).

Estimators: RW_full (the stationary point on the full sample, ``P_n alpha Y``) and
ARW_cf (§1.4, ``K = 5``, the representer and the OLS regression both cross-fitted).
"""

from __future__ import annotations

import collections
import itertools
import json

import numpy as np

from . import baselines, metrics, population
from .seeds import Seeds, fold_ids

EXP = 13
THETA0 = 1.0
ETA_SCALE = 0.75
N_VALUES = (1000, 4000)
CELLS = [[n] for n in N_VALUES]  # registered order
R = 1000
K = 5
SQRT3 = np.sqrt(3.0)
COMPATIBLE = ("C-SQ", "C-UKL1", "C-BKL1")
INCOMPATIBLE = ("I-SQexp", "I-BKLexp", "I-UKLlin")
PAIRS = COMPATIBLE + INCOMPATIBLE
ESTIMATORS = ("RW_full", "ARW_cf")
ARM_LABELS = [f"{p}|{e}" for p in PAIRS for e in ESTIMATORS]
P = 8  # treatment interaction of (1, z1, z2, z3)
#: gamma0(d, z) = theta0 d + z1 + 0.5 z2 on the basis (D(1, z), (1 - D)(1, z)).
RHO = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0])
#: index offsets of §1.3 A-4: v_ref = link^{-1}(alpha_ref), |alpha_ref| = 2
V_REF = {"I-SQexp": 0.0, "I-BKLexp": 0.0, "I-UKLlin": 1.0}
#: registered DGP13 values (outward-rounded range of e, E alpha0^2 checked by the reviewer)
E_RANGE = (0.1385765, 0.9452441)
E_ALPHA0_SQ = 5.0423229
DGP_TOL = 1e-6

#: §1.3 C. Compatible, lambda = 0 and no constraint: max_j |grad_j| <= 1e-8
#: max(1, ||P_n m(W, phi)||_inf) (genriesz scales ``tol`` by that factor);
#: general link: max_j |Delta_hat(alpha_hat, psi_j)| <= 1e-8 (absolute).
SAMPLE_TOL = 1e-8
GENERAL_TOL = 1e-8
BALANCE_TOL = 1e-8  # H13 (a) I_psi and (b) I_phi
CERT_GRADIENT = 1e-12  # §1.9 (i), (i')
CERT_BETA_DIFF = 1e-10  # §1.9 (i), (i')
CERT_MARGIN = 1e-4  # §1.9 (iii), (iii')
CERT_EIG_RATIO = 1e-6  # (ii')
MULTISTART = 20  # (iv')
MULTISTART_TOL = 1e-8  # (iv')
DELTA_REL_TOL = 0.05  # H13 (c)
BIAS_REL_TOL = 0.1  # H13 (d), incompatible pairs
COVERAGE_TOL = 0.03  # H13 (e)


def gr():
    import genriesz

    return genriesz


def sign(A):
    return np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)


sign.vectorized = True  # one call on all rows (genriesz BregmanGenerator.branch_fn)


def _s(X):
    return 2.0 * np.atleast_2d(X)[:, 0] - 1.0


def exp_link(X, v):
    s = _s(X)
    return s * (1.0 + np.exp(s * v))


def exp_dlink(X, v):
    return np.exp(_s(X) * v)


def exp_d2link(X, v):
    s = _s(X)
    return s * np.exp(s * v)


def lin_link(X, v):
    return _s(X) * (1.0 + v)


def lin_dlink(X, v):
    return _s(X) * np.ones_like(v)


def lin_d2link(X, v):
    return np.zeros_like(v)


LINKS = {
    "I-SQexp": (exp_link, exp_dlink, exp_d2link),
    "I-BKLexp": (exp_link, exp_dlink, exp_d2link),
    "I-UKLlin": (lin_link, lin_dlink, lin_d2link),
}


def generator(pair):
    g = gr()
    return {
        "C-SQ": lambda: g.SquaredGenerator(C=0.0),
        "C-UKL1": lambda: g.UKLGenerator(C=1.0, branch_fn=sign),
        "C-BKL1": lambda: g.BKLGenerator(C=1.0, branch_fn=sign),
        "I-SQexp": lambda: g.SquaredGenerator(C=0.0),
        "I-BKLexp": lambda: g.BKLGenerator(C=1.0, branch_fn=sign),
        "I-UKLlin": lambda: g.UKLGenerator(C=1.0, branch_fn=sign),
    }[pair]()


def features(Z):
    """phi(z) = (1, z1, z2, z3)."""
    Z = np.atleast_2d(Z)
    return np.column_stack([np.ones(len(Z)), Z[:, 0], Z[:, 1], Z[:, 2]])


def basis():
    g = gr()
    return g.TreatmentInteractionBasis(base_basis=g.CallableBasis(features))


def alpha_ref(X):
    return np.where(np.atleast_2d(X)[:, 0] == 1, 2.0, -2.0)


def eta(Z):
    q = Z[:, 1] ** 2 - 1.0
    return ETA_SCALE * (Z[:, 0] - 0.5 * Z[:, 1] + 0.6 * q)


def gamma0(D, Z):
    return THETA0 * D + Z[:, 0] + 0.5 * Z[:, 1]


def draw(rng, n):
    """Z ~ Unif(-sqrt3, sqrt3)^3, D ~ Bern(expit eta), Y = gamma0 + N(0, 1) (as E-12)."""
    Z = rng.uniform(-SQRT3, SQRT3, size=(n, 3))
    e = 1.0 / (1.0 + np.exp(-eta(Z)))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = gamma0(D, Z) + rng.standard_normal(n)
    return D, Z, Y


def model(pair):
    """An unfitted representer model of the pair (lambda = 0)."""
    g = gr()
    gen = generator(pair)
    if pair in COMPATIBLE:
        return g.GRRGLM(
            basis=basis(), generator=gen, functional=g.ATEFunctional(0), penalty=None, lam=0.0,
            offset=g.offset_from_alpha(gen, alpha_ref),
        )  # fmt: skip
    link, dlink, d2link = LINKS[pair]
    return g.GRRGeneralLink(
        gen, link, dlink, d2link, basis=basis(), functional=g.ATEFunctional(0),
        index_offset=V_REF[pair],
    )  # fmt: skip


def _finite_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------- Stage 0


def quadrature(points):
    """Population rows (1, z) and (0, z) on a tensor Gauss-Legendre grid, with P(D, Z)."""
    x, w = np.polynomial.legendre.leggauss(points)
    z = SQRT3 * x
    wz = w / 2.0  # uniform density on [-sqrt3, sqrt3]
    Zq = np.array(list(itertools.product(z, z, z)))
    Wq = np.array([a * b * c for a, b, c in itertools.product(wz, wz, wz)])
    e = 1.0 / (1.0 + np.exp(-eta(Zq)))
    X = np.vstack(
        [np.column_stack([np.ones(len(Zq)), Zq]), np.column_stack([np.zeros(len(Zq)), Zq])]
    )
    wts = np.concatenate([Wq * e, Wq * (1.0 - e)])
    return X, wts, Zq, Wq, e


def dgp_check(points=64):
    """The registered DGP13 values: the range of e over the support box contains the
    exact range (outward rounding) and E alpha0^2 agrees at 1e-6."""
    _, _, _, Wq, e = quadrature(points)
    lo = 1.0 / (1.0 + np.exp(-ETA_SCALE * (-SQRT3 - 0.6 - 0.25 / 2.4)))
    hi = 1.0 / (1.0 + np.exp(-ETA_SCALE * (SQRT3 + 0.5 * SQRT3 + 0.6 * 3.0 - 0.6)))
    e_alpha0 = float(Wq @ (1.0 / e + 1.0 / (1.0 - e)))
    return {
        "e_range_exact": [lo, hi],
        "e_range_registered": list(E_RANGE),
        "E_alpha0_sq": e_alpha0,
        "E_alpha0_sq_registered": E_ALPHA0_SQ,
        "ok": bool(
            E_RANGE[0] <= lo and hi <= E_RANGE[1] and abs(e_alpha0 - E_ALPHA0_SQ) <= DGP_TOL
        ),
    }


def _index_range(beta, v_ref):
    """Exact range of ``v_ref + f(z)' beta_arm`` over the support box, per arm (d = 1, 0)."""
    out = {}
    for d, c in ((1.0, beta[:4]), (0.0, beta[4:])):
        half = SQRT3 * float(np.sum(np.abs(c[1:])))
        out[d] = (v_ref(d) + c[0] - half, v_ref(d) + c[0] + half)
    return out


def support_margins(pair, beta):
    """§1.9 (iii) and (iii'): the margins over the whole support box (observed and
    counterfactual rows: the box holds every z for both d).

    The index is affine in z, so its range per arm is exact (interval arithmetic).
    Compatible: the margin of the dual coordinate to the finite end of the range
    of g' (``generator.dual_margin``). Incompatible: the margin of the index to the
    link domain (``v > 0`` for I-UKLlin) and of alpha to the domain of g
    (``generator.boundary_margin``, monotone in ``s v`` for both links), each
    ``+inf`` when there is no finite end.
    """
    gen = generator(pair)
    if pair in COMPATIBLE:

        def u_ref(d):
            Xd = np.array([[d, 0.0, 0.0, 0.0]])
            return float(gen.grad(Xd, alpha_ref(Xd))[0])

        margins = []
        for d, (lo, hi) in _index_range(beta, u_ref).items():
            Xd = np.array([[d, 0.0, 0.0, 0.0], [d, 0.0, 0.0, 0.0]])
            margins.append(float(np.min(gen.dual_margin(Xd, np.array([lo, hi])))))
        return {"dual": min(margins)}
    link = LINKS[pair][0]
    m_index, m_alpha = [], []
    for d, (lo, hi) in _index_range(beta, lambda d: V_REF[pair]).items():
        m_index.append(lo if pair == "I-UKLlin" else np.inf)
        Xd = np.array([[d, 0.0, 0.0, 0.0], [d, 0.0, 0.0, 0.0]])
        with np.errstate(over="ignore"):
            a = link(Xd, np.array([lo, hi]))
        m_alpha.append(float(np.min(gen.boundary_margin(Xd, a))))
    return {"index": float(min(m_index)), "alpha": float(min(m_alpha))}


def _margin_kind(value):
    return "unbounded" if np.isinf(value) and value > 0 else "finite"


def population_moments(X, w, alpha, dalpha_dbeta, score, H):
    """§1.9 quantities from the stacked Z-estimator of RW and the ARW expansion.

    ``score`` holds the per-row first-order conditions (the tangent imbalance
    contributions ``alpha psi_j - m(W, psi_j)``), ``H`` their Jacobian (the
    population Hessian) and ``dalpha_dbeta`` the derivative of alpha in beta.
    RW: ``IF = alpha Y - theta* - G' H^{-1} score``, ``G = E[gamma0 d alpha/d beta]``.
    ARW (gamma_hat OLS on the span, which contains gamma0): ``IF = m(W, gamma0) -
    theta0 + (alpha - delta' G_phi^{-1} phi) eps``. Var(eps) = 1. The SE of both
    estimators estimates ``Var(m(W, gamma0) + alpha eps)``.
    """
    g = gr()
    D, Z = X[:, 0], X[:, 1:]
    bas = basis()
    bas.fit(X)
    Phi = np.asarray(bas(X), dtype=float)
    M = np.asarray(g.ATEFunctional(0).m_basis_matrix(X, bas), dtype=float)
    g0 = gamma0(D, Z)
    m_g0 = gamma0(np.ones(len(D)), Z) - gamma0(np.zeros(len(D)), Z)
    theta_star = float(w @ (alpha * g0))
    delta = w @ (alpha[:, None] * Phi - M)
    G = (w * g0) @ dalpha_dbeta
    core = alpha * g0 - theta_star - score @ np.linalg.solve(H, G)
    E_a2 = float(w @ alpha**2)
    var_m = float(w @ (m_g0 - w @ m_g0) ** 2)
    G_phi = Phi.T @ (Phi * w[:, None])
    corr = Phi @ np.linalg.solve(G_phi, delta)
    return {
        "theta_star": theta_star,
        "b": theta_star - THETA0,
        "b_identity_gap": abs(theta_star - THETA0 - float(RHO @ delta)),
        "delta": delta,
        "sigma2_true_rw": float(w @ core**2) + E_a2,
        "sigma2_true_arw": var_m + float(w @ (alpha - corr) ** 2),
        "sigma2_se": var_m + E_a2,
        "E_alpha2": E_a2,
        "max_abs_alpha": float(np.max(np.abs(alpha))),
    }


def population_compatible(pair, points):
    """§1.9 (i)-(ii): the population dual solution (genriesz-reviewed Newton, tol 1e-13)."""
    g = gr()
    gen = generator(pair)
    X, w, *_ = quadrature(points)
    bas = basis()
    bas.fit(X)
    Phi = np.asarray(bas(X), dtype=float)
    M = np.asarray(g.ATEFunctional(0).m_basis_matrix(X, bas), dtype=float)
    off = np.asarray(gen.grad(X, alpha_ref(X)), dtype=float)
    sol = population.solve(generator=gen, X=X, w=w, Phi=Phi, M=M, offset=off)
    H = Phi.T @ (Phi * (w * gen.dual_eval(X, off + Phi @ sol.beta)[2])[:, None])
    eig = np.linalg.eigvalsh(0.5 * (H + H.T))
    out = {
        "status": sol.status,
        "beta": sol.beta,
        "max_gradient": sol.max_gradient,
        "min_hessian_eig": float(eig.min()),
        "max_hessian_eig": float(eig.max()),
    }
    if sol.status != "ok":
        return out, None
    dalpha = gen.dual_eval(X, off + Phi @ sol.beta)[2]
    mom = population_moments(
        X, w, sol.alpha, dalpha[:, None] * Phi, sol.alpha[:, None] * Phi - M, H
    )
    return out, mom


def _general_fit(pair, X, w, beta0=None):
    mdl = model(pair)
    res = mdl.fit(X, sample_weight=w, beta0=beta0, tol=CERT_GRADIENT)
    return mdl, res


def population_incompatible(pair, points, starts=None):
    """(i')-(ii') from the registered start ``v_ref`` (beta = 0), by the same damped
    Newton as the sample fits (``GRRGeneralLink.fit`` with the quadrature weights,
    stopping at max_j |dF/dbeta_j| <= 1e-12); (iv') from ``starts`` if given."""
    g = gr()
    X, w, *_ = quadrature(points)
    mdl, res = _general_fit(pair, X, w)
    out = {
        "status": res.status,
        "beta": res.beta,
        "max_gradient": res.tangent_imbalance,
        "n_iter": res.n_iter,
        "n_hessian_shifts": res.n_hessian_shifts,
    }
    if res.status != "ok":
        out.update(min_hessian_eig=float("nan"), max_hessian_eig=float("nan"))
        return out, None
    H = mdl.hessian(X, res.beta, sample_weight=w)
    eig = np.linalg.eigvalsh(0.5 * (H + H.T))
    out.update(min_hessian_eig=float(eig.min()), max_hessian_eig=float(eig.max()))
    if starts is not None:
        ms = []
        for b0 in starts:
            _, r = _general_fit(pair, X, w, beta0=b0)
            diff = float(np.max(np.abs(r.beta - res.beta))) if r.status == "ok" else None
            ms.append({"status": r.status, "max_abs_diff": diff})
        out["multistart"] = ms
    link, dlink, _ = LINKS[pair]
    v = V_REF[pair] + np.asarray(mdl.basis(X), dtype=float) @ res.beta
    Phi = np.asarray(mdl.basis(X), dtype=float)
    alpha = np.asarray(link(X, v), dtype=float)
    c, _ = mdl._coef(X, res.beta)
    M_psi = np.asarray(g.ATEFunctional(0).m_basis_matrix(X, mdl._tangent_basis(res.beta, P)))
    score = alpha[:, None] * c[:, None] * Phi - M_psi
    mom = population_moments(X, w, alpha, np.asarray(dlink(X, v))[:, None] * Phi, score, H)
    return out, mom


def multistart_points():
    """(iv'): 20 deterministic starts, the first 20 unscrambled Sobol points of
    dimension 8 mapped to the l_inf ball of radius 1 around beta = 0 (v = v_ref)."""
    from scipy.stats import qmc

    pts = qmc.Sobol(d=P, scramble=False).random_base2(m=5)[:MULTISTART]
    return 2.0 * pts - 1.0


def multistart_agrees(ms):
    """(iv'): all 20 starts are recorded, at least one converged (status ``ok``; no
    vacuous agreement), and every converged start is within 1e-8 (max abs) of the
    solution from the registered start."""
    conv = [m for m in ms if m["status"] == "ok"]
    return (
        len(ms) == MULTISTART
        and len(conv) > 0
        and all(m["max_abs_diff"] <= MULTISTART_TOL for m in conv)
    )


def stage0():
    """Every pair at 48 and 64 points, its certificate, and the predictions per n.

    Compatible pairs: §1.9 (i)-(iv). Incompatible pairs: (i')-(iv'). The
    population quantities are reported for every solved pair; ``c_n`` (RW and
    ARW) and the H13 predictions exist only for certified pairs (descriptive
    otherwise). JSON-compatible: unavailable values are ``None``; margins carry
    their kind.
    """
    from .inference import predicted_coverage

    starts = multistart_points()
    rows = []
    for pair in PAIRS:
        if pair in COMPATIBLE:
            r48, _ = population_compatible(pair, 48)
            r64, mom = population_compatible(pair, 64)
        else:
            r48, _ = population_incompatible(pair, 48)
            r64, mom = population_incompatible(pair, 64, starts=starts)
        solved = r48["status"] == "ok" and r64["status"] == "ok"
        beta_diff = float(np.max(np.abs(r48["beta"] - r64["beta"]))) if solved else None
        margins = support_margins(pair, r64["beta"]) if solved else {}
        checks = {
            "i_gradient": bool(
                solved
                and r48["max_gradient"] <= CERT_GRADIENT
                and r64["max_gradient"] <= CERT_GRADIENT
            ),
            "i_beta_diff": bool(solved and beta_diff <= CERT_BETA_DIFF),
            "iii_margins": bool(solved and all(m >= CERT_MARGIN for m in margins.values())),
        }
        if pair in COMPATIBLE:
            # (ii) positive definite Hessian; (iv) the dual objective is convex, so a
            # stationary point with a positive definite Hessian is the unique minimizer
            checks["ii_iv_positive_definite"] = bool(solved and r64["min_hessian_eig"] > 0)
        else:
            checks["ii_isolated"] = bool(
                solved and r64["min_hessian_eig"] >= CERT_EIG_RATIO * r64["max_hessian_eig"]
            )
            checks["iv_multistart"] = bool(solved and multistart_agrees(r64.get("multistart", [])))
        cert = bool(solved and all(checks.values()))
        row = {
            "pair": pair,
            "compatible": pair in COMPATIBLE,
            "certified": cert,
            "checks": checks,
            "status_48": r48["status"],
            "status": r64["status"],
            "beta_diff_48_64": beta_diff,
            "max_gradient_48": _finite_or_none(r48["max_gradient"]),
            "max_gradient_64": _finite_or_none(r64["max_gradient"]),
            "min_hessian_eig_64": _finite_or_none(r64["min_hessian_eig"]),
            "max_hessian_eig_64": _finite_or_none(r64["max_hessian_eig"]),
            "support_margins": {k: _finite_or_none(v) for k, v in margins.items()},
            "support_margin_kinds": {k: _margin_kind(v) for k, v in margins.items()},
            "beta": [_finite_or_none(x) for x in r64["beta"]],
        }
        if "multistart" in r64:
            row["multistart"] = r64["multistart"]
            row["multistart_converged"] = sum(m["status"] == "ok" for m in r64["multistart"])
        if mom is not None:
            row.update(
                {
                    k: (
                        [_finite_or_none(x) for x in v]
                        if isinstance(v, np.ndarray)
                        else _finite_or_none(v)
                    )
                    for k, v in mom.items()
                }
            )
        row["c_n"] = (
            {
                est: {
                    str(n): predicted_coverage(
                        np.sqrt(row["sigma2_se"]),
                        np.sqrt(row[f"sigma2_true_{est.split('_')[0].lower()}"]),
                        row["b"] if est == "RW_full" else 0.0,
                        n,
                    )
                    for n in N_VALUES
                }
                for est in ESTIMATORS
            }
            if cert
            else None
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------- Stage 1 replication


def _failed(status, **extra):
    return {"estimate": float("nan"), "se": float("nan"), "status": status, **extra}


def _ols(X, Y, X_fit, Y_fit):
    """Exact least squares on the treatment-interaction span; returns ``predict(rows)``."""
    bas = basis()
    bas.fit(X)
    coef, *_ = np.linalg.lstsq(np.asarray(bas(X_fit)), Y_fit, rcond=None)
    return lambda rows: np.asarray(bas(rows)) @ coef


def _m_ols(pred, Xt):
    n = len(Xt)
    return pred(np.column_stack([np.ones(n), Xt[:, 1:]])) - pred(
        np.column_stack([np.zeros(n), Xt[:, 1:]])
    )


def fit_representer(pair, X_fit):
    """Fit the pair on ``X_fit``. Returns ``(status, model, record)``.

    ``record`` (for a successful fit) holds the training imbalances: ``I_psi`` in
    the tangent directions (``max_j |Delta_hat(alpha_hat, psi_j)|``), ``I_phi`` in
    the regressors, the regressor imbalance vector ``Delta_hat`` and the scale
    ``max(1, ||P_n m(W, phi)||_inf)``. For a compatible pair the tangent
    directions are the regressors, so ``I_psi = I_phi``.
    """
    D = X_fit[:, 0]
    if not (np.any(D == 1) and np.any(D == 0)):
        return "degenerate_functional", None, None
    mdl = model(pair)
    if pair in COMPATIBLE:
        fr = mdl.fit(X_fit, tol=SAMPLE_TOL)
    else:
        fr = mdl.fit(X_fit, tol=GENERAL_TOL)
    if fr.status != "ok":
        return str(fr.status), mdl, None
    Phi = np.asarray(mdl.basis(X_fit), dtype=float)
    M = np.asarray(gr().ATEFunctional(0).m_basis_matrix(X_fit, mdl.basis), dtype=float)
    a = np.asarray(mdl.predict_alpha(X_fit), dtype=float)
    delta = np.mean(a[:, None] * Phi - M, axis=0)
    i_phi = float(np.max(np.abs(delta)))
    i_psi = i_phi if pair in COMPATIBLE else float(fr.tangent_imbalance)
    rec = {
        "I_psi": i_psi,
        "I_phi": i_phi,
        "delta": delta,
        "scale": max(1.0, float(np.max(np.abs(M.mean(axis=0))))),
    }
    return "ok", mdl, rec


class _Fits:
    """Warnings and training records of the individual fits of one estimator record."""

    def __init__(self):
        self.fits, self.records, self.converged, self.last_converged = [], [], True, True

    def run(self, stage, fold, fn):
        result, counts, converged = baselines.run_recording_warnings(fn)
        if counts:
            self.fits.append([int(fold), stage, counts])
        self.last_converged = converged
        self.converged = self.converged and converged
        return result

    def fields(self, other=None):
        other = other or {}
        n = sum(sum(c.values()) for _, _, c in self.fits) + sum(other.values())
        ips = [r["I_psi"] for r in self.records]
        iph = [r["I_phi"] for r in self.records]
        return {
            "train_fits": len(self.records),
            "I_psi_max": float(max(ips)) if ips else float("nan"),
            "I_phi_max": float(max(iph)) if iph else float("nan"),
            "scale_max": float(max(r["scale"] for r in self.records)) if ips else float("nan"),
            "n_warnings": int(n),
            "warnings": json.dumps({"fits": self.fits, "other": other}),
        }


def _rw_full(X, Y, pair):
    fits = _Fits()
    status, mdl, rec = fits.run("riesz", -1, lambda: fit_representer(pair, X))
    if status == "ok" and not fits.last_converged:
        status = "convergence_warning"
    detail = json.dumps([["full", "riesz", status]])
    if status != "ok":
        return _failed(status, fold_status=detail, **fits.fields())
    fits.records.append(rec)

    def infer():
        a = np.asarray(mdl.predict_alpha(X), dtype=float)
        theta = float(np.mean(a * Y))
        pred = _ols(X, Y, X, Y)
        psi = _m_ols(pred, X) + a * (Y - pred(X)) - theta
        # registration §1.4 RW_full: the sample SD of psi (ddof = 1, as rw_full_inference)
        return a, theta, float(np.sqrt(np.var(psi, ddof=1) / len(Y)))

    (a, theta, se), other, converged = baselines.run_recording_warnings(infer)
    common = {"fold_status": detail, **fits.fields(other)}
    if not converged:
        return _failed("convergence_warning", **common)
    return {
        "estimate": theta,
        "se": se,
        "status": "ok",
        "max_abs_alpha": float(np.max(np.abs(a))),
        "ess": metrics.ess(a),
        **{f"delta_hat_{j}": float(rec["delta"][j]) for j in range(P)},
        **common,
    }


def _arw_cf(X, Y, pair, folds):
    """ARW_cf (§1.4): per fold, the representer and the OLS regression fit on the
    complement; the score on the fold. A failed fold fails the replication with its
    own status; the balance of the folds fit before it is still recorded."""
    n = len(Y)
    psi, alpha = np.empty(n), np.empty(n)
    detail, eval_imb = [], []
    fits = _Fits()

    def run():
        for k in np.unique(folds):
            tr, te = folds != k, folds == k
            status, mdl, rec = fits.run("riesz", k, lambda tr=tr: fit_representer(pair, X[tr]))
            if status == "ok" and not fits.last_converged:
                status = "convergence_warning"
            detail.append([str(int(k)), "riesz", status])
            if status != "ok":
                return status
            fits.records.append(rec)
            Xt = X[te]
            # the observed rows of the fold and the counterfactual rows m evaluates (§1.3 C)
            pts = np.vstack([Xt, gr().ATEFunctional(0).evaluation_points(Xt, mdl.basis)])
            _, outside, nonfinite = mdl.classify(pts)
            if np.any(outside):
                detail.append([str(int(k)), "prediction", "domain_prediction"])
                return "domain_prediction"
            if np.any(nonfinite):
                detail.append([str(int(k)), "prediction", "nonfinite"])
                return "nonfinite"
            a = np.asarray(mdl.classify(Xt)[0], dtype=float)
            # evaluation-sample imbalance (§1.5): the fold whose representer was not fit on it
            Phi_t = np.asarray(mdl.basis(Xt), dtype=float)
            M_t = np.asarray(gr().ATEFunctional(0).m_basis_matrix(Xt, mdl.basis), dtype=float)
            eval_imb.append(float(np.max(np.abs(np.mean(a[:, None] * Phi_t - M_t, axis=0)))))
            pred = fits.run("outcome", k, lambda tr=tr: _ols(X, Y, X[tr], Y[tr]))
            psi[te] = _m_ols(pred, Xt) + a * (Y[te] - pred(Xt))
            alpha[te] = a
        return "ok" if np.all(np.isfinite(psi)) else "nonfinite"

    status, other, converged = baselines.run_recording_warnings(run)
    converged = converged and fits.converged
    common = {"fold_status": json.dumps(detail), **fits.fields(other)}
    if status != "ok" or not converged:
        return _failed(status if status != "ok" else "convergence_warning", **common)
    theta = float(np.mean(psi))
    return {
        "estimate": theta,
        "se": float(np.std(psi - theta) / np.sqrt(n)),  # §1.4: sigma^2 = P_n psi^2
        "status": "ok",
        "max_abs_alpha": float(np.max(np.abs(alpha))),
        "ess": metrics.ess(alpha),
        "eval_imbalance": float(max(eval_imb)),
        **common,
    }


def replicate(task):
    """One replication of one cell: every pair and estimator. ``task = (cell, rep, entropy)``."""
    cell_index, rep, entropy = task
    (n,) = CELLS[cell_index]
    seeds = Seeds(EXP, entropy=entropy)
    D, Z, Y = draw(seeds.data(cell_index, rep), n)
    X = np.column_stack([D, Z])
    folds = fold_ids(n, K, seeds.folds(cell_index, rep))
    out = {}
    for pair in PAIRS:
        out[f"{pair}|RW_full"] = _rw_full(X, Y, pair)
        out[f"{pair}|ARW_cf"] = _arw_cf(X, Y, pair, folds)
    return out


def tasks_for(entropy, reps, cells=None):
    cells = range(len(CELLS)) if cells is None else cells
    return [(ci, r, entropy) for ci in cells for r in range(reps)]


# ---------------------------------------------------------------- aggregation

NUMERIC_FIELDS = (
    "estimate", "se", "max_abs_alpha", "ess", "train_fits", "I_psi_max", "I_phi_max",
    "scale_max", "eval_imbalance", "n_warnings", *(f"delta_hat_{j}" for j in range(P)),
)  # fmt: skip
TEXT_FIELDS = ("status", "warnings", "fold_status")


def raw_frame(tasks, results):
    """One row per replication x pair/estimator; failures are rows too (§1.7)."""
    import pandas as pd

    rows = []
    for (ci, rep, _), res in zip(tasks, results, strict=True):
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
    return int(raw["n_warnings"].sum())


def balance_checks(raw):
    """H13 (a) and (b) (deterministic). Any violation raises: an implementation error
    stops the run.

    (a) every successful fit (RW_full; each ARW_cf fold, also those fit before a
    later fold failed) has ``I_psi <= 1e-8``; (b) every successful fit of a
    compatible pair has ``I_phi <= 1e-8``. A successful record carries the
    balance of all its fits (1 for RW_full, K for ARW_cf).
    """
    ok = raw["status"] == "ok"
    expected = np.where(raw["arm"].str.endswith("RW_full"), 1, K)
    missing = int((ok & (raw["train_fits"] != expected)).sum())
    has = raw["train_fits"] > 0
    compatible = raw["arm"].str.startswith("C-")
    i_psi = raw.loc[has, "I_psi_max"]
    i_phi = raw.loc[has & compatible, "I_phi_max"]
    bal = {
        "fits_checked": int(raw.loc[has, "train_fits"].sum()),
        "I_psi_max": float(i_psi.max()) if len(i_psi) else None,
        "I_phi_max_compatible": float(i_phi.max()) if len(i_phi) else None,
        "I_phi_max_incompatible": (
            float(raw.loc[has & ~compatible, "I_phi_max"].max())
            if (has & ~compatible).any()
            else None
        ),
        "scale_max": float(raw.loc[has, "scale_max"].max()) if has.any() else None,
        "tolerance": BALANCE_TOL,
    }
    violations = []
    if missing:
        violations.append(f"{missing} successful records without the balance of every fit")
    if len(i_psi) and not i_psi.max() <= BALANCE_TOL:
        violations.append("(a) tangent imbalance above 1e-8")
    if len(i_phi) and not i_phi.max() <= BALANCE_TOL:
        violations.append("(b) compatible regressor imbalance above 1e-8")
    if violations:
        raise AssertionError(f"H13 (a)/(b) violated: {violations}; {json.dumps(bal)}")
    return bal


def summarise(raw, population_rows):
    """Registered metrics (§1.5) per cell x pair/estimator, in the registered order,
    with the Stage 0 predictions of certified pairs (RW: ``b``; ARW: ``b = 0``)."""
    import pandas as pd

    from .e12 import POSITIVE_VARIANCE, moment_status

    pop = {r["pair"]: r for r in population_rows}
    rows = []
    for ci, (n,) in enumerate(CELLS):
        for label in ARM_LABELS:
            g = raw[(raw["cell"] == ci) & (raw["arm"] == label)].sort_values("rep")
            if g.empty:  # e.g. the per-cell aggregation of Stage 0.5
                continue
            ok = (g["status"] == "ok").to_numpy()
            est, se = g["estimate"].to_numpy(), g["se"].to_numpy()
            pair, _, estimator = label.partition("|")
            row = {"cell": ci, "n": n, "arm": label, "pair": pair, "estimator": estimator}
            row.update(metrics.failure_rate(ok))
            row.update(metrics.coverage(est, se, ok, THETA0))
            row["status_counts"] = json.dumps(
                dict(sorted(collections.Counter(g["status"]).items()))
            )
            row["warnings"] = int(g["n_warnings"].sum())
            x = est[ok]
            ms = row["moment_status"] = moment_status(x)
            r_s = int(ok.sum())
            if r_s >= 1:
                row.update(metrics.ci_length(se, ok))
                row.update(metrics.max_weight_summary(g["max_abs_alpha"].to_numpy(), ok))
                row["ess_median"] = float(g.loc[ok, "ess"].median())
                row["I_psi_max"] = float(g.loc[ok, "I_psi_max"].max())
                row["I_phi_median"] = float(g.loc[ok, "I_phi_max"].median())
                row["I_phi_max"] = float(g.loc[ok, "I_phi_max"].max())
                if estimator == "ARW_cf":
                    row["eval_imbalance_median"] = float(g.loc[ok, "eval_imbalance"].median())
                    row["eval_imbalance_max"] = float(g.loc[ok, "eval_imbalance"].max())
                if estimator == "RW_full":
                    for j in range(P):
                        row[f"delta_hat_mean_{j}"] = float(g.loc[ok, f"delta_hat_{j}"].mean())
            if r_s >= 2:
                var, m4 = metrics.moment_terms(x)
                row.update({"moment_var": var, "moment_m4": m4, "moment_radicand": m4 - var**2})
                row.update(metrics.bias(est, ok, THETA0))
                row["rmse"] = metrics.rmse(est, ok, THETA0)
                row["root_n_sd"] = float(np.sqrt(n) * x.std(ddof=1))
            if ms in POSITIVE_VARIANCE:
                row["se_ratio"] = metrics.se_ratio(est, se, ok)
            if ms == "ok":
                row.update(metrics.root_n_sd(est, ok, n))
            elif ms == "zero_radicand":
                row["root_n_sd_mcse"] = 0.0
            p = pop[pair]
            row["certified"] = bool(p["certified"])
            if p["certified"]:
                est_key = "rw" if estimator == "RW_full" else "arw"
                row.update(
                    {
                        "b_pred": p["b"] if estimator == "RW_full" else 0.0,
                        "sigma_true": float(np.sqrt(p[f"sigma2_true_{est_key}"])),
                        "root_n_sd_pred": float(np.sqrt(p[f"sigma2_true_{est_key}"])),
                        "sigma_se_pred": float(np.sqrt(p["sigma2_se"])),
                        "c_pred": p["c_n"][estimator][str(n)],
                    }
                )
                if estimator == "RW_full":
                    for j in range(P):
                        row[f"delta_pop_{j}"] = p["delta"][j]
            rows.append(row)
    return pd.DataFrame(rows)


def family_h13(summ, raw, seeds):
    """H13 (c)-(e), one family (Holm 0.01, §1.8), certified pairs only.

    (c) incompatible pairs, n = 4000, RW_full: the mean of ``Delta_hat_j`` against
    ``[delta_j - 0.05|delta_j|, delta_j + 0.05|delta_j|]``, se from family 2
    (B = 10000, stream 4; successful replications resampled; calls in the order
    pair, j). (d) RW_full bias against ``b`` (compatible: §1.8 bias null with
    ``b = 0``; incompatible: ``[b - 0.1|b|, b + 0.1|b|]``, se = SD/sqrt(R_s)).
    (e) RW_full and ARW_cf unconditional coverage, ``|c - c(n)| <= 0.03``.
    An undefined test stays in the family as a non-rejecting entry (p = 1) and is
    listed in ``unavailable``.
    """
    from . import inference
    from .bootstrap import FamilyBootstrap, bootstrap_se
    from .e12 import POSITIVE_VARIANCE, UNAVAILABLE_P

    tests, unavailable = {}, []
    cert = summ[summ["certified"] == True]  # noqa: E712
    fb = FamilyBootstrap(seeds, 2)
    for pair in INCOMPATIBLE:
        r = cert[(cert["pair"] == pair) & (cert["n"] == 4000) & (cert["estimator"] == "RW_full")]
        if r.empty:
            continue
        r = r.iloc[0]
        g = raw[(raw["cell"] == r["cell"]) & (raw["arm"] == r["arm"]) & (raw["status"] == "ok")]
        for j in range(P):
            key = f"(c)|{pair}|n=4000|delta_{j}"
            x = g.sort_values("rep")[f"delta_hat_{j}"].to_numpy()
            if x.size < 2:
                tests[key] = UNAVAILABLE_P
                unavailable.append(f"{key} (fewer_than_two)")
                continue
            boot = fb.replicate(lambda idx, v=x: v[idx].mean(axis=1), x.size)
            se = bootstrap_se(boot, 2)
            if not se > 0:
                tests[key] = UNAVAILABLE_P
                unavailable.append(f"{key} (zero_bootstrap_se)")
                continue
            d = float(r[f"delta_pop_{j}"])
            tests[key] = inference.interval_null_pvalue(
                float(x.mean()), se, d - DELTA_REL_TOL * abs(d), d + DELTA_REL_TOL * abs(d)
            )
    for _, r in cert[cert["estimator"] == "RW_full"].iterrows():
        key = f"(d)|{r['pair']}|n={r['n']}|bias"
        g = raw[(raw["cell"] == r["cell"]) & (raw["arm"] == r["arm"]) & (raw["status"] == "ok")]
        x = g["estimate"].to_numpy()
        if r["moment_status"] not in POSITIVE_VARIANCE:
            tests[key] = UNAVAILABLE_P
            unavailable.append(f"{key} ({r['moment_status']})")
        elif r["pair"] in COMPATIBLE:
            tests[key] = inference.bias_pvalue(x, THETA0, 0.0, r["sigma_true"], int(r["n"]))
        else:
            b = float(r["b_pred"])
            se = float(x.std(ddof=1) / np.sqrt(x.size))
            tests[key] = inference.interval_null_pvalue(
                float(x.mean()) - THETA0, se, b - BIAS_REL_TOL * abs(b), b + BIAS_REL_TOL * abs(b)
            )
    for _, r in cert.iterrows():
        key = f"(e)|{r['arm']}|n={r['n']}|coverage"
        tests[key] = inference.coverage_pvalue(
            int(r["covered"]), int(r["R"]), float(r["c_pred"]), tol=COVERAGE_TOL
        )
    verdict = inference.judge_family("H13", tests) if tests else None
    return verdict, tests, unavailable


# ---------------------------------------------------------------- table and figure (summary.csv)

LABEL = {
    "C-SQ": "C-SQ",
    "C-UKL1": "C-UKL ($C=1$)",
    "C-BKL1": "C-BKL ($C=1$)",
    "I-SQexp": "I-SQexp",
    "I-BKLexp": "I-BKLexp",
    "I-UKLlin": "I-UKLlin",
}
END = " \\\\"


def _fmt(x, d=3):
    if x is None or x != x:
        return "--"
    s = f"{float(x):.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s  # no "-0.000"


def _sci(x):
    return "--" if x is None or x != x else f"{float(x):.1e}"


def tables(S):
    """``tab_E13`` from the rows of ``summary.csv`` only (§4)."""
    heads = [
        "Pair", "Estimator", "$n$", "Cert.", "Bias ($b$)", "$\\sqrt{n}$SD (pred.)", "SE ratio",
        "Coverage ($c(n)$)", "Cov. succ.", "Fail \\%", "$\\max I_\\psi$", "med. $I_\\phi$",
    ]  # fmt: skip
    lines = ["\\begin{tabular}{llrlrrrrrrrr}", "\\hline", " & ".join(heads) + END, "\\hline"]
    for pair in PAIRS:
        for est in ESTIMATORS:
            for n in N_VALUES:
                r = S[(S["pair"] == pair) & (S["estimator"] == est) & (S["n"] == n)].iloc[0]
                lines.append(
                    " & ".join(
                        [
                            LABEL[pair], est.replace("_", "\\_"), str(n),
                            "yes" if r["certified"] else "no",
                            f"{_fmt(r.get('bias'))} ({_fmt(r.get('b_pred'))})",
                            f"{_fmt(r.get('root_n_sd'), 2)} ({_fmt(r.get('root_n_sd_pred'), 2)})",
                            _fmt(r.get("se_ratio"), 2),
                            f"{_fmt(r.get('coverage'))} ({_fmt(r.get('c_pred'))})",
                            _fmt(r.get("coverage_conditional")),
                            f"{100 * r['failure_rate']:.1f}",
                            _sci(r.get("I_psi_max")), _sci(r.get("I_phi_median")),
                        ]
                    )
                    + END
                )  # fmt: skip
    fam = json.loads(S["family_verdict"].iloc[0]) if "family_verdict" in S else None
    lines += ["\\hline", "\\end{tabular}"]
    status = ["\\begin{tabular}{llrrl}", "\\hline",
              "Pair & Estimator & $n$ & Successes & Statuses" + END, "\\hline"]  # fmt: skip
    for pair in PAIRS:
        for est in ESTIMATORS:
            for n in N_VALUES:
                r = S[(S["pair"] == pair) & (S["estimator"] == est) & (S["n"] == n)].iloc[0]
                if int(r["R_s"]) == int(r["R"]):
                    continue
                counts = ", ".join(
                    f"{k.replace('_', chr(92) + '_')}: {v}"
                    for k, v in json.loads(r["status_counts"]).items()
                )
                status.append(
                    " & ".join([LABEL[pair], est.replace("_", "\\_"), str(n), str(int(r["R_s"])),
                                counts]) + END  # fmt: skip
                )
    status += ["\\hline", "\\end{tabular}"]
    return {"tab_E13": "\n".join(lines) + "\n", "tab_E13_status": "\n".join(status) + "\n"}, fam


def figure(S):
    """``fig_E13_tangent``: (left) training imbalance in the tangent directions and in
    the regressors per pair (n = 4000, RW_full); (right) mean Delta_hat_j against the
    population delta_j for the incompatible pairs (n = 4000, certified pairs)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    sub = S[(S["n"] == 4000) & (S["estimator"] == "RW_full")].set_index("pair").loc[list(PAIRS)]
    xs = np.arange(len(PAIRS))
    floor = 1e-18
    ax1.bar(xs - 0.2, np.maximum(sub["I_psi_max"].to_numpy(float), floor), 0.4,
            label="$\\max I_\\psi$ (tangent)", color="#4C72B0")
    ax1.bar(xs + 0.2, np.maximum(sub["I_phi_median"].to_numpy(float), floor), 0.4,
            label="median $I_\\phi$ (regressors)", color="#DD8452")  # fmt: skip
    ax1.axhline(BALANCE_TOL, color="k", lw=0.8, ls="--")
    ax1.set_yscale("log")
    ax1.set_xticks(xs, list(PAIRS), rotation=30, ha="right")
    ax1.set_ylabel("training imbalance")
    ax1.legend(fontsize=7, loc="upper left")
    markers = {"I-SQexp": "o", "I-BKLexp": "s", "I-UKLlin": "^"}
    lim = 0.0
    for pair in INCOMPATIBLE:
        r = sub.loc[pair]
        if not bool(r["certified"]):
            continue
        d = np.array([r[f"delta_pop_{j}"] for j in range(P)], float)
        m = np.array([r[f"delta_hat_mean_{j}"] for j in range(P)], float)
        ax2.scatter(d, m, marker=markers[pair], label=pair, s=22)
        lim = max(lim, float(np.max(np.abs(np.concatenate([d, m])))))
    lim = 1.05 * lim if lim > 0 else 1.0
    ax2.plot([-lim, lim], [-lim, lim], color="k", lw=0.8)
    ax2.set_xlabel("population $\\delta_j$")
    ax2.set_ylabel("mean $\\widehat\\Delta_j$ ($n=4000$)")
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(fontsize=7)
    fig.tight_layout()
    return fig
