"""E-19: inference for the average marginal effect (H19-Inf, H19-Cex; PR4c
``cor:ame_asymp_normality`` and ``rem:ame_conditions``).

Registration: doc/2026-10-07_experiment_registration.md, E-19, §1.4–§1.9. This module
holds the two DGPs, the Stage 0 population computation, the replication function
(importable for process parallelism) and the aggregation shared by the pilot and the
confirmatory stage. The notebook ``16_E19_ame_inference.ipynb`` runs the stages and
writes the outputs; its last cell makes the table and the figure from ``summary.csv``
only (§4).

Three parts (cells in this order, one stream-0 key per cell and replication):

- ``inference`` (DGP19Q, R = 2000, n in {1000, 2000, 4000}): SQ-Riesz (``lambda = 0``,
  basis ``{1, d, d^3, z1, sin z2}``, which contains ``alpha_0``) and OLS on
  ``{1, sin d, d, d z1, z1, z2, z2^2}`` (which contains ``gamma_0``), both cross-fitted
  with ``K = 5``; ARW (H19-Inf) and TMLE (reported, no hypothesis).
- ``counterexamples`` (DGP19Q, R = 5000, n in {1000, 4000}): the oracle perturbations
  of ``rem:ame_conditions``, not fitted: (i) ``alpha_hat = alpha_0``, ``gamma_hat =
  gamma_0 + j_n^{-1} sin(j_n d)``, ``j_n = ceil(n^{1/2})``; (ii) ``alpha_hat = alpha_0``,
  ``gamma_hat = gamma_0 + n^{-1/4} 1[d > 0]`` with the almost-everywhere derivative.
- ``illustration`` (DGP19L, R = 2000, n in {1000, 2000, 4000}; descriptive): SQ-Riesz on
  200 random Fourier features and ridge regression on another 200, cross-fitted.

``X = (d, z1, z2)``; the functional is the derivative in ``d`` (coordinate 0).
"""

from __future__ import annotations

import collections
import json
import math

import numpy as np

from . import baselines, metrics
from .seeds import Seeds, fold_ids

EXP = 19
SQRT3 = np.sqrt(3.0)
K = 5
PARTS = ("inference", "counterexamples", "illustration")
R_PART = {"inference": 2000, "counterexamples": 5000, "illustration": 2000}
N_PART = {
    "inference": (1000, 2000, 4000),
    "counterexamples": (1000, 4000),
    "illustration": (1000, 2000, 4000),  # not fixed by the registration (AI choice)
}
CELLS = [[p, n] for p in PARTS for n in N_PART[p]]  # registered order
ESTIMATORS = {
    "inference": ("ARW_cf", "TMLE_cf"),
    "counterexamples": ("Cex_i", "Cex_ii"),
    "illustration": ("ARW_cf", "TMLE_cf"),
}
ARM_LABELS = [f"{p}|{e}" for p in PARTS for e in ESTIMATORS[p]]

#: §1.3 C, lambda = 0 and no constraint: max_j |grad_j| <= 1e-8 max(1, ||P_n m(W, phi)||_inf)
SAMPLE_TOL = 1e-8
CERT_GRADIENT = 1e-12  # §1.9 (i)
CERT_BETA_DIFF = 1e-10  # §1.9 (i)
#: H19-Cex (i): |n Var(theta_hat)/(V* + 1/2) - 1| <= 0.06
CEX_VAR_DELTA = 0.06
#: D | Z quadrature: Gauss-Legendre on [-D_HALF_WIDTH, D_HALF_WIDTH] (the density is below
#: exp(-0.1 * 10^4 + 13) there); Z: Gauss-Legendre on the support box.
D_HALF_WIDTH = 10.0
#: Counterexample (i) oscillates as cos(j d) with j up to 64: a finer d rule (composite
#: Gauss-Legendre, panels x points) for its exact finite-n reference value.
CEX_PANELS = (400, 800)
CEX_PANEL_POINTS = 16
#: illustration (DGP19L): random Fourier features (AI choice; the registration fixes only
#: the count). Bandwidth 1 on standardized inputs, l2 penalties 1e-3 for both nuisances.
RFF_FEATURES = 200
RFF_SIGMA = 1.0
RIESZ_LAM = 1e-3
RIDGE_LAM = 1e-3
ALPHA0_COEF = np.array([0.0, 1.0, 0.4, -0.5, -0.3])  # alpha_0 on {1, d, d^3, z1, sin z2}


def gr():
    import genriesz

    return genriesz


# ---------------------------------------------------------------- DGPs


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


def alpha0_l(X):
    """DGP19L (``D = mu + L``, L standard logistic): ``tanh((d - mu)/2)``."""
    return np.tanh((X[:, 0] - mu(X[:, 1:])) / 2.0)


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


def draw_l(rng, n):
    """DGP19L: as DGP19Q with ``D = mu + L``, L standard logistic."""
    Z = rng.uniform(-SQRT3, SQRT3, size=(n, 2))
    D = mu(Z) + rng.logistic(size=n)
    X = np.column_stack([D, Z])
    Y = gamma0(X) + rng.standard_normal(n)
    return X, Y


# ---------------------------------------------------------------- dictionaries


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


def rff_basis(seed):
    return gr().RBFRandomFourierBasis(
        n_features=RFF_FEATURES, sigma=RFF_SIGMA, standardize=True, random_state=int(seed)
    )


def _finite_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------- Stage 0


def _gl(a, b, points):
    x, w = np.polynomial.legendre.leggauss(points)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def _composite_gl(a, b, panels, points):
    edges = np.linspace(a, b, panels + 1)
    xs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        x, w = _gl(lo, hi, points)
        xs.append(x)
        ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


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


def d_rule(points):
    """The d rule paired with ``points`` Gauss-Legendre nodes per z axis: composite
    Gauss-Legendre with ``points`` panels of 16 nodes on [-10, 10] (a single
    Gauss-Legendre rule of 48 or 64 nodes on that interval agrees only to about 1e-5)."""
    return _composite_gl(-D_HALF_WIDTH, D_HALF_WIDTH, points, CEX_PANEL_POINTS)


def population_q(points):
    """Population quantities of DGP19Q on a rule with ``points`` nodes per z axis and
    :func:`d_rule` in d."""
    X, w = quadrature_q(points, d_rule(points))
    a0 = alpha0_q(X)
    theta0 = float(w @ dgamma0(X))
    psi0 = dgamma0(X) - theta0
    e_a2 = float(w @ a0**2)
    v_star = float(w @ psi0**2) + e_a2  # Var(eps) = 1
    Phi, dPhi = feat_alpha(X), dfeat_alpha(X)
    G = Phi.T @ (Phi * w[:, None])
    mbar = w @ dPhi
    # SQ (g = alpha^2, C = 0): population objective E[alpha^2 - 2 m(W, alpha)], alpha = Phi beta
    beta = np.linalg.solve(G, mbar)
    grad = 2.0 * (G @ beta - mbar)
    eig = np.linalg.eigvalsh(2.0 * G)
    # Riesz identity on the dictionaries (alpha_0 is the representer): E[alpha_0 phi] = E[d phi]
    riesz_gap = float(
        max(
            np.max(np.abs(w @ (a0[:, None] * Phi) - mbar)),
            np.max(np.abs(w @ (a0[:, None] * feat_gamma(X)) - w @ dfeat_gamma(X))),
        )
    )
    return {
        "theta0": theta0,
        "E_alpha0_sq": e_a2,
        "V_star": v_star,
        "beta_sq": beta,
        "max_gradient": float(np.max(np.abs(grad))),
        "min_hessian_eig": float(eig.min()),
        "riesz_identity_gap": riesz_gap,
        "alpha_star_minus_alpha0_max": float(np.max(np.abs(Phi @ beta - a0))),
        "f_D_at_0": f_d_zero(points),
    }


def f_d_zero(points):
    """``f_D(0) = E_Z f(0 | Z)`` for DGP19Q."""
    zx, zw = _gl(-SQRT3, SQRT3, points)
    zw = zw / (2.0 * SQRT3)
    dx, dw = d_rule(points)
    Z1, Z2 = np.meshgrid(zx, zx, indexing="ij")
    Z = np.column_stack([Z1.ravel(), Z2.ravel()])
    wz = np.outer(zw, zw).ravel()
    m = mu(Z)
    c = (np.exp(-((dx[None, :] - m[:, None]) ** 2) / 2.0 - 0.1 * dx[None, :] ** 4) * dw).sum(1)
    f0 = np.exp(-(m**2) / 2.0) / c
    return float(wz @ f0)


def cex_i_variance(n, z_points, panels):
    """Exact ``n Var(theta_hat)`` of counterexample (i) at sample size ``n``:
    ``Var(psi_0 + cos(j D) - alpha_0 sin(j D)/j)``, ``j = ceil(sqrt n)``."""
    j = math.ceil(math.sqrt(n))
    X, w = quadrature_q(z_points, _composite_gl(-D_HALF_WIDTH, D_HALF_WIDTH, panels, CEX_PANEL_POINTS))
    theta0 = float(w @ dgamma0(X))
    d = X[:, 0]
    s = dgamma0(X) - theta0 + np.cos(j * d) - alpha0_q(X) * np.sin(j * d) / j
    mean = float(w @ s)
    return {"j": j, "mean": mean, "variance": float(w @ s**2) + float(w @ alpha0_q(X) ** 2) - mean**2}


def population_l(points):
    """``theta_0``, ``E alpha_0^2`` and ``V*`` of DGP19L (logistic noise), by the same
    tensor rule with the logistic density of ``D - mu``."""
    zx, zw = _gl(-SQRT3, SQRT3, points)
    zw = zw / (2.0 * SQRT3)
    ux, uw = _gl(-40.0, 40.0, 4 * points)
    f = np.exp(-np.abs(ux)) / (1.0 + np.exp(-np.abs(ux))) ** 2
    Z1, Z2 = np.meshgrid(zx, zx, indexing="ij")
    Z = np.column_stack([Z1.ravel(), Z2.ravel()])
    wz = np.outer(zw, zw).ravel()
    D = mu(Z)[:, None] + ux[None, :]
    w = (wz[:, None] * (f * uw)[None, :]).ravel()
    X = np.column_stack([D.ravel(), np.repeat(Z, len(ux), 0)])
    theta0 = float(w @ dgamma0(X))
    e_a2 = float(w @ alpha0_l(X) ** 2)
    return {
        "theta0": theta0,
        "E_alpha0_sq": e_a2,
        "V_star": float(w @ (dgamma0(X) - theta0) ** 2) + e_a2,
        "mass": float(w.sum()),
    }


def stage0():
    """The population quantities at 48 and 64 nodes per axis, the §1.9 certificate of the
    SQ population minimizer (alpha_0 lies in the span, so the minimizer is alpha_0), the
    H19-Inf predictions (b = 0, sigma_true = sigma_SE = sqrt(V*), c(n) = 0.95), the
    H19-Cex predictions and the exact finite-n reference values of counterexample (i)."""
    from .inference import predicted_coverage

    q48, q64 = population_q(48), population_q(64)
    beta_diff = float(np.max(np.abs(q48["beta_sq"] - q64["beta_sq"])))
    checks = {
        "i_gradient": bool(
            q48["max_gradient"] <= CERT_GRADIENT and q64["max_gradient"] <= CERT_GRADIENT
        ),
        "i_beta_diff": bool(beta_diff <= CERT_BETA_DIFF),
        "ii_positive_definite": bool(q64["min_hessian_eig"] > 0),
        # (iii) SQ has domain R: no finite end; (iv) the SQ objective is convex.
        "iii_domain_unbounded": True,
        "alpha_star_is_alpha0": bool(
            float(np.max(np.abs(q64["beta_sq"] - ALPHA0_COEF))) <= CERT_BETA_DIFF
        ),
    }
    v = q64["V_star"]
    inference = {
        "certified": bool(all(checks.values())),
        "checks": checks,
        "theta0": q64["theta0"],
        "theta0_48": q48["theta0"],
        "V_star": v,
        "V_star_48": q48["V_star"],
        "E_alpha0_sq": q64["E_alpha0_sq"],
        "beta_sq": [float(x) for x in q64["beta_sq"]],
        "beta_diff_48_64": beta_diff,
        "max_gradient_48": q48["max_gradient"],
        "max_gradient_64": q64["max_gradient"],
        "min_hessian_eig_64": q64["min_hessian_eig"],
        "riesz_identity_gap_64": q64["riesz_identity_gap"],
        "b": 0.0,
        "sigma_true": math.sqrt(v),
        "sigma_se": math.sqrt(v),
        "c_n": {str(n): predicted_coverage(math.sqrt(v), math.sqrt(v), 0.0, n)
                for n in N_PART["inference"]},  # fmt: skip
    }
    fd0 = q64["f_D_at_0"]
    cex = {
        "f_D_at_0": fd0,
        "f_D_at_0_48": q48["f_D_at_0"],
        "V_star_plus_half": v + 0.5,
        "i": {
            str(n): {
                "prediction_Q": 1.0,
                "exact": [cex_i_variance(n, 48, CEX_PANELS[0]), cex_i_variance(n, 64, CEX_PANELS[1])],
            }
            for n in N_PART["counterexamples"]
        },
        "ii": {str(n): {"mean_root_n_error": -(n**0.25) * fd0} for n in N_PART["counterexamples"]},
    }
    for n in N_PART["counterexamples"]:
        ex = cex["i"][str(n)]["exact"][1]
        cex["i"][str(n)]["exact_Q"] = ex["variance"] / (v + 0.5)
    lq48, lq64 = population_l(48), population_l(64)
    illustration = {
        "theta0": lq64["theta0"],
        "theta0_48": lq48["theta0"],
        "V_star": lq64["V_star"],
        "E_alpha0_sq": lq64["E_alpha0_sq"],
        "mass": lq64["mass"],
    }
    return {"inference": inference, "counterexamples": cex, "illustration": illustration}


# ---------------------------------------------------------------- Stage 1 replication


def _failed(status, **extra):
    return {"estimate": float("nan"), "se": float("nan"), "status": status, **extra}


class _Rec:
    """Warnings of the individual fits of one estimator record (as E-12/E-13)."""

    def __init__(self):
        self.fits, self.converged, self.last_converged = [], True, True

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
        return {"n_warnings": int(n), "warnings": json.dumps({"fits": self.fits, "other": other})}


def _fit_sq_riesz(X_fit, basis, penalty, lam):
    """SQ-Riesz (C = 0, compatible link alpha = u/2, no offset) for the AME. Returns
    ``(status, alpha_fn, dalpha_fn, train_imbalance, fitted_basis)``."""
    g = gr()
    mdl = g.GRRGLM(
        basis=basis, generator=g.SquaredGenerator(C=0.0), functional=g.AMEFunctional(0),
        penalty=penalty, lam=lam,
    )  # fmt: skip
    fr = mdl.fit(X_fit, tol=SAMPLE_TOL)
    if fr.status != "ok":
        return str(fr.status), None, None, None, None
    beta = np.asarray(mdl.beta_, dtype=float)

    def alpha(rows):
        return np.asarray(mdl.predict_alpha(rows), dtype=float)

    def dalpha(rows):
        # alpha = (Phi beta)/2 for SQ with C = 0 and no offset
        return 0.5 * np.asarray(mdl.basis.derivative(rows, 0), dtype=float) @ beta

    Phi = np.asarray(mdl.basis(X_fit), dtype=float)
    M = np.asarray(mdl.basis.derivative(X_fit, 0), dtype=float)
    imb = float(np.max(np.abs(np.mean(alpha(X_fit)[:, None] * Phi - M, axis=0))))
    return "ok", alpha, dalpha, imb, mdl.basis


def _ols(F, y, lam=0.0):
    """Least squares (``lam = 0``: exact, minimum-norm) or ridge ``(F'F/n + lam I)^{-1} F'y/n``."""
    if lam == 0.0:
        coef, *_ = np.linalg.lstsq(F, y, rcond=None)
        return coef
    n, p = F.shape
    return np.linalg.solve(F.T @ F / n + lam * np.eye(p), F.T @ y / n)


def _finite(*values):
    return all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in values)


def _scores(a, da, g_hat, dg_hat, Y):
    """ARW (§1.4) and TMLE (§1.4, §1.3 E) from the cross-fitted nuisances. Returns
    ``(status, arw, tmle)``; every quantity is checked before ``ok`` is returned."""
    n = len(Y)
    psi = dg_hat + a * (Y - g_hat)
    theta = float(np.mean(psi))
    se = float(np.std(psi - theta) / np.sqrt(n))  # §1.4: sigma^2 = P_n psi^2
    denom = float(np.sum(a**2))
    if not (_finite(psi, theta, se, denom) and denom > 0):
        return "nonfinite", None, None
    # TMLE: linear fluctuation of the cross-fitted regression along alpha_hat
    eps = float(np.sum(a * (Y - g_hat)) / denom)
    g_star, dg_star = g_hat + eps * a, dg_hat + eps * da
    theta_t = float(np.mean(dg_star))
    psi_t = dg_star + a * (Y - g_star) - theta_t
    se_t = float(np.sqrt(np.mean(psi_t**2) / n))
    if not _finite(eps, g_star, dg_star, theta_t, psi_t, se_t):
        return "nonfinite", None, None
    return "ok", (theta, se), (theta_t, se_t, eps)


def _crossfit(X, Y, folds, riesz_basis, gamma_feats, penalty, lam_riesz, lam_ridge):
    """ARW_cf and TMLE_cf (§1.4) with fold-wise SQ-Riesz and (ridge) least squares.

    The nuisances, the scores, the fluctuation and the SEs are computed inside the
    warning recorder; a non-finite value anywhere fails both arms (``nonfinite``)."""
    n = len(Y)
    a, da, g_hat, dg_hat = (np.empty(n) for _ in range(4))
    detail, imb, eval_imb = [], [], []
    rec = _Rec()
    out = {}

    def run():
        for k in np.unique(folds):
            tr, te = folds != k, folds == k
            status, af, daf, im, bas = rec.run(
                "riesz", k, lambda tr=tr: _fit_sq_riesz(X[tr], riesz_basis(k), penalty, lam_riesz)
            )
            if status == "ok" and not rec.last_converged:
                status = "convergence_warning"
            detail.append([str(int(k)), "riesz", status])
            if status != "ok":
                return status
            imb.append(im)
            feats, dfeats = gamma_feats(k, X[tr])
            coef = rec.run("outcome", k, lambda tr=tr: _ols(feats(X[tr]), Y[tr], lam_ridge))
            Xt = X[te]
            a[te], da[te] = af(Xt), daf(Xt)
            g_hat[te], dg_hat[te] = feats(Xt) @ coef, dfeats(Xt) @ coef
            if not _finite(a[te], da[te], g_hat[te], dg_hat[te]):
                detail.append([str(int(k)), "prediction", "nonfinite"])
                return "nonfinite"
            # evaluation-sample imbalance (§1.5) on the Riesz dictionary of the fold
            eval_imb.append(_eval_imbalance(af, bas, Xt))
        status, arw, tmle = _scores(a, da, g_hat, dg_hat, Y)
        out.update(arw=arw, tmle=tmle)
        return status

    status, other, converged = baselines.run_recording_warnings(run)
    converged = converged and rec.converged
    common = {"fold_status": json.dumps(detail), **rec.fields(other),
              "train_imbalance_max": float(max(imb)) if imb else float("nan")}  # fmt: skip
    if status != "ok" or not converged:
        failed = _failed(status if status != "ok" else "convergence_warning", **common)
        return failed, dict(failed)
    weights = {"max_abs_alpha": float(np.max(np.abs(a))), "ess": metrics.ess(a),
               "eval_imbalance": float(max(eval_imb))}  # fmt: skip
    theta, se = out["arw"]
    theta_t, se_t, eps = out["tmle"]
    arw = {"estimate": theta, "se": se, "status": "ok", **weights, **common}
    tmle = {"estimate": theta_t, "se": se_t, "status": "ok", "epsilon": eps, **weights, **common}
    return arw, tmle


def _eval_imbalance(af, basis, Xt):
    """``max_j |P_fold[alpha_hat phi_j - d phi_j / d d]|`` on the evaluation fold."""
    Phi = np.asarray(basis(Xt), dtype=float)
    M = np.asarray(basis.derivative(Xt, 0), dtype=float)
    return float(np.max(np.abs(np.mean(af(Xt)[:, None] * Phi - M, axis=0))))


def _inference(X, Y, folds):
    def gamma_feats(k, X_tr):
        return feat_gamma, dfeat_gamma

    return _crossfit(X, Y, folds, lambda k: alpha_basis(), gamma_feats, None, 0.0, 0.0)


def _illustration(X, Y, folds, seed):
    """RFF bases: fold k uses the seeds ``(seed, k)`` for the Riesz features and
    ``(seed, K + k)`` for the regression features (both fit on the training folds)."""

    def riesz_basis(k):
        return rff_basis(int(seed) * 100 + int(k))

    def gamma_feats(k, X_tr):
        b = rff_basis(int(seed) * 100 + K + int(k))
        b.fit(X_tr)
        return (lambda rows: np.asarray(b(rows), dtype=float),
                lambda rows: np.asarray(b.derivative(rows, 0), dtype=float))  # fmt: skip

    return _crossfit(X, Y, folds, riesz_basis, gamma_feats, "l2", RIESZ_LAM, RIDGE_LAM)


def _counterexamples(X, Y, n):
    """Oracle perturbations of ``rem:ame_conditions`` (alpha_hat = alpha_0, not fitted;
    the fold structure plays no role): the full-sample score average and its sample SD,
    computed inside the warning recorder and checked for finiteness."""
    a0 = alpha0_q(X)
    d = X[:, 0]
    j = math.ceil(math.sqrt(n))
    c = n**-0.25
    arms = {
        "Cex_i": (gamma0(X) + np.sin(j * d) / j, dgamma0(X) + np.cos(j * d)),
        # the almost-everywhere derivative of the indicator is 0
        "Cex_ii": (gamma0(X) + c * (d > 0), dgamma0(X)),
    }
    out = {}
    for name, (g, dg) in arms.items():

        def score(g=g, dg=dg):
            psi = dg + a0 * (Y - g)
            theta = float(np.mean(psi))
            return psi, theta, float(np.std(psi - theta) / np.sqrt(n))

        (psi, theta, se), counts, converged = baselines.run_recording_warnings(score)
        fields = {"n_warnings": int(sum(counts.values())), "warnings": json.dumps({"other": counts})}
        if not _finite(psi, theta, se):
            out[name] = _failed("nonfinite", **fields)
        elif not converged:
            out[name] = _failed("convergence_warning", **fields)
        else:
            out[name] = {"estimate": theta, "se": se, "status": "ok", **fields}
    return out


def replicate(task):
    """One replication of one cell. ``task = (cell, rep, entropy)``."""
    cell_index, rep, entropy = task
    part, n = CELLS[cell_index]
    seeds = Seeds(EXP, entropy=entropy)
    rng = seeds.data(cell_index, rep)
    if part == "illustration":
        X, Y = draw_l(rng, n)
    else:
        X, Y = draw_q(rng, n)
    if part == "counterexamples":
        res = _counterexamples(X, Y, n)
    else:
        folds = fold_ids(n, K, seeds.folds(cell_index, rep))
        if part == "inference":
            arw, tmle = _inference(X, Y, folds)
        else:
            seed = int(seeds.baseline(cell_index, rep).integers(0, 2**31 - 1) // 1000)
            arw, tmle = _illustration(X, Y, folds, seed)
        res = {"ARW_cf": arw, "TMLE_cf": tmle}
    return {f"{part}|{e}": r for e, r in res.items()}


def tasks_for(entropy, reps=None, cells=None):
    """Tasks of the given cells; ``reps`` overrides the registered R of every part (the
    pilot)."""
    cells = range(len(CELLS)) if cells is None else cells
    return [
        (ci, r, entropy) for ci in cells for r in range(reps or R_PART[CELLS[ci][0]])
    ]


# ---------------------------------------------------------------- aggregation

NUMERIC_FIELDS = (
    "estimate", "se", "max_abs_alpha", "ess", "eval_imbalance", "train_imbalance_max", "epsilon",
    "n_warnings",
)  # fmt: skip
TEXT_FIELDS = ("status", "warnings", "fold_status")


def raw_frame(tasks, results):
    """One row per replication x arm; failures are rows too (§1.7)."""
    import pandas as pd

    rows = []
    for (ci, rep, _), res in zip(tasks, results, strict=True):
        part, n = CELLS[ci]
        for label, r in res.items():
            rows.append(
                {
                    "cell": ci, "part": part, "n": n, "rep": rep, "arm": label,
                    **{k: float(r.get(k, np.nan)) for k in NUMERIC_FIELDS},
                    **{k: str(r.get(k, "")) for k in TEXT_FIELDS},
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


def total_warnings(raw):
    return int(raw["n_warnings"].sum())


def _theta0(part, pop):
    return pop["illustration" if part == "illustration" else "inference"]["theta0"]


def summarise(raw, pop):
    """Per cell x arm: the §1.5 metrics against the Stage 0 ``theta_0``; for the inference
    part the H19-Inf predictions, for the counterexamples the H19-Cex statistics."""
    import pandas as pd

    from .e12 import POSITIVE_VARIANCE, moment_status

    rows = []
    inf, cex = pop["inference"], pop["counterexamples"]
    for ci, (part, n) in enumerate(CELLS):
        for est in ESTIMATORS[part]:
            label = f"{part}|{est}"
            g = raw[(raw["cell"] == ci) & (raw["arm"] == label)].sort_values("rep")
            if g.empty:  # the per-cell aggregation of Stage 0.5
                continue
            ok = (g["status"] == "ok").to_numpy()
            e, se = g["estimate"].to_numpy(), g["se"].to_numpy()
            theta0 = _theta0(part, pop)
            row = {"cell": ci, "part": part, "n": n, "arm": label, "estimator": est}
            row.update(metrics.failure_rate(ok))
            row.update(metrics.coverage(e, se, ok, theta0))
            row["status_counts"] = json.dumps(dict(sorted(collections.Counter(g["status"]).items())))
            row["warnings"] = int(g["n_warnings"].sum())
            x = e[ok]
            ms = row["moment_status"] = moment_status(x)
            r_s = int(ok.sum())
            if r_s >= 1:
                row.update(metrics.ci_length(se, ok))
                if part != "counterexamples":
                    row.update(metrics.max_weight_summary(g["max_abs_alpha"].to_numpy(), ok))
                    row["ess_median"] = float(g.loc[ok, "ess"].median())
                    row["train_imbalance_max"] = float(g.loc[ok, "train_imbalance_max"].max())
                    row["eval_imbalance_median"] = float(g.loc[ok, "eval_imbalance"].median())
                    row["eval_imbalance_max"] = float(g.loc[ok, "eval_imbalance"].max())
            if r_s >= 2:
                var, m4 = metrics.moment_terms(x)
                row.update({"moment_var": var, "moment_m4": m4, "moment_radicand": m4 - var**2})
                row.update(metrics.bias(e, ok, theta0))
                row["rmse"] = metrics.rmse(e, ok, theta0)
                row["root_n_sd"] = float(np.sqrt(n) * x.std(ddof=1))
                row["n_var"] = float(n * x.var(ddof=1))
                row["root_n_bias"] = float(np.sqrt(n) * (x.mean() - theta0))
                row["root_n_bias_se"] = float(np.sqrt(n) * x.std(ddof=1) / np.sqrt(x.size))
            if ms in POSITIVE_VARIANCE:
                row["se_ratio"] = metrics.se_ratio(e, se, ok)
            if ms == "ok":
                row.update(metrics.root_n_sd(e, ok, n))
            row["theta0"] = theta0
            if part == "inference":
                row["certified"] = bool(inf["certified"])
                row["b_pred"] = 0.0
                row["sigma_true"] = inf["sigma_true"]
                row["root_n_sd_pred"] = inf["sigma_true"]
                row["c_pred"] = inf["c_n"][str(n)]
            elif part == "illustration":
                row["root_n_sd_ref"] = math.sqrt(pop["illustration"]["V_star"])
            else:
                row["V_star"] = inf["V_star"]
                if est == "Cex_i" and r_s >= 2:
                    row["Q"] = row["n_var"] / cex["V_star_plus_half"]
                    row["Q_pred"] = 1.0
                    row["Q_exact_n"] = cex["i"][str(n)]["exact_Q"]
                    row["Q_without_perturbation"] = inf["V_star"] / cex["V_star_plus_half"]
                if est == "Cex_ii":
                    row["root_n_bias_pred"] = cex["ii"][str(n)]["mean_root_n_error"]
            rows.append(row)
    return pd.DataFrame(rows)


def family_sentence(verdict, tests, unavailable):
    """The family's conclusion. An unavailable test stays in the Holm family as ``p = 1``
    (so the multiplicity does not depend on the outcome) but is never read as a
    computed non-rejection: a family whose tests are all unavailable is not evaluable,
    and a partly unavailable family says how many tests were computed."""
    hyp = verdict.hypothesis if verdict is not None else "family"
    m, u = len(tests), len(unavailable)
    if verdict is None or m == 0 or u == m:
        return f"{hyp}: not evaluable (no test could be computed)"
    s = verdict.sentence()
    if u:
        s += f"; {u} of {m} tests unavailable (entered as p = 1, not computed)"
    return s


def family_inf(summ, raw):
    """H19-Inf (Holm 0.01): ARW_cf of the inference part at every n: coverage
    ``|c - 0.95| <= 0.02`` (exact binomial), the SD ratio ``|sqrt(n) SD / sqrt(V*) - 1| <=
    0.05`` (fourth-moment delta se) and the bias null with ``b = 0`` (§1.8). TMLE is
    reported without a hypothesis. An undefined test stays in the family as a
    non-rejecting entry (p = 1) and is listed in ``unavailable``."""
    from . import inference
    from .e12 import POSITIVE_VARIANCE, UNAVAILABLE_P

    tests, unavailable = {}, []
    rows = summ[(summ["part"] == "inference") & (summ["estimator"] == "ARW_cf")]
    for _, r in rows.iterrows():
        if not bool(r["certified"]):
            continue
        n = int(r["n"])
        g = raw[(raw["cell"] == r["cell"]) & (raw["arm"] == r["arm"]) & (raw["status"] == "ok")]
        x = g.sort_values("rep")["estimate"].to_numpy()
        tests[f"ARW_cf|n={n}|coverage"] = inference.coverage_pvalue(
            int(r["covered"]), int(r["R"]), float(r["c_pred"])
        )
        for kind in ("sd_ratio", "bias"):
            key = f"ARW_cf|n={n}|{kind}"
            if r["moment_status"] not in POSITIVE_VARIANCE or (
                kind == "sd_ratio" and r["moment_status"] != "ok"
            ):
                tests[key] = UNAVAILABLE_P
                unavailable.append(f"{key} ({r['moment_status']})")
            elif kind == "sd_ratio":
                tests[key] = inference.sd_ratio_pvalue(x, float(r["sigma_true"]), n)
            else:
                tests[key] = inference.bias_pvalue(
                    x, float(r["theta0"]), 0.0, float(r["sigma_true"]), n
                )
    verdict = inference.judge_family("H19-Inf", tests) if tests else None
    return verdict, tests, unavailable


def family_cex(summ, raw, pop):
    """H19-Cex (Holm 0.01): (i) ``|n Var(theta_hat)/(V* + 1/2) - 1| <= 0.06``
    (fourth-moment delta se); (ii) the mean of ``sqrt(n)(theta_hat - theta_0)`` against
    the point ``-n^{1/4} f_D(0)`` (interval null of width 0, z test with se =
    sqrt(n) SD/sqrt(R_s))."""
    from . import inference
    from .e12 import UNAVAILABLE_P

    tests, unavailable = {}, []
    v = pop["inference"]["V_star"]
    for _, r in summ[summ["part"] == "counterexamples"].iterrows():
        n = int(r["n"])
        g = raw[(raw["cell"] == r["cell"]) & (raw["arm"] == r["arm"]) & (raw["status"] == "ok")]
        x = g.sort_values("rep")["estimate"].to_numpy()
        key = f"{r['estimator']}|n={n}"
        if x.size < 2:
            tests[key] = UNAVAILABLE_P
            unavailable.append(f"{key} (fewer_than_two)")
        elif r["estimator"] == "Cex_i":
            tests[key] = inference.variance_ratio_pvalue(x, n, v, CEX_VAR_DELTA)
        else:
            pred = float(r["root_n_bias_pred"])
            tests[key] = inference.interval_null_pvalue(
                float(r["root_n_bias"]), float(r["root_n_bias_se"]), pred, pred
            )
    verdict = inference.judge_family("H19-Cex", tests) if tests else None
    return verdict, tests, unavailable


# ---------------------------------------------------------------- table and figure (summary.csv)

END = " \\\\"


def _fmt(x, d=3):
    if x is None or x != x:
        return "--"
    s = f"{float(x):.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s  # no "-0.000"


def _fail(r):
    """Failure percentage; with no failure, the one-sided 95% Clopper-Pearson upper bound."""
    if r["failure_rate"] == 0 and r.get("failure_rate_cp_upper") == r.get("failure_rate_cp_upper"):
        return f"0.0 ($\\le${100 * float(r['failure_rate_cp_upper']):.2f})"
    return f"{100 * r['failure_rate']:.1f}"


def _statuses(c):
    counts = {k: v for k, v in json.loads(c).items() if k != "ok"}
    return ", ".join(f"{k.replace('_', chr(92) + '_')}: {v}" for k, v in counts.items()) or "--"


def tables(S):
    """``tab_E19`` from the rows of ``summary.csv`` only (§4): the inference and
    illustration parts (one row per estimator and n) and the counterexamples."""
    lines = [
        "\\begin{tabular}{llrrrrrrrl}",
        "\\hline",
        "Part & Estimator & $n$ & Bias & $\\sqrt{n}$SD (ref.) & SE ratio & Coverage (pred.)"
        " & Cov. succ. & Fail \\% & Failures by status" + END,
        "\\hline",
    ]
    for part, title in (("inference", "DGP19Q"), ("illustration", "DGP19L, random features")):
        lines.append("\\multicolumn{10}{l}{" + title + "}" + END)
        for est in ESTIMATORS[part]:
            for n in N_PART[part]:
                r = S[(S["part"] == part) & (S["estimator"] == est) & (S["n"] == n)].iloc[0]
                ref = r.get("root_n_sd_pred") if part == "inference" else r.get("root_n_sd_ref")
                cov = _fmt(r.get("coverage"))
                if part == "inference":
                    cov += f" ({_fmt(r.get('c_pred'))})"
                lines.append(
                    " & ".join(
                        [
                            "", est.replace("_", "\\_"), str(n), _fmt(r.get("bias")),
                            f"{_fmt(r.get('root_n_sd'), 2)} ({_fmt(ref, 2)})",
                            _fmt(r.get("se_ratio"), 2), cov, _fmt(r.get("coverage_conditional")),
                            _fail(r), _statuses(r["status_counts"]),
                        ]
                    )
                    + END
                )  # fmt: skip
    lines += [
        "\\hline",
        "\\multicolumn{10}{l}{Counterexamples (oracle perturbations, DGP19Q)}" + END,
        "Example & Statistic & $n$ & \\multicolumn{2}{r}{Estimate} & \\multicolumn{2}{r}{Prediction}"
        " & \\multicolumn{3}{r}{Exact at $n$}" + END,
        "\\hline",
    ]
    for n in N_PART["counterexamples"]:
        r = S[(S["part"] == "counterexamples") & (S["estimator"] == "Cex_i") & (S["n"] == n)].iloc[0]
        lines.append(
            " & ".join(["(i)", "$n\\,\\mathrm{Var}(\\widehat\\theta)/(V^*+1/2)$", str(n),
                        "\\multicolumn{2}{r}{" + _fmt(r.get("Q")) + "}",
                        "\\multicolumn{2}{r}{" + _fmt(r.get("Q_pred")) + "}",
                        "\\multicolumn{3}{r}{" + _fmt(r.get("Q_exact_n")) + "}"]) + END
        )  # fmt: skip
    for n in N_PART["counterexamples"]:
        r = S[(S["part"] == "counterexamples") & (S["estimator"] == "Cex_ii") & (S["n"] == n)].iloc[0]
        lines.append(
            " & ".join(["(ii)", "mean of $\\sqrt{n}(\\widehat\\theta-\\theta_0)$", str(n),
                        "\\multicolumn{2}{r}{" + _fmt(r.get("root_n_bias")) + "}",
                        "\\multicolumn{2}{r}{" + _fmt(r.get("root_n_bias_pred")) + "}",
                        "\\multicolumn{3}{r}{" + _fmt(r.get("root_n_bias_pred")) + "}"]) + END
        )  # fmt: skip
    lines += ["\\hline", "\\end{tabular}"]
    fams = {k: json.loads(S[k].iloc[0]) for k in ("family_inf", "family_cex") if k in S}
    return {"tab_E19": "\n".join(lines) + "\n"}, fams


def figure(S):
    """``fig_E19_counterexamples``: (left) counterexample (i): ``n Var/(V* + 1/2)`` with a
    95% delta-method interval, the prediction 1 and the value without the perturbation
    ``V*/(V* + 1/2)``; (right) counterexample (ii): the mean of ``sqrt(n)(theta_hat -
    theta_0)`` with a 95% interval against ``-n^{1/4} f_D(0)``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    ns = list(N_PART["counterexamples"])
    xs = np.arange(len(ns))
    ci = S[(S["part"] == "counterexamples") & (S["estimator"] == "Cex_i")].set_index("n").loc[ns]
    se_q = np.sqrt(ci["moment_radicand"].to_numpy(float) / ci["R_s"].to_numpy(float)) * np.array(ns) / (
        ci["V_star"].to_numpy(float) + 0.5
    )
    ax1.errorbar(xs, ci["Q"], yerr=1.96 * se_q, fmt="o", color="#4C72B0", label="estimate")
    ax1.axhline(1.0, color="k", lw=0.8, label="prediction $1$")
    ax1.axhline(float(ci["Q_without_perturbation"].iloc[0]), color="k", lw=0.8, ls="--",
                label="$V^*/(V^*+1/2)$")  # fmt: skip
    ax1.set_xticks(xs, [str(n) for n in ns])
    ax1.set_xlabel("$n$")
    ax1.set_ylabel("$n\\,\\mathrm{Var}(\\hat\\theta)/(V^*+1/2)$")
    ax1.set_title("(i) $\\hat\\gamma=\\gamma_0+j_n^{-1}\\sin(j_nd)$", fontsize=9)
    ax1.legend(fontsize=7)
    cii = S[(S["part"] == "counterexamples") & (S["estimator"] == "Cex_ii")].set_index("n").loc[ns]
    ax2.errorbar(xs, cii["root_n_bias"], yerr=1.96 * cii["root_n_bias_se"], fmt="o",
                 color="#DD8452", label="estimate")  # fmt: skip
    ax2.plot(xs, cii["root_n_bias_pred"], "kx", ms=8, label="$-n^{1/4}f_D(0)$")
    ax2.axhline(0.0, color="k", lw=0.5, ls=":")
    ax2.set_xticks(xs, [str(n) for n in ns])
    ax2.set_xlabel("$n$")
    ax2.set_ylabel("mean of $\\sqrt{n}(\\hat\\theta-\\theta_0)$")
    ax2.set_title("(ii) $\\hat\\gamma=\\gamma_0+n^{-1/4}1[d>0]$", fontsize=9)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    return fig
