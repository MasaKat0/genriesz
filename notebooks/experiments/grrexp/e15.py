"""E-15: the Riesz weighted estimator and undersmoothing (H15-Suf, H15-Bias).

Design: doc/2026-10-07_experiment_registration.md, E-15, §1.3, §1.4 (AutoDML-lasso),
§1.5–§1.9 (parent repository). Under the author's decision U28 (2026-10-08) the
design document is not a binding registration; the choices it leaves open are
listed in the parent repository's ``docs/spec/02_data-and-experiments.md`` §2.1
("E-15 の" rows). The notebook ``13_E15_rw_undersmoothing.ipynb`` runs the stages.

Part A (deterministic): the scalar example of PR4a, ``theta_hat = (1 - lambda) Ybar``,
``Y ~ N(1, 1)``; exact coverage ``Phi(1.96 - tau) - Phi(-1.96 - tau)``,
``tau = sqrt(n) lambda theta0 / (1 - lambda)``; symbolic against floating point.

Part B (Monte Carlo): DGP13 plus 22 irrelevant covariates (25 in all), the
treatment-interaction dictionary ``D(1, z_1..z_25)``, ``(1 - D)(1, z_1..z_25)``
(``p = 52``); SQ-l1 and UKL(C=1)-l1 (offset ``alpha_ref = +-2``) under the rules
L_th (two-step), L_us and L_0; RW_full and ARW_cf; AutoDML-lasso (§1.4) as a
separate ARW row.
"""

from __future__ import annotations

import collections
import itertools
import json
import math

import numpy as np

from . import baselines, metrics, population
from .seeds import Seeds, fold_ids

EXP = 15
THETA0 = 1.0
ETA_SCALE = 0.75
P_COV = 25  # z_1, ..., z_25 (z_4, ..., z_25 irrelevant; z_3 enters neither e nor gamma0)
P = 2 * (P_COV + 1)  # 52
N_VALUES = (500, 1000, 2000, 4000, 8000)
CELLS = [[n] for n in N_VALUES]
R = 2000
K = 5
SQRT3 = math.sqrt(3.0)
GENERATORS = ("SQ", "UKL1")
RULES = ("L_th", "L_us", "L_0")
ESTIMATORS = ("RW_full", "ARW_cf")
GRR_ARMS = [f"{g}-{r}|{e}" for g in GENERATORS for r in RULES for e in ESTIMATORS]
AUTODML_ARM = "AutoDML|ARW_cf"
ARM_LABELS = GRR_ARMS + [AUTODML_ARM]
#: gamma0(d, z) = theta0 d + z_1 + 0.5 z_2 on (D(1, z), (1 - D)(1, z)).
RHO = np.zeros(P)
RHO[[0, 1, 2]] = (THETA0, 1.0, 0.5)
RHO[[P_COV + 2, P_COV + 3]] = (1.0, 0.5)
ALPHA_LEVEL = 0.05  # the 0.05 of the lambda rule sqrt(2 log(2p/0.05)/n)

#: §1.3 C: l1 -> proximal residual <= 1e-6 lambda (genriesz FISTA default);
#: lambda = 0 -> max_j |grad_j| <= 1e-8 max(1, ||P_n m(W, phi)||_inf).
SAMPLE_TOL_L0 = 1e-8
ANEM_TOL = 1e-10  # the identity of §E-15 Part B, checked in every RW_full fit
PART_A_TOL = 1e-12
CERT_GRADIENT = 1e-12
CERT_BETA_DIFF = 1e-10
IRRELEVANT_TOL = 1e-10  # |beta*_j| of the irrelevant coordinates (symmetry), by quadrature
CANCELLATION = 0.25  # |rho' s| below this: no H15-Bias prediction
SUF_TOL = 0.02  # H15-Suf: |c - 0.95| <= 0.02
STD_BIAS_TOL = 0.25  # H15-Bias: |bias/SD - tau_pred| <= 0.25
BIAS_COVERAGE_TOL = 0.03  # H15-Bias: |c - c(n)| <= 0.03
SUF_N = (4000, 8000)
BIAS_N = (2000, 4000, 8000)

# AutoDML-lasso (§1.4, CNS A.1.1 and A.2, frozen constants)
ADML_C = (1.0, 0.1, 0.1)
ADML_OUTER = 10
ADML_OUTER_TOL = 1e-8
ADML_INNER = 10000
ADML_INNER_TOL = 1e-10
ADML_LOADING_SHIFT = 0.2


def gr():
    import genriesz

    return genriesz


def sign(A):
    return np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)


sign.vectorized = True  # one call on all rows (genriesz BregmanGenerator.branch_fn)


def generator(name):
    g = gr()
    return {
        "SQ": lambda: g.SquaredGenerator(C=0.0),
        "UKL1": lambda: g.UKLGenerator(C=1.0, branch_fn=sign),
    }[name]()


def features(Z):
    """phi(z) = (1, z_1, ..., z_k) for the columns given."""
    Z = np.atleast_2d(Z)
    return np.column_stack([np.ones(len(Z)), Z])


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
    """Z ~ Unif(-sqrt3, sqrt3)^25, D ~ Bern(expit eta), Y = gamma0 + N(0, 1) (DGP13 + 22)."""
    Z = rng.uniform(-SQRT3, SQRT3, size=(n, P_COV))
    e = 1.0 / (1.0 + np.exp(-eta(Z)))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = gamma0(D, Z) + rng.standard_normal(n)
    return D, Z, Y


def model(gen_name, lam):
    g = gr()
    gen = generator(gen_name)
    return g.GRRGLM(
        basis=basis(), generator=gen, functional=g.ATEFunctional(0),
        penalty="l1" if lam > 0 else None, lam=float(lam),
        offset=g.offset_from_alpha(gen, alpha_ref),
    )  # fmt: skip


def lam_scale(n_fit):
    """sqrt(2 log(2p/0.05) / n)."""
    return math.sqrt(2.0 * math.log(2.0 * P / ALPHA_LEVEL) / n_fit)


def s_hat(alpha, Phi, M):
    """S_hat(alpha) = max_j {P_n xi_j^2}^{1/2}, xi_j = alpha phi_j - m(W, phi_j)."""
    xi = alpha[:, None] * Phi - M
    return float(np.sqrt(np.max(np.mean(xi**2, axis=0))))


def _finite_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------- Part A


def part_a_rules(n):
    return {
        "L_th": 2.0 * math.sqrt(2.0 * math.log(2.0 / ALPHA_LEVEL) / n),
        "L_us": n**-0.5 / math.log(n),
        "L_0": 0.0,
    }


PART_A_N = (10**2, 10**3, 10**4, 10**5, 10**6)


def part_a():
    """Exact coverage of the scalar example, symbolic (sympy, 50 digits) and float."""
    import sympy as sp
    from scipy import stats

    rows = []
    for n in PART_A_N:
        for rule in RULES:
            # the rules in exact arithmetic
            ns = sp.Integer(n)
            lam_s = {
                "L_th": 2 * sp.sqrt(2 * sp.log(sp.Rational(2) / sp.Rational(1, 20)) / ns),
                "L_us": 1 / (sp.sqrt(ns) * sp.log(ns)),
                "L_0": sp.Integer(0),
            }[rule]
            tau_s = sp.sqrt(ns) * lam_s * THETA0 / (1 - lam_s)
            z = sp.Rational(196, 100)
            cdf = lambda x: (1 + sp.erf(x / sp.sqrt(2))) / 2  # noqa: E731
            c_s = cdf(z - tau_s) - cdf(-z - tau_s)
            lam_f = part_a_rules(n)[rule]
            tau_f = math.sqrt(n) * lam_f * THETA0 / (1.0 - lam_f)
            c_f = float(stats.norm.cdf(1.96 - tau_f) - stats.norm.cdf(-1.96 - tau_f))
            c_exact = float(sp.N(c_s, 50))
            rows.append(
                {
                    "n": n, "rule": rule, "lambda": lam_f, "tau": tau_f, "coverage": c_f,
                    "coverage_symbolic": c_exact, "abs_diff": abs(c_f - c_exact),
                    "lambda_abs_diff": abs(lam_f - float(sp.N(lam_s, 50))),
                    "tau_abs_diff": abs(tau_f - float(sp.N(tau_s, 50))),
                }
            )  # fmt: skip
    ok = all(
        r["abs_diff"] <= PART_A_TOL
        and r["lambda_abs_diff"] <= PART_A_TOL
        and r["tau_abs_diff"] <= PART_A_TOL * max(1.0, abs(r["tau"]))
        for r in rows
    )
    return rows, ok


# ---------------------------------------------------------------- Stage 0 (Part B population)


def quadrature(points, dims=3):
    """Rows (1, z) and (0, z) on a tensor Gauss-Legendre grid over (z_1..z_dims) with
    P(D, Z); the remaining covariates do not enter e or gamma0."""
    x, w = np.polynomial.legendre.leggauss(points)
    z = SQRT3 * x
    wz = w / 2.0
    Zq = np.array(list(itertools.product(*([z] * dims))))
    Wq = np.prod(np.array(list(itertools.product(*([wz] * dims)))), axis=1)
    e = 1.0 / (1.0 + np.exp(-eta(Zq)))
    X = np.vstack(
        [np.column_stack([np.ones(len(Zq)), Zq]), np.column_stack([np.zeros(len(Zq)), Zq])]
    )
    wts = np.concatenate([Wq * e, Wq * (1.0 - e)])
    return X, wts


def population_fit(gen_name, points, dims=3):
    """The population lambda = 0 minimizer on (1, z_1..z_dims) per arm."""
    g = gr()
    gen = generator(gen_name)
    X, w = quadrature(points, dims)
    bas = basis()
    bas.fit(X)
    Phi = np.asarray(bas(X), dtype=float)
    M = np.asarray(g.ATEFunctional(0).m_basis_matrix(X, bas), dtype=float)
    off = np.asarray(gen.grad(X, alpha_ref(X)), dtype=float)
    sol = population.solve(generator=gen, X=X, w=w, Phi=Phi, M=M, offset=off)
    return sol, X, w, Phi, M, off, gen


def _pick(beta, dims):
    """The coefficients of (1, z_1, z_2) per arm from a fit on (1, z_1..z_dims)."""
    k = dims + 1
    return np.array([beta[0], beta[1], beta[2], beta[k], beta[k + 1], beta[k + 2]])


def stage0():
    """Part A and the H15-Bias predictions of both generators (§1.9 certificate at
    48 and 64 points; the irrelevant coordinates vanish, checked with z_3 and z_4 on a
    four-dimensional grid). JSON-compatible."""
    rows_a, ok_a = part_a()
    gens = []
    for gen_name in GENERATORS:
        s48 = population_fit(gen_name, 48)[0]
        s64, X, w, Phi, M, off, gen = population_fit(gen_name, 64)
        solved = s48.status == "ok" and s64.status == "ok"
        beta_diff = float(np.max(np.abs(s48.beta - s64.beta))) if solved else None
        s4 = population_fit(gen_name, 24, dims=4)[0]
        # irrelevant coordinates: z_3 (index 3 per arm, 3-dim fit) and z_3, z_4 (4-dim fit)
        irr3 = [s64.beta[3], s64.beta[7]]
        irr4 = [s4.beta[3], s4.beta[4], s4.beta[8], s4.beta[9]]
        checks = {
            "i_gradient": bool(
                solved and s48.max_gradient <= CERT_GRADIENT and s64.max_gradient <= CERT_GRADIENT
            ),
            "i_beta_diff": bool(solved and beta_diff <= CERT_BETA_DIFF),
            "ii_iv_positive_definite": bool(solved and s64.min_hessian_eig > 0),
            "iii_margin": bool(solved and s64.min_dual_margin >= 1e-4),
            "irrelevant_zero": bool(
                s4.status == "ok"
                and max(abs(float(v)) for v in irr3 + irr4) <= IRRELEVANT_TOL
            ),
        }
        cert = all(checks.values())
        row = {
            "generator": gen_name,
            "certified": cert,
            "checks": checks,
            "status_48": s48.status,
            "status_64": s64.status,
            "status_4d": s4.status,
            "beta_diff_48_64": beta_diff,
            "max_gradient_64": _finite_or_none(s64.max_gradient),
            "min_hessian_eig_64": _finite_or_none(s64.min_hessian_eig),
            "min_dual_margin_64": _finite_or_none(s64.min_dual_margin),
            "beta": [_finite_or_none(b) for b in s64.beta],
            "irrelevant_beta_3d": [_finite_or_none(v) for v in irr3],
            "irrelevant_beta_4d": [_finite_or_none(v) for v in irr4],
        }
        if solved:
            a = s64.alpha
            xi = a[:, None] * Phi - M
            # E xi_j^2 for (1, z_1, z_2, z_3) per arm; an irrelevant z_k (k >= 3)
            # equals the intercept of its arm (independent, unit variance)
            e_xi2 = w @ xi**2
            s_star = float(np.sqrt(np.max(e_xi2)))
            sigma = float(np.sqrt(w @ a**2))  # Var(eps) = 1, m(W, gamma0) = theta0
            b6 = _pick(s64.beta, 3)  # (int, z1, z2) treated; (int, z1, z2) control
            signs = np.sign(b6)
            rho6 = np.array([THETA0, 1.0, 0.5, 0.0, 1.0, 0.5])
            rho_s = float(rho6 @ signs)
            cancel = abs(rho_s) < CANCELLATION
            tau = -2.0 * s_star * math.sqrt(2.0 * math.log(2.0 * P / ALPHA_LEVEL)) * rho_s / sigma
            from scipy import stats

            c_pred = float(stats.norm.cdf(1.96 - tau) - stats.norm.cdf(-1.96 - tau))
            row.update(
                {
                    "E_xi2": [float(v) for v in e_xi2],
                    "S_star": s_star,
                    "sigma": sigma,
                    "E_alpha2": float(w @ a**2),
                    "beta_rho_coords": [float(v) for v in b6],
                    "min_abs_beta_rho_coords": float(np.min(np.abs(b6[[0, 1, 2, 4, 5]]))),
                    "signs": [int(v) for v in signs],
                    "rho_s": rho_s,
                    "cancellation": bool(cancel),
                    "tau_pred": None if (cancel or not cert) else tau,
                    "c_pred": None if (cancel or not cert) else c_pred,
                    "tau_unconditional": tau,
                }
            )
        gens.append(row)
    return {"part_a": rows_a, "part_a_ok": ok_a, "generators": gens}


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


class _Warn:
    """Warnings of the individual fits of one record (§1.1)."""

    def __init__(self):
        self.fits, self.converged, self.last = [], True, True

    def run(self, tag, fn):
        result, counts, converged = baselines.run_recording_warnings(fn)
        if counts:
            self.fits.append([tag, counts])
        self.last = converged
        self.converged = self.converged and converged
        return result

    def fields(self, other=None):
        other = other or {}
        n = sum(sum(c.values()) for _, c in self.fits) + sum(other.values())
        return {"n_warnings": int(n), "warnings": json.dumps({"fits": self.fits, "other": other})}


def _fit(gen_name, lam, X_fit, beta0=None):
    mdl = model(gen_name, lam)
    fr = mdl.fit(X_fit, beta0=beta0, tol=None if lam > 0 else SAMPLE_TOL_L0)
    return str(fr.status), mdl


def fit_rules(gen_name, X_fit, warn, tag):
    """The three rules on one fitting sample. Returns ``{rule: (status, model, lam)}``.

    L_0: lambda = 0. L_th: lambda_1 = 2 S_hat(alpha_ref) c, then lambda_th =
    2 S_hat(alpha_hat^(1)) c, ``c = sqrt(2 log(2p/0.05)/n_fit)``. L_us: lambda_us =
    S_hat(alpha_hat^(1)) n_fit^{-1/2}/log n_fit. A failed first step fails L_th and
    L_us with its status. L_th and L_us start from the first-step coefficients.
    """
    g = gr()
    n_fit = len(X_fit)
    bas = basis()
    bas.fit(X_fit)
    Phi = np.asarray(bas(X_fit), dtype=float)
    M = np.asarray(g.ATEFunctional(0).m_basis_matrix(X_fit, bas), dtype=float)
    out = {}

    def one(rule, lam, beta0=None):
        status, mdl = warn.run(f"{tag}|{rule}", lambda: _fit(gen_name, lam, X_fit, beta0))
        if status == "ok" and not warn.last:
            status = "convergence_warning"
        return status, mdl

    st0, m0 = one("L_0", 0.0)
    out["L_0"] = (st0, m0, 0.0)
    lam1 = 2.0 * s_hat(alpha_ref(X_fit), Phi, M) * lam_scale(n_fit)
    st1, m1 = one("step1", lam1)
    if st1 != "ok":
        out["L_th"] = (st1, None, float("nan"))
        out["L_us"] = (st1, None, float("nan"))
        return out
    s1 = s_hat(np.asarray(m1.predict_alpha(X_fit), dtype=float), Phi, M)
    lam_th = 2.0 * s1 * lam_scale(n_fit)
    lam_us = s1 * n_fit**-0.5 / math.log(n_fit)
    st, m = one("L_th", lam_th, m1.beta_)
    out["L_th"] = (st, m, lam_th)
    st, m = one("L_us", lam_us, m1.beta_)
    out["L_us"] = (st, m, lam_us)
    return out


def _rw_record(X, Y, status, mdl, lam, warn_fields, detail):
    if status != "ok":
        return _failed(status, lam=lam, fold_status=detail, **warn_fields)
    g = gr()
    a = np.asarray(mdl.predict_alpha(X), dtype=float)
    theta = float(np.mean(a * Y))
    Phi = np.asarray(mdl.basis(X), dtype=float)
    M = np.asarray(g.ATEFunctional(0).m_basis_matrix(X, mdl.basis), dtype=float)
    delta = np.mean(a[:, None] * Phi - M, axis=0)
    D, Z = X[:, 0], X[:, 1:]
    g0 = gamma0(D, Z)
    m_g0 = gamma0(np.ones(len(D)), Z) - gamma0(np.zeros(len(D)), Z)
    lhs = theta - float(np.mean(m_g0 + a * (Y - g0)))
    anem_gap = abs(lhs - float(RHO @ delta))
    pred = _ols(X, Y, X, Y)
    psi = _m_ols(pred, X) + a * (Y - pred(X)) - theta
    return {
        "estimate": theta,
        "se": float(np.sqrt(np.var(psi, ddof=1) / len(Y))),  # §1.4 RW_full
        "status": "ok",
        "lam": lam,
        "nnz": int(np.sum(mdl.beta_ != 0)),
        "max_abs_alpha": float(np.max(np.abs(a))),
        "ess": metrics.ess(a),
        "anem_gap": anem_gap,
        "rho_delta": float(RHO @ delta),
        "fold_status": detail,
        **warn_fields,
    }


def _rw_full(X, Y):
    """Every GRR arm on the full sample (one fit per rule and generator)."""
    out = {}
    for gen_name in GENERATORS:
        warn = _Warn()
        fits = fit_rules(gen_name, X, warn, "full")
        wf = warn.fields()
        for rule in RULES:
            status, mdl, lam = fits[rule]
            detail = json.dumps([["full", rule, status]])
            out[f"{gen_name}-{rule}|RW_full"] = _rw_record(X, Y, status, mdl, lam, wf, detail)
    return out


def _arw_cf(X, Y, folds):
    """ARW_cf (§1.4) of every GRR arm: per fold the three rules fit on the complement
    (the rules are computed on the training fold), the OLS outcome regression on the
    complement, the score on the fold. A failed fit or prediction fails that arm only;
    a warning of the outcome regression fails every arm of the replication."""
    n = len(Y)
    g = gr()
    arms = {f"{gen}-{rule}": {"psi": np.empty(n), "alpha": np.empty(n), "detail": [],
                              "status": "ok", "lam": [], "eval_imb": []}
            for gen in GENERATORS for rule in RULES}  # fmt: skip
    warn = {gen: _Warn() for gen in GENERATORS}
    warn_out = _Warn()
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        Xt = X[te]
        pred = warn_out.run(f"{k}|outcome", lambda tr=tr: _ols(X, Y, X[tr], Y[tr]))
        D_tr = X[tr, 0]
        for gen_name in GENERATORS:
            if not (np.any(D_tr == 1) and np.any(D_tr == 0)):
                fits = {r: ("degenerate_functional", None, float("nan")) for r in RULES}
            else:
                fits = fit_rules(gen_name, X[tr], warn[gen_name], str(int(k)))
            for rule in RULES:
                arm = arms[f"{gen_name}-{rule}"]
                if arm["status"] != "ok":
                    continue
                status, mdl, lam = fits[rule]
                arm["detail"].append([str(int(k)), rule, status])
                if status != "ok":
                    arm["status"] = status
                    continue
                pts = np.vstack([Xt, g.ATEFunctional(0).evaluation_points(Xt, mdl.basis)])
                _, outside, nonfinite = mdl.classify(pts)
                if np.any(outside) or np.any(nonfinite):
                    st = "domain_prediction" if np.any(outside) else "nonfinite"
                    arm["detail"].append([str(int(k)), "prediction", st])
                    arm["status"] = st
                    continue
                a = np.asarray(mdl.classify(Xt)[0], dtype=float)
                Phi_t = np.asarray(mdl.basis(Xt), dtype=float)
                M_t = np.asarray(g.ATEFunctional(0).m_basis_matrix(Xt, mdl.basis), dtype=float)
                arm["eval_imb"].append(float(np.max(np.abs(np.mean(a[:, None] * Phi_t - M_t, 0)))))
                arm["psi"][te] = _m_ols(pred, Xt) + a * (Y[te] - pred(Xt))
                arm["alpha"][te] = a
                arm["lam"].append(lam)
    out = {}
    for name, arm in arms.items():
        gen_name = name.split("-")[0]
        label = f"{name}|ARW_cf"
        # the outcome warnings are recorded once, with the SQ arms
        other = warn_out.fields()["warnings"] if gen_name == "SQ" else None
        wf = warn[gen_name].fields()
        if other is not None:
            extra = json.loads(other)
            wf = {"n_warnings": wf["n_warnings"] + sum(sum(c.values()) for _, c in extra["fits"]),
                  "warnings": json.dumps({"fits": json.loads(wf["warnings"])["fits"],
                                          "outcome": extra["fits"]})}  # fmt: skip
        common = {"fold_status": json.dumps(arm["detail"]), **wf}
        status = arm["status"]
        if status == "ok" and not warn_out.converged:
            status = "convergence_warning"
        if status == "ok" and not np.all(np.isfinite(arm["psi"])):
            status = "nonfinite"
        if status != "ok":
            out[label] = _failed(status, **common)
            continue
        theta = float(np.mean(arm["psi"]))
        out[label] = {
            "estimate": theta,
            "se": float(np.std(arm["psi"] - theta) / np.sqrt(n)),
            "status": "ok",
            "lam": float(np.mean(arm["lam"])),
            "max_abs_alpha": float(np.max(np.abs(arm["alpha"]))),
            "ess": metrics.ess(arm["alpha"]),
            "eval_imbalance": float(max(arm["eval_imb"])),
            **common,
        }
    return out


# ---------------------------------------------------------------- AutoDML-lasso (§1.4)


def adml_dictionary(X):
    """b = (1, D, z_1..z_25, D z_1..D z_25) and m(W, b) (ATE)."""
    D, Z = X[:, 0], X[:, 1:]
    n = len(D)
    B = np.column_stack([np.ones(n), D, Z, D[:, None] * Z])
    Mb = np.column_stack([np.zeros(n), np.ones(n), np.zeros_like(Z), Z])
    return B, Mb


def _soft(x, t):
    return math.copysign(max(abs(x) - t, 0.0), x)


def coordinate_descent(G, Mh, thr, rho0):
    """CNS A.2: minimize ``rho'G rho - 2 rho'M + 2 sum_j thr_j |rho_j|`` by cyclic
    soft-thresholding from ``rho0``; stop when a sweep changes no coefficient by more
    than 1e-10, or after 10000 sweeps. Returns ``(rho, sweeps, converged)``."""
    rho = np.array(rho0, dtype=float)
    diag = np.diag(G)
    g_rho = G @ rho
    change = np.inf
    for sweep in range(1, ADML_INNER + 1):
        change = 0.0
        for j in range(len(rho)):
            zj = Mh[j] - (g_rho[j] - diag[j] * rho[j])
            new = _soft(zj, thr[j]) / diag[j]
            d = new - rho[j]
            if d != 0.0:
                g_rho += G[:, j] * d
                rho[j] = new
                change = max(change, abs(d))
        if change <= ADML_INNER_TOL:
            return rho, sweep, True
    return rho, ADML_INNER, False


def adml_fit(X_fit):
    """CNS A.1.1 / A.2 on one training sample. Returns ``(status, rho, info)``."""
    B, Mb = adml_dictionary(X_fit)
    n, p = B.shape
    G = B.T @ B / n
    Mh = Mb.mean(axis=0)
    low = max(2, math.ceil(p / 40))
    rho = np.zeros(p)
    try:
        rho[:low] = np.linalg.solve(G[:low, :low], Mh[:low])
    except np.linalg.LinAlgError:
        return "singular", None, {}
    diag = np.diag(G).copy()
    if np.any(diag <= 0):
        return "degenerate_functional", None, {}
    c1, c2, c3 = ADML_C
    from scipy import stats

    r_l = c1 / math.sqrt(n) * float(stats.norm.ppf(1.0 - c2 / (2.0 * p)))
    outer, inner_capped, sweeps_total = 0, 0, 0
    for outer in range(1, ADML_OUTER + 1):
        resid = B @ rho
        Dj = np.sqrt(np.mean((B * resid[:, None] - Mb) ** 2, axis=0)) + ADML_LOADING_SHIFT
        thr = r_l * Dj
        thr[0] = r_l * c3 * Dj[0]
        old = rho.copy()
        rho, sweep, converged = coordinate_descent(G, Mh, thr, rho)
        sweeps_total += sweep
        inner_capped += int(not converged)
        if not np.all(np.isfinite(rho)):
            return "nonfinite", None, {}
        if np.max(np.abs(rho - old)) <= ADML_OUTER_TOL:
            break
    capped = bool(np.max(np.abs(rho - old)) > ADML_OUTER_TOL)
    return "ok", rho, {"outer": outer, "outer_capped": capped, "inner_capped": inner_capped,
                       "sweeps": sweeps_total}  # fmt: skip


def _autodml(X, Y, folds):
    n = len(Y)
    psi, alpha = np.empty(n), np.empty(n)
    detail, outer_capped, inner_capped = [], 0, 0
    warn = _Warn()
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        D_tr = X[tr, 0]
        if not (np.any(D_tr == 1) and np.any(D_tr == 0)):
            detail.append([str(int(k)), "riesz", "degenerate_functional"])
            return _failed("degenerate_functional", fold_status=json.dumps(detail), **warn.fields())
        status, rho, info = warn.run(f"{k}|autodml", lambda tr=tr: adml_fit(X[tr]))
        detail.append([str(int(k)), "riesz", status])
        if status != "ok":
            return _failed(status, fold_status=json.dumps(detail), **warn.fields())
        outer_capped += int(info["outer_capped"])
        inner_capped += int(info["inner_capped"])
        Bt, _ = adml_dictionary(X[te])
        a = Bt @ rho
        pred = warn.run(f"{k}|outcome", lambda tr=tr: _ols(X, Y, X[tr], Y[tr]))
        psi[te] = _m_ols(pred, X[te]) + a * (Y[te] - pred(X[te]))
        alpha[te] = a
    common = {"fold_status": json.dumps(detail), "adml_outer_capped": outer_capped,
              "adml_inner_capped": inner_capped, **warn.fields()}  # fmt: skip
    if not warn.converged:
        return _failed("convergence_warning", **common)
    if not np.all(np.isfinite(psi)):
        return _failed("nonfinite", **common)
    theta = float(np.mean(psi))
    return {
        "estimate": theta,
        "se": float(np.std(psi - theta) / np.sqrt(n)),
        "status": "ok",
        "max_abs_alpha": float(np.max(np.abs(alpha))),
        "ess": metrics.ess(alpha),
        **common,
    }


def replicate(task):
    """One replication of one cell: every arm. ``task = (cell, rep, entropy)``."""
    cell_index, rep, entropy = task
    (n,) = CELLS[cell_index]
    seeds = Seeds(EXP, entropy=entropy)
    D, Z, Y = draw(seeds.data(cell_index, rep), n)
    X = np.column_stack([D, Z])
    folds = fold_ids(n, K, seeds.folds(cell_index, rep))
    out = _rw_full(X, Y)
    out.update(_arw_cf(X, Y, folds))
    out[AUTODML_ARM] = _autodml(X, Y, folds)
    return out


def tasks_for(entropy, reps, cells=None):
    cells = range(len(CELLS)) if cells is None else cells
    return [(ci, r, entropy) for ci in cells for r in range(reps)]


# ---------------------------------------------------------------- aggregation

NUMERIC_FIELDS = (
    "estimate", "se", "lam", "nnz", "max_abs_alpha", "ess", "anem_gap", "rho_delta",
    "eval_imbalance", "adml_outer_capped", "adml_inner_capped", "n_warnings",
)  # fmt: skip
TEXT_FIELDS = ("status", "warnings", "fold_status")


def raw_frame(tasks, results):
    """One row per replication x arm; failures are rows too (§1.7)."""
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
    """Warnings of a replication: the fits of one generator and estimator are shared by
    the three rules and recorded in each of their rows, so each (cell, rep, generator,
    estimator) group is counted once."""
    est = raw["arm"].str.split("|").str[-1]
    gen = raw["arm"].str.split("|").str[0].str.split("-").str[0]
    first = raw.assign(grp=gen + "|" + est).drop_duplicates(["cell", "rep", "grp"])
    return int(first["n_warnings"].sum())


def anem_check(raw):
    """The identity of Part B in every successful RW_full fit; a violation raises."""
    rw = raw[raw["arm"].str.endswith("RW_full") & (raw["status"] == "ok")]
    gap = float(rw["anem_gap"].max()) if len(rw) else None
    out = {"fits_checked": int(len(rw)), "max_gap": gap, "tolerance": ANEM_TOL}
    if len(rw) and not gap <= ANEM_TOL:
        raise AssertionError(f"ANEM identity violated: {json.dumps(out)}")
    return out


def summarise(raw, stage0_json):
    """§1.5 metrics per cell x arm, with the H15-Bias predictions of L_th RW_full."""
    import pandas as pd

    from .e12 import POSITIVE_VARIANCE, moment_status

    pred = {g["generator"]: g for g in stage0_json["generators"]}
    rows = []
    for ci, (n,) in enumerate(CELLS):
        for label in ARM_LABELS:
            g = raw[(raw["cell"] == ci) & (raw["arm"] == label)].sort_values("rep")
            if g.empty:
                continue
            ok = (g["status"] == "ok").to_numpy()
            est, se = g["estimate"].to_numpy(), g["se"].to_numpy()
            row = {"cell": ci, "n": n, "arm": label}
            row.update(metrics.failure_rate(ok))
            row.update(metrics.coverage(est, se, ok, THETA0))
            row["status_counts"] = json.dumps(dict(sorted(collections.Counter(g["status"]).items())))
            row["warnings"] = int(g["n_warnings"].sum())
            x = est[ok]
            ms = row["moment_status"] = moment_status(x)
            if ok.sum() >= 1:
                row.update(metrics.ci_length(se, ok))
                row.update(metrics.max_weight_summary(g["max_abs_alpha"].to_numpy(), ok))
                row["ess_median"] = float(g.loc[ok, "ess"].median())
                if label != AUTODML_ARM:
                    row["lam_median"] = float(g.loc[ok, "lam"].median())
                if label.endswith("RW_full"):
                    row["nnz_median"] = float(g.loc[ok, "nnz"].median())
                    row["anem_gap_max"] = float(g.loc[ok, "anem_gap"].max())
                if label == AUTODML_ARM:
                    row["adml_outer_capped"] = int(g["adml_outer_capped"].fillna(0).sum())
                    row["adml_inner_capped"] = int(g["adml_inner_capped"].fillna(0).sum())
            if ok.sum() >= 2:
                var, m4 = metrics.moment_terms(x)
                row.update({"moment_var": var, "moment_m4": m4, "moment_radicand": m4 - var**2})
                row.update(metrics.bias(est, ok, THETA0))
                row["rmse"] = metrics.rmse(est, ok, THETA0)
                row["root_n_sd"] = float(np.sqrt(n) * x.std(ddof=1))
                if x.std(ddof=1) > 0:
                    row["std_bias"] = float((x.mean() - THETA0) / x.std(ddof=1))
            if ms in POSITIVE_VARIANCE:
                row["se_ratio"] = metrics.se_ratio(est, se, ok)
            if ms == "ok":
                row.update(metrics.root_n_sd(est, ok, n))
            gen_name = label.split("-")[0]
            if label.endswith("L_th|RW_full") and pred[gen_name]["tau_pred"] is not None:
                row["tau_pred"] = pred[gen_name]["tau_pred"]
                row["c_pred"] = pred[gen_name]["c_pred"]
            rows.append(row)
    return pd.DataFrame(rows)


def suf_arms():
    return [f"{g}-{r}" for g in GENERATORS for r in ("L_us|RW_full", "L_0|RW_full",
                                                       "L_th|ARW_cf")] + [AUTODML_ARM]  # fmt: skip


def families(summ, raw, seeds):
    """H15-Suf and H15-Bias (Holm 0.01 each, §1.8).

    H15-Suf: n in {4000, 8000}, coverage ``|c - 0.95| <= 0.02`` of L_us and L_0 RW_full,
    L_th ARW_cf (both generators) and AutoDML ARW_cf. H15-Bias (generators without
    cancellation): L_th RW_full, n in {2000, 4000, 8000}, the standardized bias
    ``|bias/SD - tau_pred| <= 0.25`` (se from family 2, B = 10000: generators in order,
    then n ascending, the successful replications resampled) and the coverage
    ``|c - c(n)| <= 0.03``. An undefined test stays as a non-rejecting entry (p = 1).
    """
    from . import inference
    from .bootstrap import FamilyBootstrap, bootstrap_se
    from .e12 import UNAVAILABLE_P

    suf = {}
    for arm in suf_arms():
        for n in SUF_N:
            r = summ[(summ["arm"] == arm) & (summ["n"] == n)].iloc[0]
            suf[f"{arm}|n={n}|coverage"] = inference.coverage_pvalue(
                int(r["covered"]), int(r["R"]), 0.95, tol=SUF_TOL
            )
    bias_tests, unavailable = {}, []
    fb = FamilyBootstrap(seeds, 2)
    for gen_name in GENERATORS:
        arm = f"{gen_name}-L_th|RW_full"
        for n in BIAS_N:
            r = summ[(summ["arm"] == arm) & (summ["n"] == n)].iloc[0]
            if not np.isfinite(r.get("tau_pred", np.nan)):
                continue
            g = raw[(raw["cell"] == r["cell"]) & (raw["arm"] == arm) & (raw["status"] == "ok")]
            x = g.sort_values("rep")["estimate"].to_numpy()
            key = f"{arm}|n={n}|std_bias"
            if x.size < 2 or not x.std(ddof=1) > 0:
                bias_tests[key] = UNAVAILABLE_P
                unavailable.append(f"{key} (fewer_than_two_or_zero_sd)")
            else:

                def stat(idx, v=x):
                    s = v[idx]
                    sd = s.std(axis=1, ddof=1)
                    return np.divide(s.mean(axis=1) - THETA0, sd, out=np.full(sd.shape, np.nan),
                                     where=sd > 0)  # fmt: skip

                boot = fb.replicate(stat, x.size, finite=False)
                if not np.all(np.isfinite(boot)):
                    bias_tests[key] = UNAVAILABLE_P
                    unavailable.append(f"{key} (zero_sd_resample)")
                else:
                    t = float((x.mean() - THETA0) / x.std(ddof=1))
                    tau = float(r["tau_pred"])
                    bias_tests[key] = inference.interval_null_pvalue(
                        t, bootstrap_se(boot, 2), tau - STD_BIAS_TOL, tau + STD_BIAS_TOL
                    )
            bias_tests[f"{arm}|n={n}|coverage"] = inference.coverage_pvalue(
                int(r["covered"]), int(r["R"]), float(r["c_pred"]), tol=BIAS_COVERAGE_TOL
            )
    v_suf = inference.judge_family("H15-Suf", suf)
    v_bias = inference.judge_family("H15-Bias", bias_tests) if bias_tests else None
    return {"H15-Suf": v_suf, "H15-Bias": v_bias}, {"H15-Suf": suf, "H15-Bias": bias_tests}, unavailable


# ---------------------------------------------------------------- tables and figure

LABEL = {"SQ": "SQ-$\\ell_1$", "UKL1": "UKL($C=1$)-$\\ell_1$"}
RULE_LABEL = {"L_th": "$\\lambda_{\\mathrm{th}}$", "L_us": "$\\lambda_{\\mathrm{us}}$",
              "L_0": "$\\lambda=0$"}  # fmt: skip
END = " \\\\"


def _fmt(x, d=3):
    if x is None or x != x:
        return "--"
    s = f"{float(x):.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def _arm_label(arm):
    if arm == AUTODML_ARM:
        return "AutoDML-lasso", "ARW\\_cf"
    name, est = arm.split("|")
    gen_name, rule = name.split("-")
    return f"{LABEL[gen_name]}, {RULE_LABEL[rule]}", est.replace("_", "\\_")


def table_exact(part_a_rows):
    """``tab_E15_exact``: Part A, lambda, tau and the exact coverage per n and rule."""
    top = " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{RULE_LABEL[r]}}}" for r in RULES)
    sub = "$n$" + " & $\\lambda$ & $\\tau$ & $c$" * len(RULES)
    lines = ["\\begin{tabular}{r" + "rrr" * len(RULES) + "}", "\\hline", top + END,
             sub + END, "\\hline"]  # fmt: skip
    by = {(r["n"], r["rule"]): r for r in part_a_rows}
    for n in PART_A_N:
        cells = [f"$10^{{{int(round(math.log10(n)))}}}$"]
        for rule in RULES:
            r = by[(n, rule)]
            cells += [_fmt(r["lambda"], 4), _fmt(r["tau"], 3), _fmt(r["coverage"], 3)]
        lines.append(" & ".join(cells) + END)
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_mc(S):
    """``tab_E15_mc``: a longtable body (head repeated) of every arm and n."""
    heads = ["Arm", "Est.", "$n$", "Bias", "Bias/SD ($\\tau$)", "$\\sqrt n$SD", "SE ratio",
             "Coverage ($c$)", "Cov. succ.", "Fail \\%", "med. $\\lambda$"]  # fmt: skip
    head = ["\\hline", " & ".join(heads) + END, "\\hline"]
    lines = ["\\begin{tabular}{llrrrrrrrrr}", *head, "\\endfirsthead", *head, "\\endhead"]
    for arm in ARM_LABELS:
        for n in N_VALUES:
            r = S[(S["arm"] == arm) & (S["n"] == n)].iloc[0]
            name, est = _arm_label(arm)
            target = r.get("c_pred")
            lines.append(
                " & ".join(
                    [
                        name, est, str(n), _fmt(r.get("bias")),
                        f"{_fmt(r.get('std_bias'), 2)} ({_fmt(r.get('tau_pred'), 2)})",
                        _fmt(r.get("root_n_sd"), 2), _fmt(r.get("se_ratio"), 2),
                        f"{_fmt(r.get('coverage'))} ({_fmt(target)})",
                        _fmt(r.get("coverage_conditional")),
                        f"{100 * r['failure_rate']:.1f}",
                        _fmt(r.get("lam_median"), 4),
                    ]
                )
                + END
            )  # fmt: skip
    lines += ["\\hline", "\\end{tabular}"]
    status = ["\\begin{tabular}{llrrl}", "\\hline",
              "Arm & Est. & $n$ & Successes & Statuses" + END, "\\hline"]  # fmt: skip
    for arm in ARM_LABELS:
        for n in N_VALUES:
            r = S[(S["arm"] == arm) & (S["n"] == n)].iloc[0]
            if int(r["R_s"]) == int(r["R"]):
                continue
            name, est = _arm_label(arm)
            counts = ", ".join(
                f"{k.replace('_', chr(92) + '_')}: {v}"
                for k, v in json.loads(r["status_counts"]).items()
            )
            status.append(" & ".join([name, est, str(n), str(int(r["R_s"])), counts]) + END)
    status += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n", "\n".join(status) + "\n"


def tables(S, part_a_rows):
    mc, status = table_mc(S)
    return {"tab_E15_exact": table_exact(part_a_rows), "tab_E15_mc": mc, "tab_E15_status": status}


def figure(S):
    """``fig_E15_coverage``: unconditional coverage against n, one panel per generator."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    styles = {"L_th|RW_full": ("o", "-"), "L_us|RW_full": ("s", "-"), "L_0|RW_full": ("^", "-"),
              "L_th|ARW_cf": ("o", "--")}  # fmt: skip
    for ax, gen_name in zip(axes, GENERATORS, strict=True):
        for key, (mk, ls) in styles.items():
            sub = S[S["arm"] == f"{gen_name}-{key}"].sort_values("n")
            rule, est = key.split("|")
            ax.plot(sub["n"], sub["coverage"], marker=mk, ls=ls, label=f"{rule}, {est}")
        sub = S[S["arm"] == AUTODML_ARM].sort_values("n")
        ax.plot(sub["n"], sub["coverage"], marker="x", ls=":", color="k", label="AutoDML, ARW_cf")
        ax.axhline(0.95, color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(list(N_VALUES), [str(n) for n in N_VALUES])
        ax.set_xlabel("$n$")
        ax.set_title({"SQ": "SQ-$\\ell_1$", "UKL1": "UKL($C=1$)-$\\ell_1$"}[gen_name])
    axes[0].set_ylabel("coverage")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig
