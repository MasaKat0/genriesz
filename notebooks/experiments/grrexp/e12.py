"""E-12 (main table): exact balance, inference, omitted direction, BP existence.

Registration: doc/2026-10-07_experiment_registration.md, E-12, §1.4–§1.9.
The notebook ``10_E12_omitted_direction_balance.ipynb`` holds the aggregation,
tables and figures; this module holds the DGP, the arms, the Stage 0 population
computation, and the replication function (importable for process parallelism).
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np

from . import population
from .seeds import Seeds, fold_ids

EXP = 12
THETA0 = 1.0
S_VALUES = (0.5, 1.25)
SETS = ("Include", "Omit")
N_VALUES = (500, 1000, 2000, 4000)
CELLS = [list(c) for c in itertools.product(S_VALUES, SETS, N_VALUES)]  # registered order
R = 2000
K = 5
SQRT3 = np.sqrt(3.0)
GRR_ARMS = ("SQ", "UKL1", "BKL1", "BP_C1", "BP_C0")
ALL_ARMS = GRR_ARMS + ("EB",)
REFERENCE_GATE = {  # registration E-12 Stage 0: stop unless these agree at 1e-6
    ("UKL1", "Omit", 1.25, "sigma2_true"): 7.2906549,
    ("UKL1", "Omit", 1.25, "sigma2_se"): 7.6013032,
    ("SQ", "Include", 0.5, "E_alpha2"): 4.3664466,
    ("UKL1", "Include", 0.5, "E_alpha2"): 4.4173369,
}


def gr():
    import genriesz

    return genriesz


def sign(A):
    return np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)


def generator(arm):
    g = gr()
    return {
        "SQ": lambda: g.SquaredGenerator(C=0.0),
        "UKL1": lambda: g.UKLGenerator(C=1.0, branch_fn=sign),
        "BKL1": lambda: g.BKLGenerator(C=1.0, branch_fn=sign),
        "BP_C1": lambda: g.BPGenerator(omega=0.5, C=1.0, branch_fn=sign),
        "BP_C0": lambda: g.BPGenerator(omega=0.5, C=0.0, branch_fn=sign),
        "EB": lambda: g.UKLGenerator(C=0.0, branch_fn=sign),
    }[arm]()


def features(Z, regressor_set):
    """phi(z): (1, z1, z2, z3, q) for Include, (1, z1, z2, z3) for Omit; q = z2^2 - 1."""
    Z = np.atleast_2d(Z)
    cols = [np.ones(len(Z)), Z[:, 0], Z[:, 1], Z[:, 2]]
    if regressor_set == "Include":
        cols.append(Z[:, 1] ** 2 - 1.0)
    return np.column_stack(cols)


def basis(regressor_set):
    g = gr()
    b = g.TreatmentInteractionBasis(
        base_basis=g.CallableBasis(lambda Z: features(Z, regressor_set))
    )
    return b


def alpha_ref(X):
    return np.where(np.atleast_2d(X)[:, 0] == 1, 2.0, -2.0)


def eta(Z, s):
    q = Z[:, 1] ** 2 - 1.0
    return s * (Z[:, 0] - 0.5 * Z[:, 1] + 0.6 * q)


def gamma0(D, Z):
    return THETA0 * D + Z[:, 0] + 0.5 * (Z[:, 1] ** 2 - 1.0)


def draw(rng, n, s):
    Z = rng.uniform(-SQRT3, SQRT3, size=(n, 3))
    e = 1.0 / (1.0 + np.exp(-eta(Z, s)))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = gamma0(D, Z) + rng.standard_normal(n)
    return D, Z, Y


# ---------------------------------------------------------------- Stage 0


def quadrature(points):
    x, w = np.polynomial.legendre.leggauss(points)
    z = SQRT3 * x
    wz = w / 2.0  # uniform density on [-sqrt3, sqrt3]: dz/(2 sqrt3) with dz = sqrt3 dx
    Z = np.array(list(itertools.product(z, z, z)))
    W = np.array([a * b * c for a, b, c in itertools.product(wz, wz, wz)])
    return Z, W


def population_cell(arm, regressor_set, s, points):
    """Population solution and the §1.9 quantities on a tensor Gauss-Legendre grid."""
    g = gr()
    gen = generator(arm)
    Zq, Wq = quadrature(points)
    e = 1.0 / (1.0 + np.exp(-eta(Zq, s)))
    X = np.vstack(
        [np.column_stack([np.ones(len(Zq)), Zq]), np.column_stack([np.zeros(len(Zq)), Zq])]
    )
    w = np.concatenate([Wq * e, Wq * (1.0 - e)])
    bas = basis(regressor_set)
    bas.fit(X)
    Phi = np.asarray(bas(X))
    fn = g.ATEFunctional(0)
    M = np.asarray(fn.m_basis_matrix(X, bas))
    off = np.asarray(gen.grad(X, alpha_ref(X)))
    sol = population.solve(generator=gen, X=X, w=w, Phi=Phi, M=M, offset=off)
    D, Z = X[:, 0], X[:, 1:]
    a = sol.alpha
    g0 = gamma0(D, Z)
    coef, *_ = np.linalg.lstsq(Phi * np.sqrt(w)[:, None], g0 * np.sqrt(w), rcond=None)
    gb = Phi @ coef
    m_gb = M @ coef  # m(W, gamma_b) = gamma_b(1, z) - gamma_b(0, z)
    v = off + Phi @ sol.beta
    _, _, dalpha = gen.dual_eval(X, v)
    h_beta = a[:, None] * Phi - M
    J = Phi.T @ (Phi * (w * dalpha)[:, None])
    gvec = Phi.T @ (w * dalpha * (g0 - gb))
    theta_star = float(w @ (m_gb + a * (g0 - gb)))
    corr = h_beta @ np.linalg.solve(J, gvec)
    c = m_gb + a * (g0 - gb) - theta_star - corr
    E_a2 = float(w @ a**2)
    return {
        "status": sol.status,
        "max_gradient": sol.max_gradient,
        "min_hessian_eig": sol.min_hessian_eig,
        "min_dual_margin_grid": sol.min_dual_margin,
        "beta": sol.beta,
        "theta_star": theta_star,
        "b": theta_star - THETA0,
        "sigma2_true": float(w @ c**2) + E_a2,
        "sigma2_se": float(w @ (m_gb + a * (g0 - gb) - theta_star) ** 2) + E_a2,
        "E_alpha2": E_a2,
        "offset_at_ref": off,
    }


def support_dual_margin(arm, regressor_set, beta):
    """§1.9 (iii): exact range of the dual coordinate over the whole support box, per arm."""
    gen = generator(arm)
    p = 5 if regressor_set == "Include" else 4
    margins = []
    for d, c in ((1.0, beta[:p]), (0.0, beta[p:])):
        u_ref = float(gen.grad(np.array([[d, 0, 0, 0]]), alpha_ref(np.array([[d]])))[0])
        lo = hi = u_ref + c[0] - (c[4] if p == 5 else 0.0)
        for j in (1, 3):
            lo -= abs(c[j]) * SQRT3
            hi += abs(c[j]) * SQRT3
        # z2 term: c2 z2 + c4 z2^2 on [-sqrt3, sqrt3]
        pts = [-SQRT3, SQRT3]
        if p == 5 and c[4] != 0 and abs(c[2] / (2 * c[4])) <= SQRT3:
            pts.append(-c[2] / (2 * c[4]))
        vals = [c[2] * t + (c[4] * t * t if p == 5 else 0.0) for t in pts]
        lo += min(vals)
        hi += max(vals)
        Xd = np.array([[d, 0, 0, 0], [d, 0, 0, 0]])
        margins.append(float(np.min(gen.dual_margin(Xd, np.array([lo, hi])))))
    return min(margins)


def stage0():
    """All arms x sets x s at 48 and 64 points; certificate (i)-(iv); predictions per n."""
    from .inference import predicted_coverage

    out = []
    for arm, rs, s in itertools.product(ALL_ARMS, SETS, S_VALUES):
        r48 = population_cell(arm, rs, s, 48)
        r64 = population_cell(arm, rs, s, 64)
        cert = (
            r48["status"] == "ok"
            and r64["status"] == "ok"
            and r48["max_gradient"] <= 1e-12
            and r64["max_gradient"] <= 1e-12
            and float(np.max(np.abs(r48["beta"] - r64["beta"]))) <= 1e-10
            and r64["min_hessian_eig"] > 0
        )
        margin = (
            support_dual_margin(arm, rs, r64["beta"]) if r64["status"] == "ok" else float("nan")
        )
        cert = bool(cert and margin >= 1e-4)
        row = {
            "arm": arm,
            "set": rs,
            "s": s,
            "certified": cert,
            "support_dual_margin": margin,
            "beta_diff_48_64": float(np.max(np.abs(r48["beta"] - r64["beta"]))),
            "max_gradient_48": r48["max_gradient"],
            "max_gradient_64": r64["max_gradient"],
            **{
                k: r64[k]
                for k in ("status", "b", "sigma2_true", "sigma2_se", "E_alpha2", "theta_star")
            },
            "beta": r64["beta"].tolist(),
        }
        row["c_n"] = {
            str(n): predicted_coverage(
                np.sqrt(row["sigma2_se"]), np.sqrt(row["sigma2_true"]), row["b"], n
            )
            for n in N_VALUES
        }
        out.append(row)
    gate = {}
    for (arm, rs, s, key), ref in REFERENCE_GATE.items():
        row = next(r for r in out if r["arm"] == arm and r["set"] == rs and r["s"] == s)
        gate[f"{arm}/{rs}/{s}/{key}"] = {
            "computed": row[key],
            "reference": ref,
            "ok": abs(row[key] - ref) <= 1e-6,
        }
    return out, gate


# ---------------------------------------------------------------- Stage 1 replication


def _grr_arm(X, Y, arm, rs, folds):
    g = gr()
    gen = generator(arm)
    bas = basis(rs)
    res = g.grr_functional(
        X=X, Y=Y, m=g.ATEFunctional(0), basis=bas, generator=gen,
        riesz_penalty=None, riesz_lam=0.0, riesz_alpha_ref=alpha_ref,
        outcome_models="shared", outcome_basis=bas, outcome_link="identity", outcome_penalty=None,
        fold_ids=folds, estimators=("arw", "tmle"),
    )  # fmt: skip
    rows = {}
    for est in ("arw", "tmle"):
        if res.success:
            e = res[est]
            rows[f"{arm}|{est.upper()}_cf"] = {"estimate": e.estimate, "se": e.se, "status": "ok"}
        else:
            rows[f"{arm}|{est.upper()}_cf"] = {
                "estimate": np.nan,
                "se": np.nan,
                "status": res.status,
            }
    # RW_full: in-sample fit on the whole sample (registration §1.4)
    bas_full = basis(rs)
    fit = g.GRRGLM(
        basis=bas_full,
        generator=gen,
        functional=g.ATEFunctional(0),
        penalty=None,
        lam=0.0,
        offset=g.offset_from_alpha(gen, alpha_ref),
    )
    fr = fit.fit(X)
    if fr.status == "ok":
        inf = g.rw_full_inference(grr=fit, X=X, Y=Y)
        a_hat = fit.predict_alpha(X)
        Phi = np.asarray(bas_full(X))
        M = np.asarray(g.ATEFunctional(0).m_basis_matrix(X, bas_full))
        imbalance = float(np.max(np.abs((a_hat[:, None] * Phi - M).mean(axis=0))))
        rows[f"{arm}|RW_full"] = {
            "estimate": inf.estimate,
            "se": inf.se,
            "status": "ok",
            "max_abs_alpha": float(np.max(np.abs(a_hat))),
            "imbalance": imbalance,
            "ess": float(np.abs(a_hat).sum() ** 2 / np.sum(a_hat**2)),
        }
        rows[f"{arm}|RW_full"]["_alpha"] = a_hat
        rows[f"{arm}|RW_full"]["_kkt"] = fr.kkt_residual if hasattr(fr, "kkt_residual") else np.nan
    else:
        rows[f"{arm}|RW_full"] = {"estimate": np.nan, "se": np.nan, "status": fr.status}
    return rows


def _arw_ef(X, Y, folds):
    """H12-EF: UKL(1), Include; the representer is fit on the evaluation fold's covariates
    (lambda = 0), the regression on the complement (registration E-12 H12-EF, PR4b
    thm:evaluation_design_crossfit)."""
    g = gr()
    gen = generator("UKL1")
    n = len(Y)
    psi = np.empty(n)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        bas = basis("Include")
        fit = g.GRRGLM(
            basis=bas,
            generator=gen,
            functional=g.ATEFunctional(0),
            penalty=None,
            lam=0.0,
            offset=g.offset_from_alpha(gen, alpha_ref),
        )
        if fit.fit(X[te]).status != "ok":
            return {"estimate": np.nan, "se": np.nan, "status": "fold_fit_failed"}
        pred = _ols(X, Y, "Include", X[tr], Y[tr])
        Xt = X[te]
        m_g = pred(np.column_stack([np.ones(te.sum()), Xt[:, 1:]])) - pred(
            np.column_stack([np.zeros(te.sum()), Xt[:, 1:]])
        )
        psi[te] = m_g + fit.predict_alpha(Xt) * (Y[te] - pred(Xt))
    theta = float(np.mean(psi))
    return {"estimate": theta, "se": float(np.std(psi - theta) / np.sqrt(n)), "status": "ok"}


def _ols(X, Y, rs, X_fit=None, Y_fit=None):
    """Exact least squares on the treatment-interaction span; returns ``predict(rows)``."""
    bas = basis(rs)
    bas.fit(X)
    Xf = X if X_fit is None else X_fit
    Yf = Y if Y_fit is None else Y_fit
    coef, *_ = np.linalg.lstsq(np.asarray(bas(Xf)), Yf, rcond=None)
    return lambda rows: np.asarray(bas(rows)) @ coef


def _ols_ra(X, Y, rs):
    pred = _ols(X, Y, rs)
    n = len(Y)
    return float(
        np.mean(
            pred(np.column_stack([np.ones(n), X[:, 1:]]))
            - pred(np.column_stack([np.zeros(n), X[:, 1:]]))
        )
    )


def replicate(task):
    """One replication of one cell: every arm. ``task = (cell_index, rep, entropy, gbm_params)``.

    Returns one record per arm/estimator plus the deterministic H12-Bal checks; no arrays.
    """
    from . import baselines

    cell_index, rep, entropy, gbm_params = task
    s, rs, n = CELLS[cell_index]
    seeds = Seeds(EXP, entropy=entropy)
    D, Z, Y = draw(seeds.data(cell_index, rep), n, s)
    X = np.column_stack([D, Z])
    folds = fold_ids(n, K, seeds.folds(cell_index, rep))
    out = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings(
            "ignore",
            message=r"(?:divide by zero|overflow|invalid value) encountered in matmul$",
            category=RuntimeWarning,
        )
        for arm in ALL_ARMS:
            out.update(_grr_arm(X, Y, arm, rs, folds))
        if rs == "Include":
            out["UKL1|ARW_EF"] = _arw_ef(X, Y, folds)
        feats = features(Z, rs)[:, 1:]
        out["LogitAIPW"] = baselines.logit_aipw(D, feats, Y, folds)
        seed3 = int(seeds.baseline(cell_index, rep).integers(0, 2**31 - 1))
        out["DML-GBM"] = baselines.dml_gbm(D, Z, Y, folds, gbm_params[0], gbm_params[1], seed3)
        # H12-Bal (deterministic): SQ RW = OLS regression adjustment; Include RW identity
        checks = {}
        sq = out["SQ|RW_full"]
        if sq["status"] == "ok":
            checks["sq_rw_equals_ols"] = abs(sq["estimate"] - _ols_ra(X, Y, rs))
        if rs == "Include":
            g0 = gamma0(D, Z)
            m_g0 = THETA0 * np.ones(n)
            for arm in ALL_ARMS:
                r = out[f"{arm}|RW_full"]
                if r["status"] == "ok":
                    checks[f"{arm}_rw_identity"] = abs(
                        r["estimate"] - float(np.mean(m_g0 + r["_alpha"] * (Y - g0)))
                    )
        if (
            rep < 20 and out["EB|RW_full"]["status"] == "ok"
        ):  # EB = UKL(C=0): independent dual check
            Phi = features(Z, rs)
            a = out["EB|RW_full"]["_alpha"]
            wt = baselines.eb_dual_newton(Phi[D == 1], Phi.sum(axis=0))
            wc = baselines.eb_dual_newton(Phi[D == 0], Phi.sum(axis=0))
            checks["eb_dual_newton_max_diff"] = (
                float("inf")
                if wt is None or wc is None
                else float(max(np.max(np.abs(a[D == 1] - wt)), np.max(np.abs(-a[D == 0] - wc))))
            )
        # H12-BP: weight-program diagnostic (§1.3 A-6, default Clarabel) when a BP full fit fails
        for arm in ("BP_C1", "BP_C0"):
            r = out[f"{arm}|RW_full"]
            if r["status"] != "ok":
                bas = basis(rs)
                cert = gr().weight_program_certificate(
                    X=X,
                    basis=bas,
                    functional=gr().ATEFunctional(0),
                    generator=generator(arm),
                    lam=0.0,
                    backend="cvxpy",
                )
                r["weight_program_verdict"] = cert.verdict
                r["weight_program_gap"] = float(cert.gap) if cert.gap == cert.gap else float("nan")
    for r in out.values():
        r.pop("_alpha", None)
        r.pop("_kkt", None)
    out["_checks"] = checks
    out["_n_warnings"] = sum(1 for w in caught if not issubclass(w.category, DeprecationWarning))
    return out
