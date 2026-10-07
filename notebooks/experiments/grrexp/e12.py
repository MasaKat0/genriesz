"""E-12 (main table): exact balance, inference, omitted direction, BP existence.

Registration: doc/2026-10-07_experiment_registration.md, E-12, §1.4–§1.9.
This module holds the DGP, the arms, the Stage 0 population computation, the
replication function (importable for process parallelism) and the aggregation
shared by the pilot and the confirmatory stage. The notebook
``10_E12_omitted_direction_balance.ipynb`` runs the stages and writes the
outputs; its last cell makes the tables from ``summary.csv`` only (§4).
"""

from __future__ import annotations

import collections
import itertools
import json

import numpy as np

from . import baselines, metrics, population
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
BP_ARMS = ("BP_C1", "BP_C0")
ARM_LABELS = [f"{a}|{est}" for a in ALL_ARMS for est in ("RW_full", "ARW_cf", "TMLE_cf")] + [
    "UKL1|ARW_EF",
    "LogitAIPW",
    "DML-GBM",
]
#: Estimators with a Stage 0 prediction (§1.9); TMLE is a descriptive comparison (§1.4).
PREDICTED_ESTIMATORS = ("RW_full", "ARW_cf", "ARW_EF")
REFERENCE_GATE = {  # registration E-12 Stage 0: stop unless these agree at 1e-6
    ("UKL1", "Omit", 1.25, "sigma2_true"): 7.2906549,
    ("UKL1", "Omit", 1.25, "sigma2_se"): 7.6013032,
    ("SQ", "Include", 0.5, "E_alpha2"): 4.3664466,
    ("UKL1", "Include", 0.5, "E_alpha2"): 4.4173369,
}
#: §1.3 C, lambda = 0 and no constraint: max_j |grad_j| <= 1e-8 max(1, ||P_n m(W, phi)||_inf).
#: genriesz scales ``tol`` by that factor on this path.
SAMPLE_TOL = 1e-8
BALANCE_TOL = 1e-8  # H12-Bal: training imbalance / max(1, ||P_n m||_inf)
IDENTITY_TOL = 1e-10  # H12-Bal identities: rounding term of the Amendment 5 bound
EB_TOL = 1e-6  # EB against the independent dual Newton, first 20 replications
EB_CHECK_REPS = 20
CERT_GRADIENT = 1e-12  # §1.9 (i)
CERT_BETA_DIFF = 1e-10  # §1.9 (i)
CERT_MARGIN = 1e-4  # §1.9 (iii)


def gr():
    import genriesz

    return genriesz


def sign(A):
    return np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)


sign.vectorized = True  # one call on all rows (genriesz BregmanGenerator.branch_fn)


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


def _finite_or_none(x):
    """JSON value of a population quantity: ``None`` when it is not available."""
    x = float(x)
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------- Stage 0


def quadrature(points):
    x, w = np.polynomial.legendre.leggauss(points)
    z = SQRT3 * x
    wz = w / 2.0  # uniform density on [-sqrt3, sqrt3]: dz/(2 sqrt3) with dz = sqrt3 dx
    Z = np.array(list(itertools.product(z, z, z)))
    W = np.array([a * b * c for a, b, c in itertools.product(wz, wz, wz)])
    return Z, W


POPULATION_KEYS = ("theta_star", "b", "sigma2_true", "sigma2_se", "E_alpha2")


def population_cell(arm, regressor_set, s, points):
    """Population solution and the §1.9 quantities on a tensor Gauss-Legendre grid.

    The quantities are computed only when the population solver returns ``ok``;
    otherwise they are NaN (not available).
    """
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
    out = {
        "status": sol.status,
        "max_gradient": sol.max_gradient,
        "min_hessian_eig": sol.min_hessian_eig,
        "beta": sol.beta,
    }
    if sol.status != "ok":
        return {**out, **{k: float("nan") for k in POPULATION_KEYS}}
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
        **out,
        "theta_star": theta_star,
        "b": theta_star - THETA0,
        "sigma2_true": float(w @ c**2) + E_a2,
        "sigma2_se": float(w @ (m_gb + a * (g0 - gb) - theta_star) ** 2) + E_a2,
        "E_alpha2": E_a2,
    }


def support_dual_margin(arm, regressor_set, beta):
    """§1.9 (iii): exact range of the dual coordinate over the whole support box, per arm.

    ``+inf`` when the generator's dual domain is unbounded on the relevant side
    (SQ, UKL, and EB = UKL with C = 0).
    """
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
    """All arms x sets x s at 48 and 64 points; certificate (i)-(iv); predictions per n.

    Every value is JSON-compatible: a quantity that is not available is
    ``None``, and the support margin carries its kind (``finite``,
    ``unbounded`` for a dual domain without a finite end, ``unavailable`` when
    the population solver did not return ``ok``). Predictions ``c_n`` exist
    only for certified cells; the population quantities of an uncertified cell
    are descriptive (§1.9).
    """
    from .inference import predicted_coverage

    out = []
    for arm, rs, s in itertools.product(ALL_ARMS, SETS, S_VALUES):
        r48 = population_cell(arm, rs, s, 48)
        r64 = population_cell(arm, rs, s, 64)
        solved = r48["status"] == "ok" and r64["status"] == "ok"
        beta_diff = float(np.max(np.abs(r48["beta"] - r64["beta"]))) if solved else None
        if solved:
            margin = support_dual_margin(arm, rs, r64["beta"])
            kind = "unbounded" if np.isinf(margin) else "finite"
        else:
            margin, kind = float("nan"), "unavailable"
        cert = bool(
            solved
            and r48["max_gradient"] <= CERT_GRADIENT
            and r64["max_gradient"] <= CERT_GRADIENT
            and beta_diff <= CERT_BETA_DIFF
            and r64["min_hessian_eig"] > 0
            and (kind == "unbounded" or (kind == "finite" and margin >= CERT_MARGIN))
        )
        row = {
            "arm": arm,
            "set": rs,
            "s": s,
            "certified": cert,
            "status_48": r48["status"],
            "status": r64["status"],
            "support_dual_margin": _finite_or_none(margin),
            "support_dual_margin_kind": kind,
            "beta_diff_48_64": beta_diff,
            "max_gradient_48": _finite_or_none(r48["max_gradient"]),
            "max_gradient_64": _finite_or_none(r64["max_gradient"]),
            "min_hessian_eig_64": _finite_or_none(r64["min_hessian_eig"]),
            **{k: _finite_or_none(r64[k]) for k in POPULATION_KEYS},
            "beta": [_finite_or_none(x) for x in r64["beta"]],
        }
        row["c_n"] = (
            {
                str(n): predicted_coverage(
                    np.sqrt(row["sigma2_se"]), np.sqrt(row["sigma2_true"]), row["b"], n
                )
                for n in N_VALUES
            }
            if cert
            else None
        )
        out.append(row)
    gate = {}
    for (arm, rs, s, key), ref in REFERENCE_GATE.items():
        row = next(r for r in out if r["arm"] == arm and r["set"] == rs and r["s"] == s)
        computed = row[key]
        gate[f"{arm}/{rs}/{s}/{key}"] = {
            "computed": computed,
            "reference": ref,
            "ok": computed is not None and abs(computed - ref) <= 1e-6,
        }
    return out, gate


# ---------------------------------------------------------------- Stage 1 replication


def _failed(status, **extra):
    return {"estimate": float("nan"), "se": float("nan"), "status": status, **extra}


def _warning_fields(counts):
    return {"n_warnings": int(sum(counts.values())), "warnings": json.dumps(counts)}


def _warning_record(fits, other=None, diagnostic=None):
    """Warnings of one estimator record (§1.1): per individual fit ``[fold, model,
    counts]``, the rest of the computation (``other``), and the H12-BP weight-program
    diagnostic, kept apart."""
    other, diagnostic = other or {}, diagnostic or {}
    n = sum(sum(c.values()) for _, _, c in fits) + sum(other.values()) + sum(diagnostic.values())
    record = {"fits": fits, "other": other, "diagnostic": diagnostic}
    return {"n_warnings": int(n), "warnings": json.dumps(record)}


class _FitWarnings:
    """``fit_hook`` of ``grr_functional``: records the warnings of each individual fit."""

    def __init__(self):
        self.fits, self.converged, self.last_converged = [], True, True

    def hook(self, stage, fold, fit):
        result, counts, converged = baselines.run_recording_warnings(fit)
        if counts:
            self.fits.append([int(fold), stage, counts])
        self.last_converged = converged
        self.converged = self.converged and converged
        return result


def _bp_diagnostic(arm, rs, X_fit):
    """§1.3 A-6: the lambda = 0 weight program on the rows of the failed fit (default
    Clarabel settings). A diagnostic only: never a claim of non-existence."""
    g = gr()
    cert = g.weight_program_certificate(
        X=X_fit,
        basis=basis(rs),
        functional=g.ATEFunctional(0),
        generator=generator(arm),
        lam=0.0,
        backend="cvxpy",
    )
    return {
        "wp_verdict": str(cert.verdict),
        "wp_gap": float(cert.gap),
        "wp_n_at_boundary": float(cert.n_at_boundary),
        "wp_min_margin": float(cert.min_margin),
        "wp_solver_status": str(cert.solver_status),
    }


def _grr_cf(X, Y, arm, rs, folds):
    """ARW_cf and TMLE_cf (one cross-fitted representer per fold, shared by both)."""
    g = gr()
    gen = generator(arm)

    fw = _FitWarnings()
    res, other, converged = baselines.run_recording_warnings(
        lambda: g.grr_functional(
            X=X, Y=Y, m=g.ATEFunctional(0), basis=basis(rs), generator=gen,
            riesz_penalty=None, riesz_lam=0.0, riesz_alpha_ref=alpha_ref, riesz_tol=SAMPLE_TOL,
            outcome_models="shared", outcome_link="identity", outcome_penalty="l2", outcome_lam=0.0,
            fold_ids=folds, estimators=("arw", "tmle"), expose_alpha_values=True,
            fit_hook=fw.hook,
        )
    )  # fmt: skip
    converged = converged and fw.converged
    diag, diag_counts = {}, {}
    failed_fit = [f for f in res.fold_status if f[1] == "riesz" and f[2] != "ok"]
    if not res.success and arm in BP_ARMS and failed_fit:
        diag, diag_counts, _ = baselines.run_recording_warnings(
            lambda: _bp_diagnostic(arm, rs, X[folds != failed_fit[0][0]])
        )
    tb = res.diagnostics.get("train_imbalance", {"max": [], "scale": []})
    common = {
        "fold_status": json.dumps([[str(v) for v in f] for f in res.fold_status]),
        **_train_balance_fields(tb["max"], tb["scale"]),
        **diag,
    }
    # One grr_functional call fits the folds shared by ARW_cf and TMLE_cf: its
    # warnings are counted once, on the ARW_cf record.
    warn = {
        "ARW_cf": _warning_record(fw.fits, other, diag_counts),
        "TMLE_cf": {"n_warnings": 0, "warnings": json.dumps({"counted_in": "ARW_cf"})},
    }
    if not res.success or not converged:
        status = res.status if not res.success else "convergence_warning"
        return {f"{arm}|{e}": _failed(status, **common, **warn[e]) for e in ("ARW_cf", "TMLE_cf")}
    d = res.diagnostics
    a = np.asarray(d["alpha_values"], dtype=float)
    fit_diag = {
        "max_abs_alpha": float(np.max(np.abs(a))),
        "ess": metrics.ess(a),
        "eval_imbalance": float(d["held_out_imbalance_max"]),
    }
    return {
        f"{arm}|{e}": {
            "estimate": float(res[e.split("_")[0].lower()].estimate),
            "se": float(res[e.split("_")[0].lower()].se),
            "status": "ok",
            **fit_diag,
            **common,
            **warn[e],
        }
        for e in ("ARW_cf", "TMLE_cf")
    }


def _train_balance_fields(maxes, scales):
    """Training balance of the representer fits that succeeded (also in a failed call)."""
    ratios = [m / s for m, s in zip(maxes, scales, strict=True)]
    return {
        "train_fits": len(ratios),
        "train_imbalance": float(max(maxes)) if maxes else float("nan"),
        "train_balance_ratio": float(max(ratios)) if ratios else float("nan"),
    }


def _balance(model, X_fit):
    """Training imbalance ``max_j |P_n(alpha phi_j - m(W, phi_j))|`` and its scale."""
    Phi = np.asarray(model.basis(X_fit), dtype=float)
    M = np.asarray(gr().ATEFunctional(0).m_basis_matrix(X_fit, model.basis), dtype=float)
    a = np.asarray(model.predict_alpha(X_fit), dtype=float)
    imb = float(np.max(np.abs(np.mean(a[:, None] * Phi - M, axis=0))))
    return imb, max(1.0, float(np.max(np.abs(M.mean(axis=0)))))


def _grr_model(arm, rs):
    g = gr()
    gen = generator(arm)
    return g.GRRGLM(
        basis=basis(rs),
        generator=gen,
        functional=g.ATEFunctional(0),
        penalty=None,
        lam=0.0,
        offset=g.offset_from_alpha(gen, alpha_ref),
    )


def _rw_full(X, Y, arm, rs):
    """RW_full: in-sample fit on the whole sample (registration §1.4). Returns the
    record and the fitted weights (for the H12-Bal identities), or ``None``."""

    model = _grr_model(arm, rs)
    fw = _FitWarnings()
    fr = fw.hook("riesz", -1, lambda: model.fit(X, tol=SAMPLE_TOL))  # fold -1: the full sample
    detail = json.dumps([["full", "riesz", str(fr.status), str(fr.message)]])
    if fr.status != "ok":
        diag, diag_counts = {}, {}
        if arm in BP_ARMS:
            diag, diag_counts, _ = baselines.run_recording_warnings(
                lambda: _bp_diagnostic(arm, rs, X)
            )
        return _failed(
            fr.status, fold_status=detail, **_warning_record(fw.fits, None, diag_counts), **diag
        ), None
    inf, other, converged = baselines.run_recording_warnings(
        lambda: gr().rw_full_inference(grr=model, X=X, Y=Y)
    )
    common = {"fold_status": detail, **_warning_record(fw.fits, other)}
    if not (converged and fw.converged):
        return _failed("convergence_warning", **common), None
    a = np.asarray(model.predict_alpha(X), dtype=float)
    imb, scale = _balance(model, X)
    rec = {
        "estimate": float(inf.estimate),
        "se": float(inf.se),
        "status": "ok",
        "max_abs_alpha": float(np.max(np.abs(a))),
        "ess": metrics.ess(a),
        **_train_balance_fields([imb], [scale]),
        **common,
    }
    return rec, a


def _arw_ef(X, Y, folds):
    """H12-EF: UKL(1), Include; the representer is fit on the evaluation fold's covariates
    (lambda = 0), the regression on the complement (registration E-12 H12-EF, PR4b
    thm:evaluation_design_crossfit). A failed fold is reported with its own status."""
    n = len(Y)

    psi, alpha = np.empty(n), np.empty(n)
    detail, imbs, scales = [], [], []
    fw = _FitWarnings()

    def fit():
        for k in np.unique(folds):
            tr, te = folds != k, folds == k
            model = _grr_model("UKL1", "Include")
            fr = fw.hook("riesz", k, lambda m=model, rows=te: m.fit(X[rows], tol=SAMPLE_TOL))
            status = str(fr.status) if fw.last_converged else "convergence_warning"
            detail.append([str(int(k)), "riesz", status, str(fr.message)])
            if status != "ok":
                return status
            Xt = X[te]
            # the fold's own fit: its balance is checked even if a later fold fails
            imb, scale = _balance(model, Xt)
            imbs.append(imb)
            scales.append(scale)
            a, outside, nonfinite = model.classify(Xt)
            if np.any(outside):
                detail.append([str(int(k)), "prediction", "domain_prediction", ""])
                return "domain_prediction"
            if np.any(nonfinite):
                detail.append([str(int(k)), "prediction", "nonfinite", ""])
                return "nonfinite"
            # a warning of the outcome fit fails the replication after the loop (as before)
            pred = fw.hook("outcome", k, lambda tr=tr: _ols(X, Y, "Include", X[tr], Y[tr]))
            m_g = pred(np.column_stack([np.ones(te.sum()), Xt[:, 1:]])) - pred(
                np.column_stack([np.zeros(te.sum()), Xt[:, 1:]])
            )
            psi[te] = m_g + a * (Y[te] - pred(Xt))
            alpha[te] = a
        return "ok" if np.all(np.isfinite(psi)) else "nonfinite"

    status, other, converged = baselines.run_recording_warnings(fit)
    converged = converged and fw.converged
    common = {
        **_warning_record(fw.fits, other),
        "fold_status": json.dumps(detail),
        **_train_balance_fields(imbs, scales),
    }
    if status != "ok" or not converged:
        return _failed(status if status != "ok" else "convergence_warning", **common)
    theta = float(np.mean(psi))
    return {
        "estimate": theta,
        "se": float(np.std(psi - theta) / np.sqrt(n)),
        "status": "ok",
        "max_abs_alpha": float(np.max(np.abs(alpha))),
        "ess": metrics.ess(alpha),
        **common,
    }


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


def _baseline(call):
    res, counts, converged = baselines.run_recording_warnings(call)
    fields = _warning_fields(counts)
    return {**res, **fields} if converged else _failed("convergence_warning", **fields)


def replicate(task):
    """One replication of one cell: every arm. ``task = (cell_index, rep, entropy, gbm_params)``.

    Returns one record per arm/estimator plus the deterministic H12-Bal checks; no arrays.
    """
    cell_index, rep, entropy, gbm_params = task
    s, rs, n = CELLS[cell_index]
    seeds = Seeds(EXP, entropy=entropy)
    D, Z, Y = draw(seeds.data(cell_index, rep), n, s)
    X = np.column_stack([D, Z])
    folds = fold_ids(n, K, seeds.folds(cell_index, rep))
    out, alphas = {}, {}
    for arm in ALL_ARMS:
        out[f"{arm}|RW_full"], alphas[arm] = _rw_full(X, Y, arm, rs)
        out.update(_grr_cf(X, Y, arm, rs, folds))
    if rs == "Include":
        out["UKL1|ARW_EF"] = _arw_ef(X, Y, folds)
    feats = features(Z, rs)[:, 1:]
    out["LogitAIPW"] = _baseline(lambda: baselines.logit_aipw(D, feats, Y, folds))
    seed3 = int(seeds.baseline(cell_index, rep).integers(0, 2**31 - 1))
    out["DML-GBM"] = _baseline(
        lambda: baselines.dml_gbm(D, Z, Y, folds, gbm_params[0], gbm_params[1], seed3)
    )

    def checks():
        # H12-Bal (deterministic, Amendment 5): SQ RW = OLS regression adjustment and the
        # Include RW identity, each with its bound ||c||_1 max_j |Delta_hat_j| + 1e-10.
        bas = basis(rs)
        bas.fit(X)
        Phi_b = np.asarray(bas(X), dtype=float)
        M_b = np.asarray(gr().ATEFunctional(0).m_basis_matrix(X, bas), dtype=float)

        def bound(coef, a):
            delta = np.mean(a[:, None] * Phi_b - M_b, axis=0)
            return float(np.sum(np.abs(coef)) * np.max(np.abs(delta)) + IDENTITY_TOL)

        c = {}
        if alphas["SQ"] is not None:
            c_ols, *_ = np.linalg.lstsq(Phi_b, Y, rcond=None)
            c["sq_rw_equals_ols"] = abs(out["SQ|RW_full"]["estimate"] - _ols_ra(X, Y, rs))
            c["sq_rw_equals_ols_bound"] = bound(c_ols, alphas["SQ"])
        if rs == "Include":
            g0 = gamma0(D, Z)
            c_g0, *_ = np.linalg.lstsq(Phi_b, g0, rcond=None)  # gamma0 is in the Include span
            for arm in ALL_ARMS:
                if alphas[arm] is not None:
                    c[f"{arm}_rw_identity"] = abs(
                        out[f"{arm}|RW_full"]["estimate"]
                        - float(np.mean(THETA0 + alphas[arm] * (Y - g0)))
                    )
                    c[f"{arm}_rw_identity_bound"] = bound(c_g0, alphas[arm])
        if rep < EB_CHECK_REPS and alphas["EB"] is not None:  # EB = UKL(C=0): independent dual
            Phi = features(Z, rs)
            a = alphas["EB"]
            wt = baselines.eb_dual_newton(Phi[D == 1], Phi.sum(axis=0), n)
            wc = baselines.eb_dual_newton(Phi[D == 0], Phi.sum(axis=0), n)
            c["eb_dual_newton_max_diff"] = (
                float("inf")
                if wt is None or wc is None
                else float(max(np.max(np.abs(a[D == 1] - wt)), np.max(np.abs(-a[D == 0] - wc))))
            )
        return c

    out["_checks"], counts, _ = baselines.run_recording_warnings(checks)
    out["_check_warnings"] = int(sum(counts.values()))
    return out


# ---------------------------------------------------------------- aggregation

NUMERIC_FIELDS = (
    "estimate",
    "se",
    "max_abs_alpha",
    "ess",
    "train_fits",
    "train_imbalance",
    "train_balance_ratio",
    "eval_imbalance",
    "n_warnings",
    "wp_gap",
    "wp_n_at_boundary",
    "wp_min_margin",
)
TEXT_FIELDS = ("status", "warnings", "fold_status", "wp_verdict", "wp_solver_status")


def tune_all(entropy):
    """GBM hyper-parameters per (s, n) on the pilot sample of the (s, Include, n) cell
    (stream 5), with warnings recorded; a ConvergenceWarning stops the run."""
    seeds = Seeds(EXP, entropy=entropy)
    params, n_warnings = {}, 0
    for ci, (s, rs, n) in enumerate(CELLS):
        if rs != "Include":
            continue
        params[(s, n)], w = tune_cell(seeds, ci)
        n_warnings += w
    return params, n_warnings


def tune_cell(seeds, ci):
    s, _, n = CELLS[ci]
    D, Z, Y = draw(seeds.pilot(ci), n, s)
    cv = fold_ids(n, 5, seeds.inner_cv(ci, 0))
    seed3 = int(seeds.baseline(ci, 0).integers(0, 2**31 - 1))
    params, counts, converged = baselines.run_recording_warnings(
        lambda: baselines.tune_gbm(D, Z, Y, cv, seed3)
    )
    if not converged:
        raise RuntimeError(f"ConvergenceWarning in the GBM tuning of cell {CELLS[ci]}: {counts}")
    return params, int(sum(counts.values()))


def tuning_cell(ci):
    """Index of the (s, Include, n) cell whose pilot sample tunes cell ``ci``."""
    s, _, n = CELLS[ci]
    return CELLS.index([s, "Include", n])


def tasks_for(entropy, reps, gbm, cells=None):
    cells = range(len(CELLS)) if cells is None else cells
    return [
        (ci, r, entropy, gbm[(CELLS[ci][0], CELLS[ci][2])]) for ci in cells for r in range(reps)
    ]


def raw_frame(tasks, results):
    """One row per replication x arm/estimator; failures are rows too (§1.7)."""
    import pandas as pd

    rows = []
    for (ci, rep, _, _), res in zip(tasks, results, strict=True):
        s, rs, n = CELLS[ci]
        checks = {f"check_{k}": v for k, v in res["_checks"].items()}
        for label in ARM_LABELS:
            if label not in res:
                continue
            r = res[label]
            rows.append(
                {
                    "cell": ci, "s": s, "set": rs, "n": n, "rep": rep, "arm": label,
                    **{k: float(r.get(k, np.nan)) for k in NUMERIC_FIELDS},
                    **{k: str(r.get(k, "")) for k in TEXT_FIELDS},
                    "n_check_warnings": res["_check_warnings"],
                    **checks,
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


def total_warnings(raw):
    per_rep = raw.drop_duplicates(["cell", "rep"])["n_check_warnings"].sum()
    return int(raw["n_warnings"].sum() + per_rep)


def balance_checks(raw):
    """H12-Bal (deterministic). Any violation raises: an implementation error stops the run.

    - every successful GRR fit (RW_full, each cross-fitted fold, each ARW_EF fold)
      has training imbalance <= 1e-8 max(1, ||P_n m(W, phi)||_inf), whatever the
      final status of its replication (a later fold may have failed); a
      successful replication carries all its fits (1 for RW_full, K otherwise);
    - the SQ RW = OLS and Include RW identities hold within their bound
      ``||c||_1 max_j |Delta_hat_j| + 1e-10`` (Amendment 5; ``c`` the OLS or gamma0
      coefficients on the span, ``Delta_hat`` the fit's imbalance);
    - EB agrees with the independent dual Newton at 1e-6 in the first 20 replications.
    """
    ok = raw["status"] == "ok"
    grr_ok = ok & raw["arm"].str.contains("|", regex=False)
    expected = np.where(raw["arm"].str.endswith("RW_full"), 1, K)
    missing = int((grr_ok & (raw["train_fits"] != expected)).sum())
    has = raw["train_fits"] > 0
    ratio = raw.loc[has, "train_balance_ratio"]
    identity_cols = [
        c
        for c in raw.columns
        if c.startswith("check_")
        and not c.endswith("_bound")
        and c != "check_eb_dual_newton_max_diff"
    ]
    identities = {c: float(raw[c].abs().max()) for c in identity_cols}
    # Amendment 5: each identity error against its own bound ||c||_1 max_j |Delta_hat_j| + 1e-10
    excess = {c: float((raw[c] - raw[c + "_bound"]).max()) for c in identity_cols}
    eb_due = (raw["arm"] == "EB|RW_full") & ok & (raw["rep"] < EB_CHECK_REPS)
    eb = (
        raw.loc[eb_due, "check_eb_dual_newton_max_diff"]
        if "check_eb_dual_newton_max_diff" in raw
        else None
    )
    bal = {
        # ARW_cf and TMLE_cf share their fits: count them once (on ARW_cf)
        "fits_checked": int(raw.loc[has & ~raw["arm"].str.endswith("TMLE_cf"), "train_fits"].sum()),
        "train_balance_ratio_max": float(ratio.max()) if len(ratio) else None,
        "train_balance_tolerance": BALANCE_TOL,
        "identities_max": identities,
        "identities_max_excess_over_bound": excess,
        "eb_checked": int(eb_due.sum()),
        "eb_max": float(eb.max()) if eb is not None and len(eb) else None,
    }
    violations = []
    if missing:
        violations.append(f"{missing} successful GRR records without the balance of every fit")
    if len(ratio) and not ratio.max() <= BALANCE_TOL:
        violations.append("training imbalance above 1e-8 max(1, ||P_n m||)")
    violations += [c for c, v in excess.items() if not v <= 0.0]
    if eb_due.any() and (eb is None or eb.isna().any() or not eb.max() <= EB_TOL):
        violations.append("EB differs from the independent dual Newton")
    if violations:
        raise AssertionError(f"H12-Bal violated: {violations}; {json.dumps(bal)}")
    return bal


def moment_status(x):
    """Which moment quantities of §1.5 and §1.8 are defined for the successful estimates.

    ``ok``; ``fewer_than_two`` (no SD); ``zero_variance`` (no bias test, no SD
    ratio); ``negative_radicand``: the plug-in ``m4 - s^4`` of the delta-method
    MCSE of the SD is negative (possible for a few successes), so the SD MCSE and
    the SD-ratio test are undefined; ``zero_radicand``: ``m4 - s^4 = 0``, so the
    SD MCSE is 0 and the SD-ratio test (a normal approximation with se 0) is
    undefined. The variance, the fourth moment and the radicand are reported in
    the summary, never truncated.
    """
    if x.size < 2:
        return "fewer_than_two"
    var, m4 = metrics.moment_terms(x)
    if not var > 0.0:
        return "zero_variance"
    radicand = m4 - var**2
    if radicand < 0.0:
        return "negative_radicand"
    return "ok" if radicand > 0.0 else "zero_radicand"


#: moment statuses with a positive variance: the bias test and the SE ratio are defined
POSITIVE_VARIANCE = ("ok", "negative_radicand", "zero_radicand")
#: an undefined test stays in its registered family as a non-rejecting entry, so
#: the Holm multiplicity of the family does not depend on the outcome (§1.8)
UNAVAILABLE_P = 1.0


def summarise(raw, population_rows):
    """Registered metrics (§1.5) per cell x arm/estimator, in the registered order.

    Failure counts and the unconditional coverage are computed for every row;
    the moments are computed where :func:`moment_status` says they are defined,
    and the rest is marked by ``moment_status``.
    Predictions come only from certified Stage 0 cells and only for RW_full,
    ARW_cf and ARW_EF (§1.9); TMLE is descriptive.
    """
    import pandas as pd

    pop = {(r["arm"], r["set"], r["s"]): r for r in population_rows}
    rows = []
    for ci, (s, rs, n) in enumerate(CELLS):
        for label in ARM_LABELS:
            g = raw[(raw["cell"] == ci) & (raw["arm"] == label)].sort_values("rep")
            if g.empty:
                continue
            ok = (g["status"] == "ok").to_numpy()
            est, se = g["estimate"].to_numpy(), g["se"].to_numpy()
            r_s = int(ok.sum())
            row = {"cell": ci, "s": s, "set": rs, "n": n, "arm": label}
            row.update(metrics.failure_rate(ok))
            row.update(metrics.coverage(est, se, ok, THETA0))
            row["status_counts"] = json.dumps(
                dict(sorted(collections.Counter(g["status"]).items()))
            )
            row["warnings"] = int(g["n_warnings"].sum())
            x = est[ok]
            ms = row["moment_status"] = moment_status(x)
            if r_s >= 1:
                row.update(metrics.ci_length(se, ok))
                if g.loc[ok, "max_abs_alpha"].notna().all():
                    row.update(metrics.max_weight_summary(g["max_abs_alpha"].to_numpy(), ok))
                    row["ess_median"] = float(g.loc[ok, "ess"].median())
                    row["train_imbalance_max"] = float(g.loc[ok, "train_imbalance"].max())
                    if g.loc[ok, "eval_imbalance"].notna().all():
                        row["eval_imbalance_median"] = float(g.loc[ok, "eval_imbalance"].median())
                        row["eval_imbalance_max"] = float(g.loc[ok, "eval_imbalance"].max())
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
            arm, _, estimator = label.partition("|")
            if estimator:
                p = pop[(arm, rs, s)]
                row["certified"] = bool(p["certified"])
                if p["certified"] and estimator in PREDICTED_ESTIMATORS:
                    row.update(
                        {
                            "b_pred": p["b"],
                            "sigma_true": float(np.sqrt(p["sigma2_true"])),
                            "root_n_sd_pred": float(np.sqrt(p["sigma2_true"])),
                            "c_pred": p["c_n"][str(n)],
                        }
                    )
            if arm in BP_ARMS:
                d = g[g["wp_verdict"] != ""]
                row["wp_diagnosed"] = len(d)
                if len(d):
                    row["wp_verdicts"] = json.dumps(
                        dict(sorted(collections.Counter(d["wp_verdict"]).items()))
                    )
                    row["wp_solver_statuses"] = json.dumps(
                        dict(sorted(collections.Counter(d["wp_solver_status"]).items()))
                    )
                    row["wp_n_at_boundary_max"] = float(d["wp_n_at_boundary"].max())
                    row["wp_gap_max"] = float(d["wp_gap"].max())
                    row["wp_min_margin_min"] = float(d["wp_min_margin"].min())
            rows.append(row)
    return pd.DataFrame(rows)


def family1_mcse(raw, summ, seeds):
    """Family 1 (§1.9): descriptive percentile 95% intervals, B = 2000, one generator.

    Order: rows of ``summ`` in order (registered cell order, then ``ARM_LABELS``),
    and within a row CI length (mean), max weight (median), SE ratio. The
    resampling unit is a successful replication (the three statistics are
    defined over successful replications); rows with fewer than two successes
    are skipped. A resample in which every SE-ratio draw repeats one estimate has
    no SD; the SE-ratio interval is then reported as undefined. Each interval
    carries its status (``ok``, ``fewer_than_two_successes``, ``no_weights``,
    ``zero_sd_resample``).
    """
    from .bootstrap import FamilyBootstrap, percentile_interval

    fb = FamilyBootstrap(seeds, 1)
    names = ("ci_length_mean", "max_weight_median", "se_ratio")
    out = {f"{k}_mcse_{side}": [np.nan] * len(summ) for k in names for side in ("lo", "hi")}
    out.update({f"{k}_mcse_status": ["fewer_than_two_successes"] * len(summ) for k in names})
    for i, (_, r) in enumerate(summ.iterrows()):
        g = raw[(raw["cell"] == r["cell"]) & (raw["arm"] == r["arm"])].sort_values("rep")
        g = g[g["status"] == "ok"]
        if len(g) < 2:
            continue
        out["max_weight_median_mcse_status"][i] = "no_weights"
        est, se = g["estimate"].to_numpy(), g["se"].to_numpy()
        length = 2.0 * metrics.Z_95 * se
        stats = [("ci_length_mean", lambda idx, v=length: v[idx].mean(axis=1))]
        if g["max_abs_alpha"].notna().all():
            mw = g["max_abs_alpha"].to_numpy()
            stats.append(("max_weight_median", lambda idx, v=mw: np.median(v[idx], axis=1)))

        def se_ratio(idx, e=est, s_=se):
            sd = e[idx].std(axis=1, ddof=1)
            mean_se = s_[idx].mean(axis=1)
            return np.divide(mean_se, sd, out=np.full(sd.shape, np.nan), where=sd > 0)

        stats.append(("se_ratio", se_ratio))
        for name, stat in stats:
            boot = fb.replicate(stat, len(g), finite=False)
            if np.all(np.isfinite(boot)):
                lo, hi = percentile_interval(boot, 1)
                out[f"{name}_mcse_lo"][i], out[f"{name}_mcse_hi"][i] = lo, hi
                out[f"{name}_mcse_status"][i] = "ok"
            else:
                out[f"{name}_mcse_status"][i] = "zero_sd_resample"
    return out


def families(summ, raw):
    """H12-Inf, H12-Bias, H12-EF and H12-BP (Holm 0.01, §1.8); ``None`` for a family
    with no certified cell in ``summ``.

    Every registered null stays in its family. A test that is undefined for the
    observed replications (see :func:`moment_status`) enters as a non-rejecting
    entry (``UNAVAILABLE_P``), so the multiplicity is the registered one, and is
    listed per family in the returned ``unavailable``.
    """
    from . import inference

    unavailable = {}

    def tests_for(hypothesis, sub):
        tests, missing = {}, unavailable.setdefault(hypothesis, [])
        for _, r in sub.iterrows():
            key = f"{r['arm']}|s={r['s']}|{r['set']}|n={r['n']}"
            tests[key + "|coverage"] = inference.coverage_pvalue(
                int(r["covered"]), int(r["R"]), r["c_pred"]
            )
            tests[key + "|failure"] = inference.failure_rate_pvalue(
                int(r["R"] - r["R_s"]), int(r["R"]), 0.01
            )
            ms = r["moment_status"]
            g = raw[(raw["cell"] == r["cell"]) & (raw["arm"] == r["arm"])]
            x = g.loc[g["status"] == "ok", "estimate"].to_numpy()
            if ms in POSITIVE_VARIANCE:
                tests[key + "|bias"] = inference.bias_pvalue(
                    x, THETA0, r["b_pred"], r["sigma_true"], r["n"]
                )
            else:
                tests[key + "|bias"] = UNAVAILABLE_P
                missing.append(f"{key}|bias ({ms})")
            if ms == "ok":
                tests[key + "|sd_ratio"] = inference.sd_ratio_pvalue(x, r["sigma_true"], r["n"])
            else:
                tests[key + "|sd_ratio"] = UNAVAILABLE_P
                missing.append(f"{key}|sd_ratio ({ms})")
        return tests

    def judge(hypothesis, tests):
        return inference.judge_family(hypothesis, tests) if tests else None

    predicted = summ["c_pred"].notna() if "c_pred" in summ else summ["n"] < 0
    big = summ["n"].isin([2000, 4000]) & predicted
    rw_arw = summ["arm"].isin([f"{a}|{e}" for a in GRR_ARMS for e in ("RW_full", "ARW_cf")])
    fam = {
        "H12-Inf": judge(
            "H12-Inf", tests_for("H12-Inf", summ[big & rw_arw & (summ["set"] == "Include")])
        ),
        "H12-Bias": judge(
            "H12-Bias", tests_for("H12-Bias", summ[big & rw_arw & (summ["set"] == "Omit")])
        ),
        "H12-EF": judge("H12-EF", tests_for("H12-EF", summ[big & (summ["arm"] == "UKL1|ARW_EF")])),
    }
    bp = summ[
        (summ["n"] == 4000)
        & (summ["certified"] == True)  # noqa: E712
        & summ["arm"].isin([f"{a}|{e}" for a in BP_ARMS for e in ("RW_full", "ARW_cf")])
    ]
    fam["H12-BP"] = judge(
        "H12-BP",
        {
            f"{r['arm']}|s={r['s']}|{r['set']}": inference.failure_rate_pvalue(
                int(r["R"] - r["R_s"]), int(r["R"]), 0.01
            )
            for _, r in bp.iterrows()
        },
    )
    return fam, unavailable


ORDER_ARMS = ("SQ", "BP_C1", "UKL1", "BKL1")


def ordering(raw, seeds):
    """H12-Ord: s = 1.25, Include, n = 4000, RW_full; replications where the four arms
    all succeeded; family 3 in adjacent-pair order; intersection-union rule."""
    from .bootstrap import FamilyBootstrap, percentile_interval

    ci = CELLS.index([1.25, "Include", 4000])
    labels = [f"{a}|RW_full" for a in ORDER_ARMS]
    sub = raw[(raw["cell"] == ci) & raw["arm"].isin(labels)]
    wide = sub.pivot(index="rep", columns="arm", values="max_abs_alpha")[labels]
    stat = sub.pivot(index="rep", columns="arm", values="status")[labels]
    all_ok = (stat == "ok").all(axis=1)
    share = float(all_ok.mean())
    res = {
        "share_all_succeeded": share,
        "n_all_succeeded": int(all_ok.sum()),
        "conditional_on": "replications in which SQ, BP(C=1), UKL(C=1) and BKL(C=1) all succeeded",
        "pairs": [],
    }
    if share < 0.5:
        res["verdict"] = "undeterminable"
        return res
    mw = wide[all_ok].to_numpy()
    fb = FamilyBootstrap(seeds, 3)
    passes = []
    for j in range(3):
        boot = fb.replicate(
            lambda idx, j=j: np.median(mw[idx, j + 1], axis=1) - np.median(mw[idx, j], axis=1),
            mw.shape[0],
        )
        lo, hi = percentile_interval(boot, 3)
        res["pairs"].append(
            {
                "pair": f"{ORDER_ARMS[j]}<{ORDER_ARMS[j + 1]}",
                "median_diff": float(np.median(mw[:, j + 1]) - np.median(mw[:, j])),
                "ci": [lo, hi],
            }
        )
        passes.append(lo > 0)
    res["verdict"] = "pass" if all(passes) else "fail"
    return res


# ---------------------------------------------------------------- tables (from summary.csv)

LABEL = {
    "SQ": "SQ",
    "UKL1": "UKL ($C=1$)",
    "BKL1": "BKL ($C=1$)",
    "BP_C1": "BP ($C=1$)",
    "BP_C0": "BP ($C=0$)",
    "EB": "EB (= UKL, $C=0$)",
    "LogitAIPW": "LogitAIPW",
    "DML-GBM": "DML-GBM",
}
END = " \\\\"  # LaTeX row end


def _fmt(x, d=3):
    return "--" if x is None or x != x else f"{float(x):.{d}f}"


def _tex(x):
    return str(x).replace("_", "\\_").replace("|", " ").replace("%", "\\%").replace("<", "$<$")


def _table(df, cols, heads, spec):
    """A table body set as a longtable by the manuscript: the head is repeated on every page."""
    head = ["\\hline", " & ".join(heads) + END, "\\hline"]
    lines = ["\\begin{tabular}{" + spec + "}", *head, "\\endfirsthead", *head, "\\endhead"]
    for _, r in df.iterrows():
        lines.append(
            " & ".join(_fmt(r[c]) if isinstance(r[c], float) else _tex(r[c]) for c in cols) + END
        )
    return "\n".join(lines + ["\\hline", "\\end{tabular}"]) + "\n"


def _main_label(row_arm, estimator):
    if row_arm in ("LogitAIPW", "DML-GBM"):
        return row_arm, LABEL[row_arm]
    if estimator == "TMLE_cf":
        return f"{row_arm}|TMLE_cf", f"{LABEL[row_arm]}, TMLE"
    return f"{row_arm}|ARW_cf", LABEL[row_arm]


def tables(S):
    """Tables and macros of E-12, from the rows of ``summary.csv`` only (§4).

    The tables are longtable bodies (``\\endfirsthead``/``\\endhead`` after the head): the
    manuscript sets them with ``longtable`` and supplies the caption.
    """
    out = {}
    main_rows = [(a, "ARW_cf") for a in ALL_ARMS] + [("LogitAIPW", ""), ("DML-GBM", "")]
    main_rows += [(a, "TMLE_cf") for a in ALL_ARMS]
    heads = ["Method"] + [
        "Bias ($b$)", "$\\sqrt{n}$SD (pred.)", "SE ratio", "Coverage ($c(n)$)", "CI length",
        "Fail \\%", "med. $\\max|\\widehat\\alpha|$",
    ] * 2  # fmt: skip
    lines = [
        "\\begin{tabular}{l" + "rrrrrrr" * 2 + "}", "\\hline",
        " & \\multicolumn{7}{c}{$n=2000$} & \\multicolumn{7}{c}{$n=4000$}" + END,
        " & ".join(heads) + END, "\\hline",
    ]  # fmt: skip
    lines += ["\\endfirsthead", *lines[1:], "\\endhead"]  # head repeated on every page
    for s in S_VALUES:
        for set_ in SETS:
            lines.append("\\multicolumn{15}{l}{$s=" + str(s) + "$, " + set_ + "}" + END)
            for arm, estimator in main_rows:
                key, name = _main_label(arm, estimator)
                cells = []
                for n in (2000, 4000):
                    r = S[(S["arm"] == key) & (S["set"] == set_) & (S["s"] == s) & (S["n"] == n)]
                    r = r.iloc[0]
                    cells += [
                        f"{_fmt(r.get('bias'))} ({_fmt(r.get('b_pred'))})",
                        f"{_fmt(r.get('root_n_sd'), 2)} ({_fmt(r.get('root_n_sd_pred'), 2)})",
                        _fmt(r.get("se_ratio"), 2),
                        f"{_fmt(r.get('coverage'))} ({_fmt(r.get('c_pred'))})",
                        _fmt(r.get("ci_length_mean")),
                        f"{100 * r['failure_rate']:.1f}",
                        _fmt(r.get("max_weight_median"), 1),
                    ]
                lines.append(" & ".join([name] + cells) + END)
    out["tab_E12_main"] = "\n".join(lines + ["\\hline", "\\end{tabular}"]) + "\n"

    full_cols = [
        "arm", "s", "set", "n", "bias", "b_pred", "root_n_sd", "root_n_sd_pred", "se_ratio",
        "coverage", "c_pred", "coverage_conditional", "ci_length_mean", "failure_rate",
        "failure_rate_cp_upper",
        "max_weight_median", "ess_median", "eval_imbalance_max",
    ]  # fmt: skip
    full = S[[c for c in full_cols if c in S.columns]]
    out["tab_E12_full"] = _table(
        full,
        list(full.columns),
        [_tex(c) for c in full.columns],
        "llll" + "r" * (len(full.columns) - 4),
    )

    bp = S[S["arm"].str.startswith("BP")].copy()

    def counts(c):
        return "" if c != c else ", ".join(f"{k}: {v}" for k, v in json.loads(c).items())

    bp["statuses"] = bp["status_counts"].map(counts)
    for c in ("wp_verdicts", "wp_solver_statuses"):
        bp[c] = bp[c].map(counts) if c in bp else ""
    for c in ("wp_n_at_boundary_max", "wp_gap_max", "wp_min_margin_min"):
        if c not in bp:
            bp[c] = np.nan
    out["tab_E12_bp"] = _table(
        bp,
        ["arm", "s", "set", "n", "failure_rate", "statuses", "wp_verdicts", "wp_n_at_boundary_max",
         "wp_gap_max", "wp_min_margin_min", "wp_solver_statuses"],
        ["Arm", "$s$", "Set", "$n$", "Failure rate", "Statuses", "Weight program",
         "At boundary (max)", "Gap (max)", "Margin (min)", "Solver"],
        "llllrlllrrl",
    )  # fmt: skip

    ordr = json.loads(S["ord"].iloc[0])
    lines = [
        "\\begin{tabular}{lrrr}", "\\hline",
        "Pair & Median difference & Lower 95\\% & Upper 95\\%" + END, "\\hline",
    ]  # fmt: skip
    for p in ordr["pairs"]:
        lines.append(
            " & ".join(
                [_tex(p["pair"]), _fmt(p["median_diff"]), _fmt(p["ci"][0]), _fmt(p["ci"][1])]
            )
            + END
        )
    lines += [
        "\\hline",
        "\\multicolumn{4}{l}{Share of replications with all four arms successful: "
        + _fmt(ordr["share_all_succeeded"]) + "}" + END,
        "\\multicolumn{4}{l}{Verdict: " + ordr["verdict"]
        + " (conditional on all four arms succeeding)}" + END,
        "\\hline", "\\end{tabular}",
    ]  # fmt: skip
    out["tab_E12_ord"] = "\n".join(lines) + "\n"

    fv = json.loads(S["family_verdicts"].iloc[0])
    unavailable = json.loads(S["unavailable_tests"].iloc[0])

    def verdict_word(k, v):
        if v is None:
            return "none"
        if v.startswith(k + ": negative"):
            return "negative"
        if unavailable.get(k):
            return "no departure detected among the defined tests"
        return "no departure detected"

    macros = {
        "EXIIR": str(int(S["R"].max())),
        "EXIIOrdVerdict": ordr["verdict"],
        "EXIIOrdShare": f"{ordr['share_all_succeeded']:.3f}",
        **{
            "EXII" + k.replace("H12-", "").replace("-", "") + "Verdict": verdict_word(k, v)
            for k, v in fv.items()
        },
        **{
            "EXII" + k.replace("H12-", "").replace("-", "") + "Undefined": str(
                len(unavailable.get(k, []))
            )
            for k in fv
        },
    }
    return out, macros
