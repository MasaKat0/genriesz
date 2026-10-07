"""E-20: two-sample inference under covariate shift (H20, ``thm:cs_asymp_normality``,
``rem:cs_no_ratio``).

Registration: doc/2026-10-07_experiment_registration.md, E-20, §1.3 A-5 (the strict
two-sample density-ratio path) and §1.4–§1.9. This module holds the DGP, the arms,
the Stage 0 population computation, the replication function (importable for
process parallelism) and the aggregation shared by the pilot and the confirmatory
stage. The notebook ``17_E20_covariate_shift_two_sample.ipynb`` runs the stages and
writes the outputs; its last cell makes the tables from ``summary.csv`` only (§4).

DGP20. Source ``Z ~ Unif(-sqrt3, sqrt3)^2``; target: the source law tilted by
``exp(t'z)``, ``t = (0.6, -0.4)``, drawn by rejection; ``r0 = exp(t'z) / M`` with
``M = E_0 exp(t'Z)``; ``Y = gamma0(Z) + eps``, ``eps ~ N(0, 1)`` (source only);
``gamma0 = sin(pi z1 / sqrt3) + 0.5 z2^2 + 0.3 z1 z2``; ``theta0 = E_1 gamma0``.

Arms (all cross-fitted with ``K = 5`` paired folds, Equation ``eq:cs_crossfit``):

- ``UKL+OLS`` (theory arm): ``r_hat`` UKL(C=0) by the strict path of
  ``fit_density_ratio`` on ``(1, z1, z2)`` with ``lambda = 0`` (``log r0`` is in
  the span, so the ratio model is correct); ``gamma_hat`` OLS on
  ``(1, z1, z2, sin(pi z1/sqrt3), z2^2, z1 z2)`` (``gamma0`` is in the span);
- ``UKL+RFF`` (illustration): the same ``r_hat``; ``gamma_hat`` ridge on random
  Fourier features;
- ``SQ+OLS`` (illustration): ``r_hat`` SQ (LSIF-type, linear link) by the strict
  path on ``(1, z1, z2)`` with ``lambda = 0`` (misspecified ratio); ``gamma_hat`` OLS.

Variance estimators (all arms): ``V_2s`` (the theorem, ``sigma_S^2/n + sigma_T^2/m``),
``V_src`` (the source term only) and ``V_pool`` (the single-sample influence
function of the pooled sample with the true share ``pi = m/(n+m)``).
"""

from __future__ import annotations

import collections
import itertools
import json

import numpy as np

from . import baselines, metrics, population
from .seeds import Seeds, fold_ids

EXP = 20
SQRT3 = np.sqrt(3.0)
TILT = np.array([0.6, -0.4])
#: registered (n, m) configurations, in the registered order
CELLS = [[500, 500], [2000, 2000], [8000, 8000], [8000, 1000], [1000, 8000]]
R = 2000
K = 5
ARMS = ("UKL+OLS", "UKL+RFF", "SQ+OLS")
THEORY_ARM = "UKL+OLS"
VARIANCES = ("V_2s", "V_src", "V_pool")
ARM_LABELS = [f"{a}|{v}" for a in ARMS for v in VARIANCES]
#: registered value of E_0 r0^2 (six significant digits)
E_R0_SQ = 1.54378
DGP_TOL = 5e-6  # half a unit in the last registered digit
#: §1.9 certificate (i)-(iii)
CERT_GRADIENT = 1e-12
CERT_BETA_DIFF = 1e-10
CERT_MARGIN = 1e-4
#: §1.3 C, lambda = 0 and no constraint: max_j |grad_j| <= 1e-8 max(1, ||mean phi(target)||_inf)
#: (genriesz scales ``solver_tol`` by that factor; its default would be 1e-10)
SAMPLE_TOL = 1e-8
#: H20 (a)-(c): |c - c_pred| <= 0.02 (§1.8, exact binomial)
COVERAGE_TOL = 0.02
#: H20 (a) only uses the configurations with min(n, m) >= 1000
H20A_MIN = 1000
#: random Fourier features of the UKL+RFF arm (illustration; not registered in detail)
RFF_FEATURES = 100
RFF_LENGTHSCALE = 1.0
RFF_RIDGE = 1.0


def gr():
    import genriesz

    return genriesz


def positive(X):
    """A density ratio is positive: every row takes the positive branch."""
    return np.ones(len(np.atleast_2d(X)), dtype=int)


positive.vectorized = True


def ratio_generator(arm):
    g = gr()
    if arm.startswith("UKL"):
        return g.UKLGenerator(C=0.0, branch_fn=positive)
    return g.SquaredGenerator(C=0.0)


def ratio_features(Z):
    """The density-ratio dictionary (1, z1, z2)."""
    Z = np.atleast_2d(Z)
    return np.column_stack([np.ones(len(Z)), Z[:, 0], Z[:, 1]])


def outcome_features(Z):
    """The OLS dictionary (1, z1, z2, sin(pi z1/sqrt3), z2^2, z1 z2); it contains gamma0."""
    Z = np.atleast_2d(Z)
    return np.column_stack(
        [
            np.ones(len(Z)), Z[:, 0], Z[:, 1], np.sin(np.pi * Z[:, 0] / SQRT3),
            Z[:, 1] ** 2, Z[:, 0] * Z[:, 1],
        ]
    )  # fmt: skip


def normaliser():
    """``M = E_0 exp(t'Z)`` for the uniform source on the box."""
    a = SQRT3 * TILT
    return float(np.prod(np.sinh(a) / a))


def r0(Z):
    Z = np.atleast_2d(Z)
    return np.exp(Z @ TILT) / normaliser()


def gamma0(Z):
    Z = np.atleast_2d(Z)
    return np.sin(np.pi * Z[:, 0] / SQRT3) + 0.5 * Z[:, 1] ** 2 + 0.3 * Z[:, 0] * Z[:, 1]


def draw_source(rng, n):
    Z = rng.uniform(-SQRT3, SQRT3, size=(n, 2))
    Y = gamma0(Z) + rng.standard_normal(n)
    return Z, Y


def draw_target(rng, m):
    """Rejection sampling from the source law: accept with ``exp(t'z - max t'z)``."""
    top = SQRT3 * float(np.sum(np.abs(TILT)))
    out = []
    got = 0
    while got < m:
        Z = rng.uniform(-SQRT3, SQRT3, size=(2 * m, 2))
        keep = rng.uniform(size=2 * m) < np.exp(Z @ TILT - top)
        out.append(Z[keep])
        got += int(keep.sum())
    return np.vstack(out)[:m]


def draw(rng, n, m):
    """The source sample, then the target sample, from the stream-0 generator."""
    Z, Y = draw_source(rng, n)
    Zt = draw_target(rng, m)
    return Z, Y, Zt


def _finite_or_none(x):
    x = float(x)
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------- Stage 0


def quadrature(points):
    """Tensor Gauss-Legendre rows on the source box with the uniform probabilities."""
    x, w = np.polynomial.legendre.leggauss(points)
    z = SQRT3 * x
    wz = w / 2.0
    Z = np.array(list(itertools.product(z, z)))
    W = np.array([a * b for a, b in itertools.product(wz, wz)])
    return Z, W / W.sum()


def dgp_check(points=64):
    """``E_0 r0^2`` (closed form and quadrature) agrees with the registered value, and
    the quadrature reproduces ``E_0 r0 = 1``."""
    a = SQRT3 * TILT
    closed = float(np.prod(np.sinh(2.0 * a) / (2.0 * a)) / normaliser() ** 2)
    Z, w = quadrature(points)
    quad = float(w @ r0(Z) ** 2)
    mass = float(w @ r0(Z))
    return {
        "E_r0_sq_closed_form": closed,
        "E_r0_sq_quadrature": quad,
        "E_r0_sq_registered": E_R0_SQ,
        "E_r0": mass,
        "ok": bool(
            abs(closed - E_R0_SQ) <= DGP_TOL
            and abs(quad - closed) <= 1e-12
            and abs(mass - 1.0) <= 1e-12
        ),
    }


def population_ratio(points):
    """The population UKL(C=0) ratio fit on ``(1, z1, z2)``: the minimizer of
    ``E_0 g*(phi'beta) - E_1 phi'beta`` by the reviewed damped Newton (tol 1e-13).

    ``E_1 phi = E_0 r0 phi``, so ``M = r0 phi`` on the source rows.
    """
    gen = ratio_generator(THEORY_ARM)
    Z, w = quadrature(points)
    Phi = ratio_features(Z)
    M = r0(Z)[:, None] * Phi
    sol = population.solve(generator=gen, X=Z, w=w, Phi=Phi, M=M, offset=np.zeros(len(Z)))
    return sol


def population_moments(points=64):
    """``theta0 = E_1 gamma0``, ``sigma_S^2 = E_0 r0^2 Var(eps)`` (Var(eps) = 1) and
    ``sigma_T^2 = Var_1 gamma0`` by quadrature."""
    Z, w = quadrature(points)
    rr, gg = r0(Z), gamma0(Z)
    theta0 = float(w @ (rr * gg))
    return {
        "theta0": theta0,
        "sigma2_S": float(w @ rr**2),
        "sigma2_T": float(w @ (rr * gg**2)) - theta0**2,
    }


def coverage_predictions(theta0, sigma2_S, sigma2_T, n, m):
    """§1.9-style predictions for the three variance estimators at ``(n, m)``.

    The sampling variance of the estimator is ``s^2 = sigma_S^2/n + sigma_T^2/m``.
    V_2s estimates ``s^2`` (``c = 0.95``); V_src estimates ``sigma_S^2/n``
    (``c_src = 2 Phi(1.96 sqrt((sigma_S^2/n)/s^2)) - 1``); V_pool estimates
    ``(V^CS + theta0^2 (1-pi)/pi)/N`` with ``pi = m/N``
    (``c_pool = 2 Phi(1.96 sqrt(1 + theta0^2 (1-pi)/(pi V^CS))) - 1``).
    """
    from scipy import stats

    from .metrics import Z_95

    s2 = sigma2_S / n + sigma2_T / m
    pi = m / (n + m)
    v_cs = sigma2_S / (1.0 - pi) + sigma2_T / pi
    c_2s = float(2.0 * stats.norm.cdf(Z_95) - 1.0)
    c_src = float(2.0 * stats.norm.cdf(Z_95 * np.sqrt((sigma2_S / n) / s2)) - 1.0)
    c_pool = float(
        2.0 * stats.norm.cdf(Z_95 * np.sqrt(1.0 + theta0**2 * (1.0 - pi) / (pi * v_cs))) - 1.0
    )
    return {
        "pi": pi,
        "V_CS": v_cs,
        "sd": float(np.sqrt(s2)),
        "c": {"V_2s": c_2s, "V_src": c_src, "V_pool": c_pool},
    }


def stage0():
    """The population ratio at 48 and 64 points with its certificate (§1.9 (i)-(iv)),
    the population moments, and the predictions per configuration."""
    s48, s64 = population_ratio(48), population_ratio(64)
    beta_star = np.array([-np.log(normaliser()), *TILT])
    solved = s48.status == "ok" and s64.status == "ok"
    beta_diff = float(np.max(np.abs(s48.beta - s64.beta))) if solved else None
    checks = {
        "i_gradient": bool(
            solved and s48.max_gradient <= CERT_GRADIENT and s64.max_gradient <= CERT_GRADIENT
        ),
        "i_beta_diff": bool(solved and beta_diff <= CERT_BETA_DIFF),
        # UKL(C=0): g' ranges over the whole real line, so the dual coordinate has no
        # finite end and the margin is +inf over the whole support (iii)
        "iii_margin": bool(solved and s64.min_dual_margin >= CERT_MARGIN),
        # (ii) positive definite Hessian; (iv) the dual objective is convex
        "ii_iv_positive_definite": bool(solved and s64.min_hessian_eig > 0.0),
    }
    certified = bool(solved and all(checks.values()))
    mom = population_moments(64)
    mom48 = population_moments(48)
    predictions = {
        f"{n},{m}": coverage_predictions(mom["theta0"], mom["sigma2_S"], mom["sigma2_T"], n, m)
        for n, m in CELLS
    }
    return {
        "certified": certified,
        "checks": checks,
        "status_48": s48.status,
        "status_64": s64.status,
        "beta_64": [float(x) for x in s64.beta],
        "beta_closed_form": [float(x) for x in beta_star],
        "beta_closed_form_max_abs_diff": float(np.max(np.abs(s64.beta - beta_star))),
        "beta_diff_48_64": beta_diff,
        "max_gradient_48": _finite_or_none(s48.max_gradient),
        "max_gradient_64": _finite_or_none(s64.max_gradient),
        "min_hessian_eig_64": _finite_or_none(s64.min_hessian_eig),
        "support_margin": _finite_or_none(s64.min_dual_margin),
        "support_margin_kind": "unbounded" if np.isinf(s64.min_dual_margin) else "finite",
        **mom,
        "moments_48_max_abs_diff": float(max(abs(mom[k] - mom48[k]) for k in mom)),
        "predictions": predictions if certified else None,
    }


# ---------------------------------------------------------------- Stage 1 replication


def _failed(status, **extra):
    return {"estimate": float("nan"), "status": status, **extra}


def _ols(Z_fit, Y_fit):
    coef, *_ = np.linalg.lstsq(outcome_features(Z_fit), Y_fit, rcond=None)
    return lambda Z: outcome_features(Z) @ coef


def rff_frequencies(generator):
    """Random Fourier features of a Gaussian kernel (length scale 1), from stream 3."""
    W = generator.standard_normal((2, RFF_FEATURES)) / RFF_LENGTHSCALE
    b = generator.uniform(0.0, 2.0 * np.pi, RFF_FEATURES)
    return W, b


def _rff(Z_fit, Y_fit, freqs):
    """Ridge on ``(1, sqrt(2/D) cos(Z W + b))``; the intercept is not penalized."""
    W, b = freqs

    def feats(Z):
        Z = np.atleast_2d(Z)
        return np.column_stack([np.ones(len(Z)), np.sqrt(2.0 / RFF_FEATURES) * np.cos(Z @ W + b)])

    F = feats(Z_fit)
    pen = RFF_RIDGE * np.eye(F.shape[1])
    pen[0, 0] = 0.0
    coef = np.linalg.solve(F.T @ F + pen, F.T @ Y_fit)
    return lambda Z: feats(Z) @ coef


def fit_ratio(arm, Zs_fit, Zt_fit):
    """The strict two-sample fit (§1.3 A-5). Returns ``(status, result)``."""
    g = gr()
    res = g.fit_density_ratio(
        Zt_fit, Zs_fit, basis=g.CallableBasis(ratio_features), generator=ratio_generator(arm),
        penalty=None, lam=0.0, solver="auto", solver_tol=SAMPLE_TOL,
    )  # fmt: skip
    return str(res.status), res


class _Warnings:
    def __init__(self):
        self.fits, self.converged = [], True

    def run(self, stage, fold, fn):
        result, counts, converged = baselines.run_recording_warnings(fn)
        if counts:
            self.fits.append([int(fold), stage, counts])
        self.converged = self.converged and converged
        return result

    def count(self):
        return int(sum(sum(c.values()) for _, _, c in self.fits))


def _cross_fit(Zs, Y, Zt, fs, ft, ratio_arm, outcome, freqs):
    """One arm: per fold pair ``k``, the ratio on the complement of both folds and the
    outcome regression on the complement of the source fold (Equation
    ``eq:cs_crossfit``). Returns the record of the three variance estimators."""
    n, m = len(Y), len(Zt)
    resid_w = np.empty(n)
    gam_t = np.empty(m)
    r_eval = np.empty(n)
    detail, grads = [], []
    folds = []
    w = _Warnings()
    for k in range(K):
        tr_s, te_s, tr_t, te_t = fs != k, fs == k, ft != k, ft == k
        status, res = w.run(
            "ratio", k, lambda tr_s=tr_s, tr_t=tr_t: fit_ratio(ratio_arm, Zs[tr_s], Zt[tr_t])
        )
        if status == "ok" and not w.converged:
            status = "convergence_warning"
        detail.append([k, "ratio", status])
        if status != "ok":
            return status, detail, w
        scale = max(1.0, float(np.max(np.abs(ratio_features(Zt[tr_t]).mean(axis=0)))))
        grads.append(float(np.max(np.abs(res.fit.gradient))) / scale)
        rr, outside, nonfinite = res.classify(Zs[te_s])
        if np.any(outside):
            detail.append([k, "prediction", "domain_prediction"])
            return "domain_prediction", detail, w
        if np.any(nonfinite):
            detail.append([k, "prediction", "nonfinite"])
            return "nonfinite", detail, w
        if outcome == "OLS":
            pred = w.run("outcome", k, lambda tr_s=tr_s: _ols(Zs[tr_s], Y[tr_s]))
        else:
            pred = w.run("outcome", k, lambda tr_s=tr_s: _rff(Zs[tr_s], Y[tr_s], freqs))
        g_s, g_t = pred(Zs[te_s]), pred(Zt[te_t])
        resid_w[te_s] = rr * (Y[te_s] - g_s)
        gam_t[te_t] = g_t
        r_eval[te_s] = rr
        folds.append(float(np.mean(g_t) + np.mean(resid_w[te_s])))
    if not w.converged:
        return "convergence_warning", detail, w
    if not (np.all(np.isfinite(resid_w)) and np.all(np.isfinite(gam_t))):
        return "nonfinite", detail, w
    theta = float(np.mean(folds))
    s2_S = float(np.mean(resid_w**2))
    s2_T = float(np.mean((gam_t - theta) ** 2))
    pi = m / (n + m)
    psi = np.concatenate([gam_t / pi - theta, resid_w / (1.0 - pi) - theta])
    rec = {
        "estimate": theta,
        "se_V_2s": float(np.sqrt(s2_S / n + s2_T / m)),
        "se_V_src": float(np.sqrt(s2_S / n)),
        "se_V_pool": float(np.sqrt(np.mean(psi**2) / (n + m))),
        "sigma2_S_hat": s2_S,
        "sigma2_T_hat": s2_T,
        "max_abs_alpha": float(np.max(np.abs(r_eval))),
        "ess": metrics.ess(r_eval),
        "ratio_gradient_max": float(max(grads)),  # scaled as in §1.3 C
    }
    return rec, detail, w


def replicate(task):
    """One replication of one configuration: every arm. ``task = (cell, rep, entropy)``."""
    cell_index, rep, entropy = task
    n, m = CELLS[cell_index]
    seeds = Seeds(EXP, entropy=entropy)
    Zs, Y, Zt = draw(seeds.data(cell_index, rep), n, m)
    fg = seeds.folds(cell_index, rep)
    fs = fold_ids(n, K, fg)
    ft = fold_ids(m, K, fg)
    freqs = rff_frequencies(seeds.baseline(cell_index, rep))
    out = {}
    for arm in ARMS:
        ratio_arm, outcome = arm.split("+")
        rec, detail, w = _cross_fit(Zs, Y, Zt, fs, ft, ratio_arm, outcome, freqs)
        common = {
            "fold_status": json.dumps([[str(a) for a in d] for d in detail]),
            "n_warnings": w.count(),
            "warnings": json.dumps(w.fits),
        }
        out[arm] = (
            {**rec, "status": "ok", **common} if isinstance(rec, dict) else _failed(rec, **common)
        )
    return out


def tasks_for(entropy, reps, cells=None):
    cells = range(len(CELLS)) if cells is None else cells
    return [(ci, r, entropy) for ci in cells for r in range(reps)]


# ---------------------------------------------------------------- aggregation

NUMERIC_FIELDS = (
    "estimate", "se", "max_abs_alpha", "ess", "sigma2_S_hat", "sigma2_T_hat",
    "ratio_gradient_max", "n_warnings",
)  # fmt: skip
TEXT_FIELDS = ("status", "warnings", "fold_status")


def raw_frame(tasks, results):
    """One row per replication x arm x variance estimator (the estimate is shared by
    the three rows of an arm); failures are rows too (§1.7). The warnings of an arm
    are counted once, on its V_2s row."""
    import pandas as pd

    rows = []
    for (ci, rep, _), res in zip(tasks, results, strict=True):
        n, m = CELLS[ci]
        for arm in ARMS:
            r = res[arm]
            for v in VARIANCES:
                row = {"cell": ci, "n": n, "m": m, "rep": rep, "arm": f"{arm}|{v}"}
                vals = {**r, "se": r.get(f"se_{v}", np.nan)}
                if v != "V_2s":
                    vals["n_warnings"] = 0
                row.update({k: float(vals.get(k, np.nan)) for k in NUMERIC_FIELDS})
                row.update({k: str(vals.get(k, "")) for k in TEXT_FIELDS})
                rows.append(row)
    return pd.DataFrame(rows)


def total_warnings(raw):
    return int(raw["n_warnings"].sum())


def ratio_balance(raw):
    """§1.3 C for every successful ratio fit (deterministic; a violation stops the run):
    the largest absolute gradient of the empirical objective, divided by
    ``max(1, ||mean phi(target)||_inf)``, is at most 1e-8 in every fold."""
    ok = (raw["status"] == "ok") & raw["arm"].str.endswith("|V_2s")
    g = raw.loc[ok, "ratio_gradient_max"]
    out = {
        "records_checked": int(ok.sum()),
        "fits_checked": int(ok.sum()) * K,
        "scaled_gradient_max": float(g.max()) if len(g) else None,
        "tolerance": SAMPLE_TOL,
    }
    if len(g) and not g.max() <= SAMPLE_TOL:
        raise AssertionError(f"ratio fit above the §1.3 C tolerance: {json.dumps(out)}")
    return out


def summarise(raw, population_record):
    """§1.5 metrics per configuration x arm x variance estimator, in the registered
    order, with the Stage 0 predictions for the theory arm."""
    import pandas as pd

    from .e12 import POSITIVE_VARIANCE, moment_status

    pop = population_record
    theta0 = float(pop["theta0"])
    rows = []
    for ci, (n, m) in enumerate(CELLS):
        for label in ARM_LABELS:
            g = raw[(raw["cell"] == ci) & (raw["arm"] == label)].sort_values("rep")
            if g.empty:  # e.g. the per-cell aggregation of Stage 0.5
                continue
            ok = (g["status"] == "ok").to_numpy()
            est, se = g["estimate"].to_numpy(), g["se"].to_numpy()
            arm, _, var = label.partition("|")
            row = {"cell": ci, "n": n, "m": m, "arm": label, "estimator": arm, "variance": var}
            row.update(metrics.failure_rate(ok))
            row.update(metrics.coverage(est, se, ok, theta0))
            row["status_counts"] = json.dumps(
                dict(sorted(collections.Counter(g["status"]).items()))
            )
            row["warnings"] = int(g["n_warnings"].sum())
            x = est[ok]
            ms = row["moment_status"] = moment_status(x)
            if ok.sum() >= 1:
                row.update(metrics.ci_length(se, ok))
                row.update(metrics.max_weight_summary(g["max_abs_alpha"].to_numpy(), ok))
                row["ess_median"] = float(g.loc[ok, "ess"].median())
                row["ratio_gradient_max"] = float(g.loc[ok, "ratio_gradient_max"].max())
            if ok.sum() >= 2:
                row.update(metrics.bias(est, ok, theta0))
                row["rmse"] = metrics.rmse(est, ok, theta0)
                row["sd"] = float(x.std(ddof=1))
            if ms in POSITIVE_VARIANCE:
                row["se_ratio"] = metrics.se_ratio(est, se, ok)
            row["certified"] = bool(pop["certified"])
            if pop["certified"]:
                p = pop["predictions"][f"{n},{m}"]
                row["sd_pred"] = p["sd"]
                row["pi"] = p["pi"]
                if arm == THEORY_ARM:
                    row["c_pred"] = p["c"][var]
            rows.append(row)
    return pd.DataFrame(rows)


def family_h20(summ):
    """H20 (a)-(c), one family (Holm 0.01, §1.8), the theory arm, exact binomial tests
    of ``|c - c_pred| <= 0.02``: (a) V_2s in the configurations with
    ``min(n, m) >= 1000``; (b) V_src and (c) V_pool in every configuration."""
    from . import inference

    tests = {}
    th = summ[(summ["estimator"] == THEORY_ARM) & summ["c_pred"].notna()]
    for part, var in (("(a)", "V_2s"), ("(b)", "V_src"), ("(c)", "V_pool")):
        for _, r in th[th["variance"] == var].iterrows():
            if part == "(a)" and min(int(r["n"]), int(r["m"])) < H20A_MIN:
                continue
            key = f"{part}|{var}|n={int(r['n'])},m={int(r['m'])}|coverage"
            tests[key] = inference.coverage_pvalue(
                int(r["covered"]), int(r["R"]), float(r["c_pred"]), tol=COVERAGE_TOL
            )
    verdict = inference.judge_family("H20", tests) if tests else None
    return verdict, tests


# ---------------------------------------------------------------- tables (summary.csv)

LABEL = {"UKL+OLS": "UKL + OLS", "UKL+RFF": "UKL + RFF ridge", "SQ+OLS": "SQ + OLS"}
VLABEL = {
    "V_2s": "$V_{\\mathrm{2s}}$",
    "V_src": "$V_{\\mathrm{src}}$",
    "V_pool": "$V_{\\mathrm{pool}}$",
}
END = " \\\\"


def _fmt(x, d=3):
    if x is None or x != x:
        return "--"
    s = f"{float(x):.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s  # no "-0.000"


def tables(S):
    """``tab_E20`` (a longtable body: the head is repeated by ``\\endfirsthead`` and
    ``\\endhead``; the manuscript supplies the caption) and ``tab_E20_status`` (failed
    replications by status, or one line saying there were none)."""
    heads = ["$(n,m)$", "Arm", "Bias", "SD (pred.)", "Fail \\%"]
    for v in VARIANCES:
        heads += [f"SE ratio {VLABEL[v]}", f"Cov. {VLABEL[v]} (pred.)", "Cov. succ."]
    head = ["\\hline", " & ".join(heads) + END, "\\hline"]
    lines = ["\\begin{tabular}{ll" + "r" * (len(heads) - 2) + "}", *head, "\\endfirsthead",
             *head, "\\endhead"]  # fmt: skip
    for ci, (n, m) in enumerate(CELLS):
        for arm in ARMS:
            rows = {v: S[(S["cell"] == ci) & (S["arm"] == f"{arm}|{v}")] for v in VARIANCES}
            if any(r.empty for r in rows.values()):
                continue
            r0_ = rows["V_2s"].iloc[0]
            cells = [
                f"$({n},{m})$", LABEL[arm], _fmt(r0_.get("bias"), 4),
                f"{_fmt(r0_.get('sd'), 4)} ({_fmt(r0_.get('sd_pred'), 4)})",
                f"{100 * r0_['failure_rate']:.1f}",
            ]  # fmt: skip
            for v in VARIANCES:
                r = rows[v].iloc[0]
                cells += [
                    _fmt(r.get("se_ratio"), 2),
                    f"{_fmt(r.get('coverage'))} ({_fmt(r.get('c_pred'))})",
                    _fmt(r.get("coverage_conditional")),
                ]
            lines.append(" & ".join(cells) + END)
    lines += ["\\hline", "\\end{tabular}"]
    status = ["\\begin{tabular}{llrl}", "\\hline",
              "$(n,m)$ & Arm & Successes & Statuses" + END, "\\hline"]  # fmt: skip
    any_failure = False
    for ci, (n, m) in enumerate(CELLS):
        for arm in ARMS:
            r = S[(S["cell"] == ci) & (S["arm"] == f"{arm}|V_2s")]
            if r.empty:
                continue
            r = r.iloc[0]
            if int(r["R_s"]) == int(r["R"]):
                continue
            any_failure = True
            counts = ", ".join(
                f"{k.replace('_', chr(92) + '_')}: {v}"
                for k, v in json.loads(r["status_counts"]).items()
            )
            status.append(
                " & ".join([f"$({n},{m})$", LABEL[arm], str(int(r["R_s"])), counts]) + END
            )
    if not any_failure:
        status.append("\\multicolumn{4}{l}{No replication failed in any arm.}" + END)
    status += ["\\hline", "\\end{tabular}"]
    fam = json.loads(S["family_verdict"].iloc[0]) if "family_verdict" in S else None
    return {"tab_E20": "\n".join(lines) + "\n", "tab_E20_status": "\n".join(status) + "\n"}, fam
