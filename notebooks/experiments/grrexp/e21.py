"""E-21: IHDP semi-synthetic data (illustration, I-type; statements S1 and S2).

Design: doc/2026-10-07_experiment_registration.md, E-21, §1.3, §1.4 (AutoDML-lasso,
TMLE), §1.5, §1.7 (data SHA-256) (parent repository). Under the author's decision
U28 (2026-10-08) the design document is not a binding registration; the choices
it leaves open are listed in the parent repository's
``docs/spec/02_data-and-experiments.md`` §2.1 ("E-21 の" rows). The notebook
``18_E21_ihdp.ipynb`` runs the stages.

Data: replications 1-100 of ``ihdp_npci_1-100`` (train 672 rows followed by test 75
rows, n = 747, 25 covariates; the first six are continuous and are standardized
within each replication). The true ATE of a replication is
``mean(mu1 - mu0)`` over the 747 rows, the true ATT is 4 in every replication
(checked).

Arms (ATE and ATT; one record per replication, estimand and arm):

- GRR: linear dictionary ``D(1, x)``, ``(1 - D)(1, x)`` (p = 52) and Gaussian RKHS
  (100 centers per arm, p = 202) x {SQ, UKL(1), BKL(1), BP(0.5, 0)} (ATE) or
  {SQ, UKL(0), BKL(0.05), BP(0.5, 0)} (ATT), ridge penalty chosen per outer
  training fold by inner 3-fold cross-validation of the held-out squared Riesz
  criterion; offsets ATE +-2, ATT treated 3, control -1. Three estimators from
  one fit: cross-fitted ARW, cross-fitted RW (``mean alpha_hat_{-k}(X) Y``) and
  cross-fitted TMLE (§1.4).
- EB: UKL(C = 0), lambda = 0, linear dictionary; each fold's weights are checked
  against an independent entropy-balancing dual Newton.
- LogitAIPW: l2 logistic propensity score with Cs chosen by 5-fold CV.
- AutoDML-lasso (§1.4, dictionary ``(1, D, x_1..x_25, D x_1..D x_25)``).
- DML-GBM: HistGradientBoosting propensity and outcome, hyper-parameters from the
  E-12 grid chosen per replication by 5-fold CV on the full sample.
- RA: ``mean m(W, gamma_hat_{-k})`` (reference; no standard error).

The outcome regression of every arm except DML-GBM is one Gaussian-RKHS ridge
regression per outer fold (penalty chosen by inner 3-fold CV), shared by all arms.
"""

from __future__ import annotations

import functools
import json
import math

import numpy as np

from . import baselines, e15, env, metrics
from .seeds import Seeds, fold_ids

EXP = 21
N = 747
CELLS = [[N]]
R = 100
K = 5
INNER = 3
N_CONT = 6  # the first six npci columns are continuous
P_COV = 25
LAM_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
RKHS_CENTERS = 100
ESTIMANDS = ("ATE", "ATT")
DICTS = ("lin", "rkhs")
GENERATORS = {"ATE": ("SQ", "UKL1", "BKL1", "BP"), "ATT": ("SQ", "UKL0", "BKL005", "BP")}
GRR_ESTIMATORS = ("ARW", "RW", "TMLE")
BASELINES = ("EB", "LogitAIPW", "AutoDML", "DMLGBM", "RA")
EB_TOL = 1e-6  # EB weights against the independent dual Newton (relative to max |alpha|)
EB_NEWTON_TOL = 1e-8  # §1.3 C scale, as for the GRR fit it checks
LAMBDA0_TOL = 1e-8  # §1.3 C for the lambda = 0 EB fit
LOGIT_CS = tuple(float(c) for c in np.logspace(-4, 4, 9))
S2_TOL = 0.044  # S2: |c - 0.95| <= 0.044 (2 MCSE at R = 100)
S1_BOOT = 10_000
S1_FAMILY = 1  # stream-4 family id of the S1 paired bootstrap (§1.9 list)


def gr():
    import genriesz

    return genriesz


# ---------------------------------------------------------------- data


@functools.lru_cache(maxsize=1)
def _npci():
    """The concatenated npci arrays (train rows, then test rows); hashes are checked
    by the run recorder (§1.7) before any replication runs."""
    tr = np.load(env.DATA_DIR / "ihdp/ihdp_npci_1-100.train.npz")
    te = np.load(env.DATA_DIR / "ihdp/ihdp_npci_1-100.test.npz")
    return {k: np.concatenate([tr[k], te[k]], axis=0) for k in ("x", "t", "yf", "mu0", "mu1")}


def replication(rep):
    """``(D, Xcov, Y, ate, att)`` of npci replication ``rep + 1`` (rep = 0..99)."""
    d = _npci()
    x = np.array(d["x"][:, :, rep], dtype=float)
    cont = x[:, :N_CONT]
    x[:, :N_CONT] = (cont - cont.mean(axis=0)) / cont.std(axis=0)
    D = np.array(d["t"][:, rep], dtype=float)
    Y = np.array(d["yf"][:, rep], dtype=float)
    tau = d["mu1"][:, rep] - d["mu0"][:, rep]
    ate = float(np.mean(tau))
    att = float(np.mean(tau[D == 1]))
    return D, x, Y, ate, att


def truths(rep):
    _, _, _, ate, att = replication(rep)
    return {"ATE": ate, "ATT": att}


# ---------------------------------------------------------------- specification


def sign(A):
    return np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)


sign.vectorized = True  # one call on all rows (genriesz BregmanGenerator.branch_fn)


def generator(name):
    g = gr()
    return {
        "SQ": lambda: g.SquaredGenerator(C=0.0),
        "UKL1": lambda: g.UKLGenerator(C=1.0, branch_fn=sign),
        "UKL0": lambda: g.UKLGenerator(C=0.0, branch_fn=sign),
        "BKL1": lambda: g.BKLGenerator(C=1.0, branch_fn=sign),
        "BKL005": lambda: g.BKLGenerator(C=0.05, branch_fn=sign),
        "BP": lambda: g.BPGenerator(C=0.0, omega=0.5, branch_fn=sign),
    }[name]()


def alpha_ref(estimand):
    hi, lo = (2.0, -2.0) if estimand == "ATE" else (3.0, -1.0)

    def ref(X):
        return np.where(np.atleast_2d(X)[:, 0] == 1, hi, lo)

    return ref


def functional(estimand, pi_hat):
    g = gr()
    if estimand == "ATE":
        return g.ATEFunctional(0)
    return g.ATTFunctional(treatment_index=0, pi=pi_hat, pi_is_estimated=True)


def _lin_features(Z):
    Z = np.atleast_2d(Z)
    return np.column_stack([np.ones(len(Z)), Z])


def basis(dictionary, center_seed):
    g = gr()
    if dictionary == "lin":
        base = g.CallableBasis(_lin_features)
    else:
        base = g.GaussianRKHSBasis(n_centers=RKHS_CENTERS, sigma="auto", include_bias=True,
                                   standardize=True, random_state=int(center_seed))  # fmt: skip
    return g.TreatmentInteractionBasis(base_basis=base)


def model(estimand, dictionary, gen_name, lam, pi_hat, center_seed):
    g = gr()
    gen = generator(gen_name)
    return g.GRRGLM(
        basis=basis(dictionary, center_seed), generator=gen,
        functional=functional(estimand, pi_hat),
        penalty="l2" if lam > 0 else None, lam=float(lam),
        offset=g.offset_from_alpha(gen, alpha_ref(estimand)),
    )  # fmt: skip


def toggle(X, d):
    out = np.array(X, dtype=float, copy=True)
    out[:, 0] = d
    return out


def m_of(estimand, X, pred, pi_hat):
    """``m(W_i, f)`` for a function ``pred(rows)`` (ATE: f(1,x) - f(0,x); ATT: D(...)/pi)."""
    diff = pred(toggle(X, 1.0)) - pred(toggle(X, 0.0))
    return diff if estimand == "ATE" else X[:, 0] * diff / pi_hat


def center(estimand, theta, D, pi_hat):
    """The centring term of the influence function (ATT: ``theta D / pi_hat``, §1.4)."""
    return theta if estimand == "ATE" else theta * D / pi_hat


# ---------------------------------------------------------------- warnings and failures


def _failed(status, **extra):
    return {"estimate": float("nan"), "se": float("nan"), "status": status, **extra}


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

    def fields(self):
        n = sum(sum(c.values()) for _, c in self.fits)
        return {"n_warnings": int(n), "warnings": json.dumps({"fits": self.fits})}


# ---------------------------------------------------------------- outcome regression


def _ridge(Phi, y, lam, free):
    """``argmin mean (y - Phi b)^2 + lam ||b[~free]||^2`` (the arm intercepts are free)."""
    n, p = Phi.shape
    pen = np.full(p, lam)
    pen[free] = 0.0
    A = Phi.T @ Phi / n + np.diag(pen)
    return np.linalg.solve(A, Phi.T @ y / n)


def _outcome_basis(center_seed, X_fit):
    bas = basis("rkhs", center_seed)
    bas.fit(X_fit)
    p = bas.n_features
    free = np.zeros(p, dtype=bool)
    free[[0, p // 2]] = True  # D * 1 and (1 - D) * 1 (include_bias puts the 1 first)
    return bas, free


def outcome_fit(X_fit, Y_fit, inner, center_seed):
    """Gaussian-RKHS ridge per treatment arm; the penalty minimizes the inner
    3-fold held-out MSE over ``LAM_GRID`` (ties: the earlier grid point).
    Returns ``(predict, lam)``."""
    mse = np.zeros(len(LAM_GRID))
    for j in range(INNER):
        tr, va = inner != j, inner == j
        bas, free = _outcome_basis(center_seed, X_fit[tr])
        P_tr, P_va = np.asarray(bas(X_fit[tr])), np.asarray(bas(X_fit[va]))
        for i, lam in enumerate(LAM_GRID):
            b = _ridge(P_tr, Y_fit[tr], lam, free)
            mse[i] += np.sum((Y_fit[va] - P_va @ b) ** 2)
    lam = LAM_GRID[int(np.argmin(mse))]
    bas, free = _outcome_basis(center_seed, X_fit)
    coef = _ridge(np.asarray(bas(X_fit)), Y_fit, lam, free)
    return (lambda rows: np.asarray(bas(rows)) @ coef), lam


# ---------------------------------------------------------------- GRR fits


def _predict_checked(mdl, estimand, X_eval):
    """Representer values at ``X_eval`` and at the counterfactual rows of ``m``; a row
    outside the domain or a non-finite value is a status (§1.3 B)."""
    pts = np.vstack([X_eval, mdl.functional.evaluation_points(X_eval, mdl.basis)])
    _, outside, nonfinite = mdl.classify(pts)
    if np.any(nonfinite):
        return "nonfinite", None
    if np.any(outside):
        return "domain_prediction", None
    return "ok", (lambda rows: np.asarray(mdl.classify(rows)[0], dtype=float))


def riesz_criterion(estimand, X_va, pred, pi_hat):
    """Held-out squared Riesz criterion ``mean alpha^2 - 2 mean m(W, alpha)``."""
    a = pred(X_va)
    return float(np.mean(a**2) - 2.0 * np.mean(m_of(estimand, X_va, pred, pi_hat)))


def select_and_fit(estimand, dictionary, gen_name, X_fit, inner, pi_hat, center_seed, warn, tag):
    """Inner CV over ``LAM_GRID`` then the refit on ``X_fit``.

    A candidate is valid when every inner fit succeeds without a convergence
    warning and its validation and counterfactual predictions are in the domain.
    No valid candidate: ``cv_failed``. Returns ``(status, model, lam, detail)``."""
    crit = np.full(len(LAM_GRID), np.inf)
    detail = []
    for i, lam in enumerate(LAM_GRID):
        total, ok = 0.0, True
        for j in range(INNER):
            tr, va = inner != j, inner == j
            mdl = model(estimand, dictionary, gen_name, lam, pi_hat, center_seed)
            fr = warn.run(f"{tag}|cv|{lam:g}|{j}", lambda mdl=mdl, tr=tr: mdl.fit(X_fit[tr]))
            st = str(fr.status)
            if st == "ok" and not warn.last:
                st = "convergence_warning"
            if st == "ok":
                st, pred = _predict_checked(mdl, estimand, X_fit[va])
            if st != "ok":
                detail.append([f"{lam:g}", j, st])
                ok = False
                break
            total += riesz_criterion(estimand, X_fit[va], pred, pi_hat) * va.sum()
        if ok and np.isfinite(total):
            crit[i] = total
    if not np.any(np.isfinite(crit)):
        return "cv_failed", None, float("nan"), detail
    lam = LAM_GRID[int(np.argmin(crit))]
    mdl = model(estimand, dictionary, gen_name, lam, pi_hat, center_seed)
    fr = warn.run(f"{tag}|refit", lambda: mdl.fit(X_fit))
    st = str(fr.status)
    if st == "ok" and not warn.last:
        st = "convergence_warning"
    return st, (mdl if st == "ok" else None), lam, detail


def _eb_fit(estimand, X_fit, pi_hat, warn, tag):
    mdl = model(estimand, "lin", "UKL0", 0.0, pi_hat, 0)
    fr = warn.run(f"{tag}|eb", lambda: mdl.fit(X_fit, tol=LAMBDA0_TOL))
    st = str(fr.status)
    if st == "ok" and not warn.last:
        st = "convergence_warning"
    return st, (mdl if st == "ok" else None)


def eb_check(estimand, mdl, X_fit, pi_hat):
    """Largest relative difference between the UKL(C = 0) weights and an independent
    entropy-balancing dual Newton on the same sample (``None`` if the Newton fails).

    ATE: each arm's weights match the full-sample mean of ``(1, x)``. ATT: the treated
    weights are ``1/pi_hat``; the control weights match ``sum_treated (1, x) / pi_hat``."""
    a = np.asarray(mdl.predict_alpha(X_fit), dtype=float)
    D = X_fit[:, 0]
    F = _lin_features(X_fit[:, 1:])
    n = len(D)
    if estimand == "ATE":
        arms = ((D == 1, F.sum(axis=0), 1.0), (D == 0, F.sum(axis=0), -1.0))
    else:
        arms = ((D == 0, F[D == 1].sum(axis=0) / pi_hat, -1.0),)
    worst = 0.0
    for rows, target, s in arms:
        w = baselines.eb_dual_newton(F[rows], target, n, tol=EB_NEWTON_TOL)
        if w is None:
            return None
        worst = max(worst, float(np.max(np.abs(s * a[rows] - w))))
    if estimand == "ATT":
        worst = max(worst, float(np.max(np.abs(a[D == 1] - 1.0 / pi_hat))))
    return worst / float(np.max(np.abs(a)))


# ---------------------------------------------------------------- baselines


def _logit_e(X_fit, X_eval, cv, warn, tag):
    from sklearn.linear_model import LogisticRegressionCV

    clf = LogisticRegressionCV(Cs=list(LOGIT_CS), cv=cv, penalty="l2", solver="lbfgs",
                               max_iter=10000, tol=1e-8, scoring="neg_log_loss")  # fmt: skip
    warn.run(f"{tag}|logit", lambda: clf.fit(X_fit[:, 1:], X_fit[:, 0]))
    return clf.predict_proba(X_eval[:, 1:])[:, 1]


def _ipw_alpha(estimand, D, e, pi_hat):
    if estimand == "ATE":
        return D / e - (1 - D) / (1 - e)
    return D / pi_hat - (1 - D) * e / (pi_hat * (1 - e))


def autodml_dictionary(estimand, X, pi_hat):
    """``b = (1, D, x_1..x_25, D x_1..D x_25)`` and ``m(W, b)`` (§1.4 order)."""
    D, Z = X[:, 0], X[:, 1:]
    n = len(D)
    B = np.column_stack([np.ones(n), D, Z, D[:, None] * Z])
    Mb = np.column_stack([np.zeros(n), np.ones(n), np.zeros_like(Z), Z])
    if estimand == "ATT":
        Mb = D[:, None] * Mb / pi_hat
    return B, Mb


def autodml_fit(estimand, X_fit, pi_hat):
    """CNS A.1.1 / A.2 (§1.4) on one training sample: ``(status, rho, info)``.

    A dictionary column that vanishes on the training sample (``G_jj = 0``, e.g.
    ``D x_j`` when no treated unit has ``x_j = 1``) leaves ``-2 rho_j M_j +
    2 thr_j |rho_j|`` in the objective: its coefficient is 0 when ``|M_j| <= thr_j``,
    and the objective is unbounded below otherwise (``degenerate_functional``). The
    coordinate descent runs on the other columns."""
    B, Mb = autodml_dictionary(estimand, X_fit, pi_hat)
    n, p = B.shape
    G = B.T @ B / n
    Mh = Mb.mean(axis=0)
    D = X_fit[:, 0]
    if not (np.any(D == 1) and np.any(D == 0)):
        return "degenerate_functional", None, {}
    act = np.diag(G) > 0
    low = max(2, math.ceil(p / 40))
    rho = np.zeros(p)
    rho[:low] = np.linalg.solve(G[:low, :low], Mh[:low])
    c1, c2, c3 = e15.ADML_C
    from scipy import stats

    r_l = c1 / math.sqrt(n) * float(stats.norm.ppf(1.0 - c2 / (2.0 * p)))
    inner_capped, sweeps_total, outer = 0, 0, 0
    old = rho.copy()
    for outer in range(1, e15.ADML_OUTER + 1):  # noqa: B007 (reported below)
        resid = B @ rho
        Dj = np.sqrt(np.mean((B * resid[:, None] - Mb) ** 2, axis=0)) + e15.ADML_LOADING_SHIFT
        thr = r_l * Dj
        thr[0] = r_l * c3 * Dj[0]
        if np.any(np.abs(Mh[~act]) > thr[~act]):
            return "degenerate_functional", None, {"zero_columns": np.flatnonzero(~act).tolist()}
        old = rho.copy()
        sub, sweep, converged = e15.coordinate_descent(
            G[np.ix_(act, act)], Mh[act], thr[act], rho[act]
        )
        rho = np.zeros(p)
        rho[act] = sub
        sweeps_total += sweep
        inner_capped += int(not converged)
        if not np.all(np.isfinite(rho)):
            return "nonfinite", None, {}
        if np.max(np.abs(rho - old)) <= e15.ADML_OUTER_TOL:
            break
    capped = bool(np.max(np.abs(rho - old)) > e15.ADML_OUTER_TOL)
    info = {"outer": outer, "outer_capped": capped, "inner_capped": inner_capped,
            "sweeps": sweeps_total, "zero_columns": np.flatnonzero(~act).tolist()}  # fmt: skip
    return "ok", rho, info


# ---------------------------------------------------------------- one replication


def _score_record(estimand, psi_parts, alpha, D, pi_hat, extra):
    """``theta`` and the §1.4 standard error from ``m + alpha (Y - gamma)`` per row."""
    theta = float(np.mean(psi_parts))
    psi = psi_parts - center(estimand, theta, D, pi_hat)
    if not (np.isfinite(theta) and np.all(np.isfinite(psi))):
        return _failed("nonfinite", **extra)
    return {"estimate": theta, "se": float(np.std(psi) / np.sqrt(len(psi))), "status": "ok",
            "max_abs_alpha": float(np.max(np.abs(alpha))), "ess": metrics.ess(alpha),
            **extra}  # fmt: skip


def replicate(task):
    """One IHDP replication: every estimand and arm. ``task = (cell, rep, entropy)``."""
    cell, rep, entropy = task
    seeds = Seeds(EXP, entropy=entropy)
    D, Z, Y, _, _ = replication(rep)
    X = np.column_stack([D, Z])
    n = len(Y)
    pi_hat = float(np.mean(D))
    folds = fold_ids(n, K, seeds.folds(cell, rep))
    rng_cv = seeds.inner_cv(cell, rep)
    # stream 2, in this order: the inner split of each outer training fold, the
    # LogitAIPW CV split of each outer training fold, the DML-GBM tuning split
    inner = {int(k): fold_ids(int((folds != k).sum()), INNER, rng_cv) for k in range(K)}
    logit_cv = {int(k): fold_ids(int((folds != k).sum()), 5, rng_cv) for k in range(K)}
    gbm_cv = fold_ids(n, 5, rng_cv)
    rng_b = seeds.baseline(cell, rep)
    # stream 3, in this order: RKHS center seed of the Riesz fits, of the outcome
    # regression, the GBM random_state
    riesz_seed, outcome_seed, gbm_seed = (int(v) for v in rng_b.integers(0, 2**31 - 1, size=3))

    # common outcome regression per outer fold
    warn_out = _Warn()
    gamma, lam_out = {}, []
    for k in range(K):
        tr = folds != k
        pred, lam = warn_out.run(
            f"{k}|outcome", lambda tr=tr, k=k: outcome_fit(X[tr], Y[tr], inner[k], outcome_seed)
        )
        gamma[k] = pred
        lam_out.append(lam)
    out_ok = warn_out.converged

    out = {}
    for estimand in ESTIMANDS:
        out.update(_grr_arms(estimand, X, Y, D, folds, inner, gamma, pi_hat, riesz_seed, out_ok))
        out.update(_baseline_arms(estimand, X, Y, D, folds, inner, logit_cv, gbm_cv, gamma,
                                  pi_hat, gbm_seed, out_ok))  # fmt: skip
    out["_outcome"] = {"lam": json.dumps(lam_out), **warn_out.fields()}
    return out


def _grr_arms(estimand, X, Y, D, folds, inner, gamma, pi_hat, center_seed, out_ok):
    n = len(Y)
    out = {}
    specs = [(dic, gen) for dic in DICTS for gen in GENERATORS[estimand]] + [("lin", "EB")]
    for dic, gen in specs:
        warn = _Warn()
        base = f"{estimand}|{dic}-{gen}" if gen != "EB" else f"{estimand}|EB"
        m_g, resid_g, alpha, m_a = (np.empty(n) for _ in range(4))
        detail, lams, eb_diff = [], [], []
        status = "ok"
        for k in range(K):
            tr, te = folds != k, folds == k
            Dt = D[tr]
            if not (np.any(Dt == 1) and np.any(Dt == 0)):
                status = "degenerate_functional"
                detail.append([k, "riesz", status])
                break
            if gen == "EB":
                st, mdl = _eb_fit(estimand, X[tr], pi_hat, warn, f"{base}|{k}")
                lam = 0.0
            else:
                st, mdl, lam, cvd = select_and_fit(estimand, dic, gen, X[tr], inner[k], pi_hat,
                                                   center_seed, warn, f"{base}|{k}")  # fmt: skip
                if cvd:
                    detail.append([k, "cv", cvd])
            detail.append([k, "riesz", st])
            if st != "ok":
                status = st
                break
            if gen == "EB":
                diff = eb_check(estimand, mdl, X[tr], pi_hat)
                if diff is None or not diff <= EB_TOL:
                    raise AssertionError(
                        f"EB weights differ from the dual Newton: {diff} ({base}, {k})"
                    )
                eb_diff.append(diff)
            st, pred = _predict_checked(mdl, estimand, X[te])
            if st != "ok":
                status = st
                detail.append([k, "prediction", st])
                break
            lams.append(lam)
            g = gamma[k]
            m_g[te] = m_of(estimand, X[te], g, pi_hat)
            resid_g[te] = Y[te] - g(X[te])
            alpha[te] = pred(X[te])
            m_a[te] = m_of(estimand, X[te], pred, pi_hat)
        extra = {"fold_status": json.dumps(detail), **warn.fields()}
        if status == "ok" and not out_ok:
            status = "convergence_warning"
        if status != "ok":
            for est in ("ARW", "RW", "TMLE") if gen != "EB" else ("ARW",):
                out[f"{base}|{est}"] = _failed(status, **extra)
            continue
        extra["lam"] = float(np.mean(lams))
        if eb_diff:
            extra["eb_diff"] = float(max(eb_diff))
        out[f"{base}|ARW"] = _score_record(estimand, m_g + alpha * resid_g, alpha, D, pi_hat, extra)
        if gen == "EB":
            continue
        # RW: mean alpha_hat_{-k}(X) Y; standard error from the same score terms (§1.4)
        theta_rw = float(np.mean(alpha * Y))
        psi_rw = m_g + alpha * resid_g - center(estimand, theta_rw, D, pi_hat)
        if not (np.isfinite(theta_rw) and np.all(np.isfinite(psi_rw))):
            out[f"{base}|RW"] = _failed("nonfinite", **extra)
        else:
            out[f"{base}|RW"] = {"estimate": theta_rw, "se": float(np.std(psi_rw) / np.sqrt(n)),
                                 "status": "ok", "max_abs_alpha": float(np.max(np.abs(alpha))),
                                 "ess": metrics.ess(alpha), **extra}  # fmt: skip
        # TMLE (§1.4): linear fluctuation, epsilon pooled over the folds
        eps = float(np.sum(alpha * resid_g) / np.sum(alpha**2))
        parts = m_g + eps * m_a + alpha * (resid_g - eps * alpha)
        out[f"{base}|TMLE"] = _score_record(estimand, parts, alpha, D, pi_hat,
                                            {**extra, "epsilon": eps})  # fmt: skip
    return out


def _baseline_arms(
    estimand, X, Y, D, folds, inner, logit_cv, gbm_cv, gamma, pi_hat, gbm_seed, out_ok
):
    n = len(Y)
    out = {}
    m_g, resid_g = np.empty(n), np.empty(n)
    for k in range(K):
        te = folds == k
        m_g[te] = m_of(estimand, X[te], gamma[k], pi_hat)
        resid_g[te] = Y[te] - gamma[k](X[te])

    # RA (reference: no standard error)
    ra = float(np.mean(m_g))
    out[f"{estimand}|RA"] = (
        {"estimate": ra, "se": float("nan"), "status": "ok"}
        if out_ok and np.isfinite(ra)
        else _failed("convergence_warning" if not out_ok else "nonfinite")
    )

    # LogitAIPW: l2 logistic propensity (CV over Cs), common outcome regression
    warn = _Warn()
    e = np.empty(n)
    for k in range(K):
        tr, te = folds != k, folds == k
        cv = [
            (np.flatnonzero(logit_cv[k] != j), np.flatnonzero(logit_cv[k] == j)) for j in range(5)
        ]
        e[te] = _logit_e(X[tr], X[te], cv, warn, str(k))
    extra = warn.fields()
    if not warn.converged:
        out[f"{estimand}|LogitAIPW"] = _failed("convergence_warning", **extra)
    elif not out_ok:
        out[f"{estimand}|LogitAIPW"] = _failed("convergence_warning", **extra)
    elif not np.all((e > 1e-12) & (e < 1 - 1e-12)):
        out[f"{estimand}|LogitAIPW"] = _failed("propensity_range", **extra)
    else:
        a = _ipw_alpha(estimand, D, e, pi_hat)
        out[f"{estimand}|LogitAIPW"] = _score_record(estimand, m_g + a * resid_g, a, D, pi_hat,
                                                     extra)  # fmt: skip

    # AutoDML-lasso (§1.4), common outcome regression
    warn = _Warn()
    alpha = np.empty(n)
    info, status = [], "ok"
    for k in range(K):
        tr, te = folds != k, folds == k
        st, rho, inf = warn.run(f"{k}|autodml", lambda tr=tr: autodml_fit(estimand, X[tr], pi_hat))
        if st != "ok":
            status = st
            break
        info.append({"fold": k, **inf})
        Bt, _ = autodml_dictionary(estimand, X[te], pi_hat)
        alpha[te] = Bt @ rho
    extra = {"adml_folds": json.dumps(info), **warn.fields()}
    if status == "ok" and not warn.converged:
        status = "convergence_warning"
    if status == "ok" and not out_ok:
        status = "convergence_warning"
    out[f"{estimand}|AutoDML"] = (_failed(status, **extra) if status != "ok" else
                                  _score_record(estimand, m_g + alpha * resid_g, alpha, D, pi_hat,
                                                extra))  # fmt: skip

    # DML-GBM: GBM propensity and outcome; hyper-parameters by 5-fold CV on the full sample
    warn = _Warn()
    Z = X[:, 1:]
    prop, outp = warn.run("tune", lambda: baselines.tune_gbm(D, Z, Y, gbm_cv, gbm_seed))
    e, mg, rg = np.empty(n), np.empty(n), np.empty(n)
    for k in range(K):
        tr, te = folds != k, folds == k

        def fit_k(tr=tr, te=te):
            clf = baselines._gbm_classifier(prop, gbm_seed).fit(Z[tr], D[tr])
            reg = baselines._gbm_regressor(outp, gbm_seed).fit(X[tr], Y[tr])
            return clf.predict_proba(Z[te])[:, 1], reg.predict

        e[te], reg_pred = warn.run(f"{k}|gbm", fit_k)
        mg[te] = m_of(estimand, X[te], reg_pred, pi_hat)
        rg[te] = Y[te] - reg_pred(X[te])
    extra = {"gbm_params": json.dumps({"propensity": prop, "outcome": outp}), **warn.fields()}
    if not warn.converged:
        out[f"{estimand}|DMLGBM"] = _failed("convergence_warning", **extra)
    elif not np.all((e > 0) & (e < 1)):
        out[f"{estimand}|DMLGBM"] = _failed("propensity_range", **extra)
    else:
        a = _ipw_alpha(estimand, D, e, pi_hat)
        out[f"{estimand}|DMLGBM"] = _score_record(estimand, mg + a * rg, a, D, pi_hat, extra)
    return out


def arm_labels():
    labels = []
    for estimand in ESTIMANDS:
        for dic in DICTS:
            for gen in GENERATORS[estimand]:
                labels += [f"{estimand}|{dic}-{gen}|{est}" for est in GRR_ESTIMATORS]
        labels += [f"{estimand}|EB|ARW"] + [f"{estimand}|{b}" for b in BASELINES if b != "EB"]
    return labels


ARM_LABELS = arm_labels()


def record_label(label):
    """The run recorder's form of an arm label (letters, digits, ``_`` and ``-``)."""
    return label.replace("|", "_")


def tasks_for(entropy, reps, cells=None):
    cells = range(len(CELLS)) if cells is None else cells
    return [(ci, r, entropy) for ci in cells for r in range(reps)]


# ---------------------------------------------------------------- frames and summaries

NUMERIC_FIELDS = ("estimate", "se", "max_abs_alpha", "ess", "lam", "eb_diff", "epsilon",
                  "n_warnings")  # fmt: skip
TEXT_FIELDS = ("status", "warnings", "fold_status", "adml_folds", "gbm_params")


def raw_frame(tasks, results):
    """One row per replication x arm (failures are rows too, §1.7). The outcome
    regression's warnings are counted once per replication on the row ``outcome``."""
    import pandas as pd

    rows = []
    for (ci, rep, _), res in zip(tasks, results, strict=True):
        tr = truths(rep)
        for arm in ARM_LABELS:
            r = res[arm]
            row = {"cell": ci, "rep": rep, "arm": arm, "theta0": tr[arm.split("|")[0]]}
            row.update({k: float(r.get(k, np.nan)) for k in NUMERIC_FIELDS})
            row.update({k: str(r.get(k, "")) for k in TEXT_FIELDS})
            rows.append(row)
        o = res["_outcome"]
        rows.append({"cell": ci, "rep": rep, "arm": "outcome", "theta0": np.nan,
                     **{k: np.nan for k in NUMERIC_FIELDS}, "n_warnings": float(o["n_warnings"]),
                     **{k: "" for k in TEXT_FIELDS}, "status": "ok", "warnings": o["warnings"],
                     "fold_status": o["lam"]})  # fmt: skip
    return pd.DataFrame(rows)


def total_warnings(raw):
    return int(raw["n_warnings"].fillna(0).sum())


def eb_summary(raw):
    eb = raw[raw["arm"].str.contains(r"\|EB\|", regex=True) & (raw["status"] == "ok")]
    return {"records_checked": int(len(eb)), "max_relative_difference":
            float(eb["eb_diff"].max()) if len(eb) else None, "tolerance": EB_TOL}  # fmt: skip


def _arm_summary(g):
    ok = (g["status"] == "ok").to_numpy()
    err = (g["estimate"] - g["theta0"]).to_numpy()
    se = g["se"].to_numpy()
    out = metrics.failure_rate(ok, g["status"].to_numpy())
    out["status_counts"] = json.dumps(out["status_counts"])
    if ok.sum() >= 2:
        e = err[ok]
        out.update({"bias": float(e.mean()), "sd": float(np.std(e, ddof=1)),
                    "rmse": float(np.sqrt(np.mean(e**2)))})  # fmt: skip
    if ok.any() and np.all(np.isfinite(se[ok])):
        lo, hi = err - 1.959963984540054 * se, err + 1.959963984540054 * se
        cov = ok & (lo <= 0) & (0 <= hi)
        out.update({"coverage": float(cov.sum() / len(ok)),
                    "coverage_conditional": float(cov.sum() / ok.sum()),
                    "se_mean": float(np.mean(se[ok]))})  # fmt: skip
        if "sd" in out and out["sd"] > 0:
            out["se_ratio"] = out["se_mean"] / out["sd"]
    if ok.any() and g["max_abs_alpha"][ok].notna().all():
        out["max_weight_median"] = float(np.median(g["max_abs_alpha"][ok]))
        out["ess_median"] = float(np.median(g["ess"][ok]))
    return out


def summarise(raw):
    import pandas as pd

    rows = []
    for arm in ARM_LABELS:
        g = raw[raw["arm"] == arm].sort_values("rep")
        rows.append({"arm": arm, **_arm_summary(g)})
    summ = pd.DataFrame(rows)
    summ["s2"] = (summ["coverage"] - 0.95).abs() <= S2_TOL
    summ.loc[summ["coverage"].isna(), "s2"] = False
    return summ


def s1_intervals(raw, seeds):
    """S1: paired bootstrap (stream 4, family 1, 10^4 draws) of the ratio of the SD of the
    RW errors to that of the ARW errors over replications in which both succeed. The
    draws are made per GRR arm in ``ARM_LABELS`` order. Returns ``{arm: (ratio, lo, hi)}``;
    the interval is undefined (NaN) when fewer than three replications qualify or a
    resample has a zero ARW standard deviation, and S1 is then not written."""
    gen = seeds.bootstrap(S1_FAMILY)
    out = {}
    for estimand in ESTIMANDS:
        for dic in DICTS:
            for g in GENERATORS[estimand]:
                base = f"{estimand}|{dic}-{g}"
                rw = raw[raw["arm"] == f"{base}|RW"].sort_values("rep")
                arw = raw[raw["arm"] == f"{base}|ARW"].sort_values("rep")
                both = (rw["status"] == "ok").to_numpy() & (arw["status"] == "ok").to_numpy()
                if both.sum() < 3:
                    out[base] = (float("nan"), float("nan"), float("nan"))
                    continue
                e_rw = (rw["estimate"] - rw["theta0"]).to_numpy()[both]
                e_arw = (arw["estimate"] - arw["theta0"]).to_numpy()[both]
                ratio = float(np.std(e_rw, ddof=1) / np.std(e_arw, ddof=1))
                idx = gen.integers(0, len(e_rw), size=(S1_BOOT, len(e_rw)))
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.std(e_rw[idx], axis=1, ddof=1) / np.std(e_arw[idx], axis=1, ddof=1)
                undefined = int(np.sum(~np.isfinite(r)))
                if undefined:  # a resample with a constant ARW error; reported, never dropped
                    out[base] = (ratio, float("nan"), float("nan"))
                    continue
                lo, hi = (float(v) for v in np.quantile(r, [0.025, 0.975]))
                out[base] = (ratio, lo, hi)
    return out


# ---------------------------------------------------------------- tables

END = " \\\\"
GEN_LABEL = {"SQ": "SQ", "UKL1": "UKL($C=1$)", "UKL0": "UKL($C=0$)", "BKL1": "BKL($C=1$)",
             "BKL005": "BKL($C=0.05$)", "BP": "BP($0.5$, $C=0$)"}  # fmt: skip
DICT_LABEL = {"lin": "linear", "rkhs": "RKHS"}
BASE_LABEL = {"EB": "EB (= UKL, $C=0$, linear)", "LogitAIPW": "LogitAIPW",
              "AutoDML": "AutoDML-lasso", "DMLGBM": "DML-GBM", "RA": "RA (reference)"}  # fmt: skip


def _fmt(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{float(x):.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def _row_label(arm):
    parts = arm.split("|")
    if parts[1] in ("EB",):
        return BASE_LABEL["EB"]
    if parts[1] in BASE_LABEL:
        return BASE_LABEL[parts[1]]
    dic, gen = parts[1].split("-")
    return f"{GEN_LABEL[gen]}, {DICT_LABEL[dic]}, {parts[2]}"


def _table(S, estimand, s1):
    heads = ["Method", "Bias", "SD", "RMSE", "SE ratio", "Coverage", "Cov.\\ succ.", "Fail \\%",
             "med.\\ $\\max|\\widehat\\alpha|$", "SD ratio RW/ARW (95\\% CI)"]  # fmt: skip
    head = ["\\hline", " & ".join(heads) + END, "\\hline"]
    lines = ["\\begin{tabular}{lrrrrrrrrl}", *head, "\\endfirsthead", *head, "\\endhead"]
    for _, r in S[S["arm"].str.startswith(estimand + "|")].iterrows():
        arm = r["arm"]
        ratio = "--"
        parts = arm.split("|")
        key = f"{parts[0]}|{parts[1]}"
        if len(parts) == 3 and parts[2] == "RW" and key in s1:
            v, lo, hi = s1[key]
            ratio = f"{_fmt(v, 2)} ({_fmt(lo, 2)}, {_fmt(hi, 2)})"
        lines.append(" & ".join([
            _row_label(arm), _fmt(r.get("bias")), _fmt(r.get("sd")), _fmt(r.get("rmse")),
            _fmt(r.get("se_ratio"), 2), _fmt(r.get("coverage")),
            _fmt(r.get("coverage_conditional")),
            f"{100 * r['failure_rate']:.0f}", _fmt(r.get("max_weight_median"), 1), ratio,
        ]) + END)  # fmt: skip
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _status_table(S):
    lines = ["\\begin{tabular}{lrl}", "\\hline", "Method & Successes & Statuses" + END, "\\hline"]
    for _, r in S.iterrows():
        if int(r["R_s"]) == int(r["R"]):
            continue
        counts = ", ".join(f"{k.replace('_', chr(92) + '_')}: {v}"
                           for k, v in json.loads(r["status_counts"]).items())  # fmt: skip
        name = f"{r['arm'].split('|')[0]}: {_row_label(r['arm'])}"
        lines.append(f"{name} & {int(r['R_s'])} & {counts}" + END)
    if len(lines) == 4:
        lines.append("\\multicolumn{3}{l}{No replication failed in any arm.}" + END)
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def tables(S, s1):
    """Tables and macros from the summary only (§4). ``s1`` maps each GRR arm base to
    ``(ratio, lo, hi)``."""
    tabs = {"tab_E21_ate": _table(S, "ATE", s1), "tab_E21_att": _table(S, "ATT", s1),
            "tab_E21_status": _status_table(S)}  # fmt: skip
    s1_arms = [b for b, (_, lo, _) in s1.items() if np.isfinite(lo) and lo > 1.0]
    s2_arms = S.loc[S["s2"] & S["arm"].str.endswith("|ARW"), "arm"].tolist()
    macros = {
        "IHDPreps": str(int(S["R"].max())),
        "EXXIScountOne": str(len(s1_arms)),
        "EXXIScountTwo": str(len(s2_arms)),
        "EXXIGRRArms": str(len(s1)),
    }
    return tabs, macros, {"S1": s1_arms, "S2": s2_arms}
