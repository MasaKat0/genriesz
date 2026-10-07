"""E-22: the Lalonde job-training data (illustration, I-type; ATT of the NSW treated).

Registration: doc/2026-10-07_experiment_registration.md (parent repository), E-22, and
§1.1–§1.7. The design document is not binding (U28); the choices it leaves open are
listed in ``CHOICES`` and in the parent repository's spec 02 §2.1. This module holds the
data, the dictionaries, the arms, the replication function over one fold split
(importable for process parallelism), the full-sample fits (weights, SMD, the EB check
and the admissibility diagnostic) and the aggregation over the 100 fold splits. The
notebook ``19_E22_lalonde.ipynb`` runs the stages; its last cell makes the tables and
the figure from ``summary.csv`` only (§4).

Sample: 614 units, 185 NSW treated (``NSW*``) and 429 PSID comparisons (``PSID*``);
outcome ``re78``; the eight raw covariates standardized on the whole sample. Target:
the ATT of the NSW treated with ``pi1_hat = 185/614``.

Arms (each cross-fitted with ``K = 5`` folds on every one of the 100 fold splits):

- GRR: generators SQ (C=0), UKL (C=0), BP (omega=0.5, C=0), BKL (C=0.05), each with its
  compatible link, on three control-arm dictionaries (the treated arm has an intercept
  only): D1 the raw eight, D2 the Dehejia-Wahba specification, D3 a Gaussian RKHS with
  100 centers. D1 and D2 with lambda = 0; D3 with a ridge penalty chosen by the held-out
  squared Riesz criterion (inner 3-fold CV on each training fold). No cap or clipping.
  ARW and TMLE per arm. UKL (C=0) x D1 is entropy balancing for the ATT (reported as the
  EB row and checked against an independent dual Newton).
- LogitAIPW (logistic MLE on the D2 features) and DML-GBM (gradient boosting with nested
  CV), both with the same control-arm outcome regression as the GRR arms in LogitAIPW.
- The outcome regression of every GRR arm and of LogitAIPW: OLS on D2, per arm.

Aggregation over splits (registration E-22): the median ``theta_med`` of the estimates
of the successful splits and ``sigma2_med = median_s{sigma2_s + n (theta_s - theta_med)^2}``;
an arm that fails in more than half of the splits is reported as "no exact fit (k/100)".
"""

from __future__ import annotations

import collections
import json

import numpy as np

from . import baselines, env
from .seeds import Seeds, fold_ids

EXP = 22
DATA_FILE = "lalonde/lalonde.csv"
RAW = ("age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75")
N, N_TREATED = 614, 185
#: the registered cell inventory: one cell, the whole sample
CELLS = [[N]]
R = 100  # fold splits (registration: 100, stream 1, seeds 0-99)
K = 5
GENERATORS = ("SQ", "UKL", "BP", "BKL")
DICTIONARIES = ("D1", "D2", "D3")
GRR_ARMS = tuple(f"{g}-{d}" for g in GENERATORS for d in DICTIONARIES)
EB_ARM = "UKL-D1"
BASELINE_ARMS = ("LogitAIPW", "DML-GBM")
ESTIMATORS = ("ARW", "TMLE")
ARM_LABELS = [f"{a}|{e}" for a in GRR_ARMS for e in ESTIMATORS] + list(BASELINE_ARMS)
#: generator constants (registration E-22)
C_VALUE = {"SQ": 0.0, "UKL": 0.0, "BP": 0.0, "BKL": 0.05}
BP_OMEGA = 0.5
#: offset of the GRR fits: alpha_ref = 3 (treated), -1 (control), as E-21 for the ATT;
#: every arm has an intercept, so it only fixes the starting point
ALPHA_REF_TREATED, ALPHA_REF_CONTROL = 3.0, -1.0
#: §1.3 C (lambda = 0, no constraint), as E-12: genriesz scales the tolerance
SAMPLE_TOL = 1e-8
#: D3: Gaussian RKHS with 100 centers and the median-heuristic bandwidth of the
#: training sample; ridge penalty chosen from LAMBDA_GRID by the held-out squared
#: Riesz (LSIF) criterion with inner 3-fold CV (the rule of E-21)
RKHS_CENTERS = 100
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
INNER_FOLDS = 3
#: DML-GBM: nested CV on each outer training fold over the E-12 grid
GBM_INNER_FOLDS = 3
#: EB check: max |w_EB - w_UKL| / max(1, max w_EB) <= EB_TOL (registration: 1e-6)
EB_TOL = 1e-6
#: BKL with C = 0.05: the control weight e/(pi1 (1-e)) exceeds C iff e exceeds this
#: (PR4a 2(d)); 0.01484 in the registration
NSW_REFERENCE = 1794  # Dehejia and Wahba (1999), the NSW experimental estimate
#: Every choice that the design document leaves open (spec 02 §2.1 of the parent).
CHOICES = {
    "offset": "alpha_ref = 3 (treated), -1 (control), the ATT offset of E-21",
    "lambda_grid": "D3 ridge grid {1e-4, 1e-3, 1e-2, 1e-1, 1} (powers of ten)",
    "rkhs": "100 centers drawn from the training rows, sigma = median heuristic",
    "logit_features": "LogitAIPW propensity on the non-constant D2 features",
    "gbm_nested_cv": "DML-GBM: inner 3-fold CV on each outer training fold (splits from "
    "stream 2 after the D3 seeds; GBM random_state from stream 3)",
    "dml_gbm_outcome": "DML-GBM keeps gradient-boosted nuisances for both the propensity and "
    "the control outcome (its definition in E-12); the common OLS(D2) outcome regression is "
    "used by every GRR arm and by LogitAIPW",
    "smd_cf_aggregation": "cross-fit SMD per covariate: median over successful splits",
    "majority": "an arm is reported unless more than half of the splits fail (k >= 50 of 100)",
    "table_bodies": "tab_E22_full and tab_E22_smd are longtable bodies (\\endfirsthead/\\endhead "
    "after the head; the manuscript sets them with longtable and adds the caption), as E-12",
    "smd_b_dictionary": "cross-fit dictionary SMD for D3 is not defined (fold-specific)",
}


def gr():
    import genriesz

    return genriesz


def sign(X):
    """Treated rows take the positive branch, control rows the negative one."""
    return np.where(np.atleast_2d(X)[:, 0] == 1, 1, -1)


sign.vectorized = True


def generator(gen):
    g = gr()
    c = C_VALUE[gen]
    if gen == "SQ":
        return g.SquaredGenerator(C=c)
    if gen == "UKL":
        return g.UKLGenerator(C=c, branch_fn=sign)
    if gen == "BP":
        return g.BPGenerator(omega=BP_OMEGA, C=c, branch_fn=sign)
    return g.BKLGenerator(C=c, branch_fn=sign)


def alpha_ref(X):
    return np.where(np.atleast_2d(X)[:, 0] == 1, ALPHA_REF_TREATED, ALPHA_REF_CONTROL)


# ---------------------------------------------------------------- data


def load():
    """``(D, Y, Zs, raw)``: treatment, outcome, the standardized raw eight and the raw frame.

    The SHA-256 of the file is checked by the run recorder before this is called."""
    import pandas as pd

    df = pd.read_csv(env.DATA_DIR / DATA_FILE)
    if len(df) != N or int(df["treat"].sum()) != N_TREATED:
        raise ValueError("unexpected Lalonde sample")
    ids = df["ID"].astype(str)
    if not (ids[df["treat"] == 1].str.startswith("NSW").all()
            and ids[df["treat"] == 0].str.startswith("PSID").all()):  # fmt: skip
        raise ValueError("treated rows must be NSW and control rows PSID")
    if df[list(RAW) + ["re78"]].isna().any().any():
        raise ValueError("missing values (no imputation)")
    raw = df[list(RAW)].to_numpy(float)
    Zs = (raw - raw.mean(axis=0)) / raw.std(axis=0, ddof=0)
    return df["treat"].to_numpy(float), df["re78"].to_numpy(float), Zs, raw


def d1_features(Zs, raw):
    return np.asarray(Zs, float)


def d2_features(Zs, raw):
    """Dehejia-Wahba: age, age^2, age^3, educ, educ^2, married, nodegree, black, hispan,
    re74, re75, u74, u75, educ * re74 (from the standardized covariates; u74 and u75 are
    the indicators of zero earnings)."""
    Zs, raw = np.atleast_2d(Zs), np.atleast_2d(raw)
    c = {name: Zs[:, j] for j, name in enumerate(RAW)}
    u74 = (raw[:, RAW.index("re74")] == 0).astype(float)
    u75 = (raw[:, RAW.index("re75")] == 0).astype(float)
    return np.column_stack([
        c["age"], c["age"] ** 2, c["age"] ** 3, c["educ"], c["educ"] ** 2, c["married"],
        c["nodegree"], c["black"], c["hispan"], c["re74"], c["re75"], u74, u75,
        c["educ"] * c["re74"],
    ])  # fmt: skip


class ATTArmBasis:
    """``phi(X) = [D, (1-D) psi(Z)]`` with ``psi`` a control-arm dictionary that contains
    the constant: the treated arm has an intercept only (registration E-22).

    ``X = [D, Zs, u74, u75]`` (column 0 the treatment). ``psi`` is a fixed function
    (D1, D2) or a Gaussian RKHS basis fitted on the training rows (D3)."""

    def __init__(self, dictionary, *, random_state=None):
        self.dictionary = dictionary
        self.random_state = random_state
        self._rkhs = None
        if dictionary == "D3":
            self._rkhs = gr().GaussianRKHSBasis(
                n_centers=RKHS_CENTERS, sigma="auto", include_bias=True, standardize=False,
                random_state=random_state,
            )  # fmt: skip

    def copy(self):
        import copy

        return copy.deepcopy(self)

    def fit(self, X, y=None):
        if self._rkhs is not None:
            self._rkhs.fit(np.atleast_2d(X)[:, 1:9])
        return self

    def _psi(self, X):
        X = np.atleast_2d(X)
        Zs, u = X[:, 1:9], X[:, 9:11]
        if self.dictionary == "D1":
            return np.column_stack([np.ones(len(X)), Zs])
        if self.dictionary == "D2":
            raw_zero = np.column_stack([np.zeros((len(X), 6)), 1.0 - u])  # u = 1{re = 0}
            return np.column_stack([np.ones(len(X)), d2_features(Zs, raw_zero)])
        return np.asarray(self._rkhs(Zs), float)

    @property
    def n_features(self):
        """The treated intercept plus the control-arm dictionary with its constant."""
        if self.dictionary == "D1":
            return 1 + 1 + len(RAW)
        if self.dictionary == "D2":
            return 1 + 1 + 14
        return 1 + self._rkhs.n_features

    def __call__(self, X):
        X = np.atleast_2d(X)
        D = X[:, :1]
        return np.column_stack([D, (1.0 - D) * self._psi(X)])

    def control_features(self, X):
        """The dictionary functions of the control arm (without the constant)."""
        return self._psi(X)[:, 1:]


def design(D, Zs, raw):
    """``X = [D, Zs, u74, u75]``: everything the bases need, in one matrix."""
    u74 = (raw[:, RAW.index("re74")] == 0).astype(float)
    u75 = (raw[:, RAW.index("re75")] == 0).astype(float)
    return np.column_stack([D, Zs, u74, u75])


def outcome_basis():
    """OLS on D2 in each arm (the common outcome regression of every arm)."""
    g = gr()

    def f(Z):
        Z = np.atleast_2d(Z)
        raw_zero = np.column_stack([np.zeros((len(Z), 6)), 1.0 - Z[:, 8:10]])
        return np.column_stack([np.ones(len(Z)), d2_features(Z[:, :8], raw_zero)])

    return g.TreatmentInteractionBasis(base_basis=g.CallableBasis(f))


# ---------------------------------------------------------------- the arms


def _failed(status, **extra):
    return {"estimate": float("nan"), "se": float("nan"), "status": str(status), **extra}


def _seed_int(generator):
    return int(generator.integers(0, 2**31 - 1))


def admissibility_thresholds():
    """No screen: the held-out squared Riesz criterion alone selects lambda."""
    return {k: None for k in gr().model_selection.default_admissibility_thresholds()}


def _smd(Zc_t, Zc_c, w):
    """``(mean_T - sum_i w_i z_i) / sd_pooled`` with signed control weights ``w`` (summing to
    one) and the unweighted pooled SD (genriesz ``_covariate_balance_smd`` convention)."""
    sd = np.sqrt(0.5 * (np.var(Zc_t, axis=0, ddof=1) + np.var(Zc_c, axis=0, ddof=1)))
    sd = np.where(sd > 0, sd, np.nan)
    return (Zc_t.mean(axis=0) - w @ Zc_c) / sd


def control_weights(alpha_c):
    """``w_i = -alpha_i / sum_C(-alpha_j)`` (signed); NaN if the sum is not positive."""
    a = -np.asarray(alpha_c, float)
    s = float(np.sum(a))
    return a / s if np.isfinite(s) and s > 0 else np.full(len(a), np.nan)


def weight_summaries(D, Zs, X, alpha, basis_fitted):
    """SMD, ESS among the controls and the share of control weights with the wrong sign."""
    c, t = D == 0, D == 1
    w = control_weights(alpha[c])
    raw_smd = _smd(Zs[t], Zs[c], w)
    out = {
        "smd_raw_max": float(np.nanmax(np.abs(raw_smd))),
        "smd_raw": json.dumps([float(v) for v in raw_smd]),
        "ess_c": float(np.sum(-alpha[c]) ** 2 / np.sum(alpha[c] ** 2)),
        "wrong_sign_share": float(np.mean(-alpha[c] < 0)),
    }
    if basis_fitted is not None:
        F = basis_fitted.control_features(X)
        out["smd_dict_max"] = float(np.nanmax(np.abs(_smd(F[t], F[c], w))))
    else:
        out["smd_dict_max"] = float("nan")
    return out


def _grr_arm(arm, X, Y, D, Zs, folds, inner_seed, center_seed):
    gen_name, dictionary = arm.split("-")
    g = gr()
    basis = ATTArmBasis(dictionary, random_state=center_seed)
    kwargs = dict(
        X=X, Y=Y, basis=basis, generator=generator(gen_name),
        riesz_alpha_ref=alpha_ref,
        outcome_models="separate", outcome_basis=outcome_basis(), outcome_link="identity",
        outcome_penalty="l2", outcome_lam=0.0,
        fold_ids=folds, estimators=("arw", "tmle"), expose_alpha_values=True,
        random_state=inner_seed,
    )  # fmt: skip
    if dictionary == "D3":
        kwargs.update(
            riesz_penalty="l2", riesz_lam=LAMBDA_GRID[0], riesz_lam_grid=list(LAMBDA_GRID),
            riesz_cv_folds=INNER_FOLDS, riesz_selection_score="squared_loss_validation",
            riesz_admissibility_thresholds=admissibility_thresholds(),
        )  # fmt: skip
    else:
        kwargs.update(riesz_penalty=None, riesz_lam=0.0, riesz_tol=SAMPLE_TOL)
    res, warns, converged = baselines.run_recording_warnings(lambda: g.grr_att(**kwargs))
    common = {
        "fold_status": json.dumps([[str(v) for v in f] for f in res.fold_status]),
        "n_warnings": int(sum(warns.values())),
        "warnings": json.dumps(warns),
    }
    if not res.success or not converged:
        status = res.status if not res.success else "convergence_warning"
        out = {e: _failed(status, **common) for e in ESTIMATORS}
        out["TMLE"]["n_warnings"] = 0  # one grr_att call: warnings counted once, on ARW
        return out
    d = res.diagnostics
    a = np.asarray(d["alpha_values"], float)
    if not np.all(np.isfinite(a)):
        out = {e: _failed("nonfinite", **common) for e in ESTIMATORS}
        out["TMLE"]["n_warnings"] = 0
        return out
    fit = {
        **weight_summaries(D, Zs, X, a, basis if dictionary != "D3" else None),
        "max_abs_alpha": float(np.max(np.abs(a))),
        "lam": float(d["riesz_cv_lam_median"]) if dictionary == "D3" else 0.0,
    }
    out = {}
    for e in ESTIMATORS:
        r = res[e.lower()]
        est, se = float(r.estimate), float(r.se)
        if not (np.isfinite(est) and np.isfinite(se)):
            out[e] = _failed("nonfinite", **common)
        else:
            out[e] = {"estimate": est, "se": se, "status": "ok", **fit, **common}
    # one grr_att call fits the representers of both estimators: warnings counted once
    out["TMLE"] = {**out["TMLE"], "n_warnings": 0}
    return out


def _att_aipw(D, Y, e, mu0):
    """ATT AIPW with the estimated treated share and its influence-function SE."""
    pi = float(np.mean(D))
    r = D * (Y - mu0) - (1 - D) * e / (1 - e) * (Y - mu0)
    theta = float(np.sum(r) / np.sum(D))
    psi = (r - D * theta) / pi
    return theta, float(np.std(psi, ddof=0) / np.sqrt(len(Y)))


def _ols_mu0(F, D, Y, folds):
    mu0 = np.empty(len(Y))
    Fc = np.column_stack([np.ones(len(Y)), F])
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        rows = tr & (D == 0)
        coef, *_ = np.linalg.lstsq(Fc[rows], Y[rows], rcond=None)
        mu0[te] = Fc[te] @ coef
    return mu0


def logit_aipw(D, Y, F2, folds):
    from sklearn.linear_model import LogisticRegression

    def fit():
        e = np.empty(len(Y))
        for k in np.unique(folds):
            tr, te = folds != k, folds == k
            clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000, tol=1e-10)
            clf.fit(F2[tr], D[tr])
            e[te] = clf.predict_proba(F2[te])[:, 1]
        return e

    e, warns, converged = baselines.run_recording_warnings(fit)
    common = {"n_warnings": int(sum(warns.values())), "warnings": json.dumps(warns),
              "e_min": float(np.min(e))}  # fmt: skip
    if not converged:
        return _failed("convergence_warning", **common)
    if not np.all((e > 1e-12) & (e < 1 - 1e-12)):
        return _failed("propensity_range", **common)
    theta, se = _att_aipw(D, Y, e, _ols_mu0(F2, D, Y, folds))
    if not (np.isfinite(theta) and np.isfinite(se)):
        return _failed("nonfinite", **common)
    return {"estimate": theta, "se": se, "status": "ok", **common}


def _inner_ids(n, generator):
    return fold_ids(n, GBM_INNER_FOLDS, generator)


def dml_gbm(D, Y, Zs, folds, inner_gen, baseline_gen):
    """Cross-fitted AIPW for the ATT with gradient-boosted propensity (features Zs) and
    control outcome regression (features Zs, fitted on the controls), hyper-parameters
    chosen on each outer training fold by inner CV over the E-12 grid (log-loss, MSE)."""
    from sklearn.metrics import log_loss

    seed = _seed_int(baseline_gen)

    def fit():
        e, mu0, chosen = np.empty(len(Y)), np.empty(len(Y)), []
        for k in np.unique(folds):
            tr, te = np.flatnonzero(folds != k), folds == k
            inner = _inner_ids(len(tr), inner_gen)
            best_p, best_o = (np.inf, None), (np.inf, None)
            for params in baselines.GBM_GRID:
                ll, mse = 0.0, 0.0
                for j in range(GBM_INNER_FOLDS):
                    itr, ite = tr[inner != j], tr[inner == j]
                    p = baselines._gbm_classifier(params, seed).fit(Zs[itr], D[itr])
                    ll += log_loss(D[ite], p.predict_proba(Zs[ite])[:, 1], labels=[0, 1]) * len(ite)
                    ctr, cte = itr[D[itr] == 0], ite[D[ite] == 0]
                    reg = baselines._gbm_regressor(params, seed).fit(Zs[ctr], Y[ctr])
                    mse += float(np.sum((reg.predict(Zs[cte]) - Y[cte]) ** 2))
                if ll < best_p[0]:
                    best_p = (ll, params)
                if mse < best_o[0]:
                    best_o = (mse, params)
            chosen.append([list(best_p[1]), list(best_o[1])])
            ctr = tr[D[tr] == 0]
            e[te] = baselines._gbm_classifier(best_p[1], seed).fit(Zs[tr], D[tr]).predict_proba(
                Zs[te])[:, 1]  # fmt: skip
            mu0[te] = baselines._gbm_regressor(best_o[1], seed).fit(Zs[ctr], Y[ctr]).predict(Zs[te])
        return e, mu0, chosen

    (e, mu0, chosen), warns, converged = baselines.run_recording_warnings(fit)
    common = {"n_warnings": int(sum(warns.values())), "warnings": json.dumps(warns),
              "gbm_params": json.dumps(chosen), "e_min": float(np.min(e))}  # fmt: skip
    if not converged:
        return _failed("convergence_warning", **common)
    if not np.all((e > 0) & (e < 1)):
        return _failed("propensity_range", **common)
    theta, se = _att_aipw(D, Y, e, mu0)
    if not (np.isfinite(theta) and np.isfinite(se)):
        return _failed("nonfinite", **common)
    return {"estimate": theta, "se": se, "status": "ok", **common}


_DATA = None


def data():
    """The sample, loaded once per process."""
    global _DATA
    if _DATA is None:
        D, Y, Zs, raw = load()
        _DATA = {"D": D, "Y": Y, "Zs": Zs, "raw": raw, "X": design(D, Zs, raw)}
    return _DATA


def replicate(task):
    """All arms on one fold split ``(cell, split, entropy)``."""
    ci, split, entropy = task
    dd = data()
    D, Y, Zs, raw, X = dd["D"], dd["Y"], dd["Zs"], dd["raw"], dd["X"]
    s = Seeds(EXP, entropy=entropy)
    folds = fold_ids(len(Y), K, s.folds(ci, split))
    inner = s.inner_cv(ci, split)
    inner_seed, center_seed = _seed_int(inner), _seed_int(inner)
    out = {}
    for arm in GRR_ARMS:
        res = _grr_arm(arm, X, Y, D, Zs, folds, inner_seed, center_seed)
        for e in ESTIMATORS:
            out[f"{arm}|{e}"] = res[e]
    out["LogitAIPW"] = logit_aipw(D, Y, d2_features(Zs, raw), folds)
    out["DML-GBM"] = dml_gbm(D, Y, Zs, folds, inner, s.baseline(ci, split))
    return out


def tasks_for(entropy, reps, cells=None):
    cells = range(len(CELLS)) if cells is None else cells
    return [(ci, r, entropy) for ci in cells for r in range(reps)]


# ---------------------------------------------------------------- full sample


def full_sample(entropy):
    """In-sample fits of every GRR arm (weights for the SMD (a)), the EB check and the
    admissibility diagnostic for BKL (C = 0.05). D3 chooses lambda on the whole sample
    by the same inner CV (seed: ``inner_cv`` stream, cell 1, rep 0)."""
    g = gr()
    dd = data()
    D, Y, Zs, raw, X = dd["D"], dd["Y"], dd["Zs"], dd["raw"], dd["X"]
    n, pi1 = len(Y), float(np.mean(D))
    m = g.ATTFunctional(treatment_index=0, pi=pi1, pi_is_estimated=True)
    gen_seed = Seeds(EXP, entropy=entropy).inner_cv(1, 0)
    inner_seed, center_seed = _seed_int(gen_seed), _seed_int(gen_seed)
    fits, alphas, n_warn_total = {}, {}, 0
    for arm in GRR_ARMS:
        gen_name, dictionary = arm.split("-")
        gen = generator(gen_name)
        basis = ATTArmBasis(dictionary, random_state=center_seed).fit(X)
        lam, penalty = 0.0, None
        if dictionary == "D3":
            cfg = g.GRRCVConfig(lam_grid=list(LAMBDA_GRID), cv_folds=INNER_FOLDS,
                                selection_score="squared_loss_validation",
                                admissibility_thresholds=admissibility_thresholds(),
                                random_state=inner_seed)  # fmt: skip
            sel, warns_cv, conv_cv = baselines.run_recording_warnings(
                lambda: g.select_grr_hyperparams(
                    X_train=X, y_train=Y, m=m, basis=ATTArmBasis("D3", random_state=center_seed),
                    generator=gen, config=cfg, riesz_penalty="l2", riesz_lam=LAMBDA_GRID[0],
                    riesz_alpha_ref=alpha_ref, outcome_penalty="l2", outcome_lam=0.0,
                    raise_on_failure=False,
                )
            )  # fmt: skip
            n_warn_total += int(sum(warns_cv.values()))
            if sel.status != "ok" or not conv_cv:
                fits[arm] = {"status": str(sel.status) if sel.status != "ok" else "convergence_warning"}
                continue
            lam, penalty = float(sel.lam), "l2"
        model = g.GRRGLM(basis=basis, generator=gen, functional=m, penalty=penalty, lam=lam,
                         offset=g.offset_from_alpha(gen, alpha_ref))  # fmt: skip
        fr, warns, converged = baselines.run_recording_warnings(
            lambda: model.fit(X, tol=SAMPLE_TOL) if penalty is None else model.fit(X)
        )
        n_warn_total += int(sum(warns.values()))
        if fr.status != "ok" or not converged:
            fits[arm] = {"status": "convergence_warning" if fr.status == "ok" else str(fr.status),
                         "n_warnings": int(sum(warns.values()))}  # fmt: skip
            continue
        a = np.asarray(model.predict_alpha(X), float)
        alphas[arm] = a
        fits[arm] = {"status": "ok", "lam": lam, "max_abs_alpha": float(np.max(np.abs(a))),
                     "n_warnings": int(sum(warns.values())),
                     **weight_summaries(D, Zs, X, a, basis)}  # fmt: skip
    c, t = D == 0, D == 1
    unweighted = _smd(Zs[t], Zs[c], np.full(int(c.sum()), 1.0 / c.sum()))
    # EB check: UKL (C=0) x D1, lambda = 0, against an independent ebal-type dual Newton
    F = np.column_stack([np.ones(n), Zs])
    w_eb = baselines.eb_dual_newton(F[c], n * F[t].mean(axis=0), n)
    eb = {"converged": w_eb is not None}
    if w_eb is not None and EB_ARM in alphas:
        diff = float(np.max(np.abs(w_eb - (-alphas[EB_ARM][c]))) / max(1.0, float(np.max(w_eb))))
        eb.update(max_rel_diff=diff, agrees=bool(diff <= EB_TOL))
    else:
        eb.update(max_rel_diff=None, agrees=False)
    # admissibility diagnostic (PR4a 2(d)): not a test of the population condition
    threshold = C_VALUE["BKL"] * pi1 / (1 + C_VALUE["BKL"] * pi1)
    e_logit, warns_l, conv_l = baselines.run_recording_warnings(
        lambda: _logistic_full(d2_features(Zs, raw), D)
    )
    n_warn_total += int(sum(warns_l.values()))
    diag = {"bkl_threshold": threshold, "logit_converged": bool(conv_l),
            # a nonconverged logistic fit gives no diagnostic
            "logit_e_min_controls": float(np.min(e_logit[c])) if conv_l else None,
            "logit_e_min_all": float(np.min(e_logit)) if conv_l else None}  # fmt: skip
    if EB_ARM in alphas:
        w = -alphas[EB_ARM][c]  # control weight e / (pi1 (1 - e))
        e_ukl = pi1 * w / (1 + pi1 * w)
        diag["ukl_d1_e_min"] = float(np.min(e_ukl))
    return {"fits": fits, "unweighted_smd": [float(v) for v in unweighted], "eb_check": eb,
            "admissibility": diag, "pi1": pi1, "n_warnings": n_warn_total}  # fmt: skip


def _logistic_full(F2, D):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000, tol=1e-10).fit(F2, D)
    return clf.predict_proba(F2)[:, 1]


# ---------------------------------------------------------------- aggregation

NUMERIC_FIELDS = (
    "estimate", "se", "max_abs_alpha", "smd_raw_max", "smd_dict_max", "ess_c",
    "wrong_sign_share", "lam", "e_min", "n_warnings",
)  # fmt: skip
TEXT_FIELDS = ("status", "warnings", "fold_status", "smd_raw", "gbm_params")


def record_label(label):
    """The run recorder's form of an arm label (letters, digits, ``_`` and ``-`` only)."""
    return label.replace("|", "_")


def raw_frame(tasks, results):
    """One row per split x arm (failures are rows too, §1.7)."""
    import pandas as pd

    rows = []
    for (ci, split, _), res in zip(tasks, results, strict=True):
        for arm in ARM_LABELS:
            r = res[arm]
            row = {"cell": ci, "split": split, "arm": arm}
            row.update({k: float(r.get(k, np.nan)) for k in NUMERIC_FIELDS})
            row.update({k: str(r.get(k, "")) for k in TEXT_FIELDS})
            rows.append(row)
    return pd.DataFrame(rows)


def total_warnings(raw):
    return int(raw["n_warnings"].sum())


def median_aggregate(theta, sigma2, n):
    """``theta_med`` and ``sigma2_med = median{sigma2_s + n (theta_s - theta_med)^2}``."""
    theta, sigma2 = np.asarray(theta, float), np.asarray(sigma2, float)
    t = float(np.median(theta))
    return t, float(np.median(sigma2 + n * (theta - t) ** 2))


def summarise(raw, full):
    """One row per arm: the split-median estimate and SE (successful splits only), the
    number of successful splits, the failures by status, the median cross-fit weight
    summaries, and the full-sample weight summaries."""
    import pandas as pd

    rows = []
    for arm in ARM_LABELS:
        g = raw[raw["arm"] == arm]
        ok = g[g["status"] == "ok"]
        k = len(ok)
        row = {"arm": arm, "R": len(g), "R_s": k,
               "status_counts": json.dumps(dict(sorted(collections.Counter(g["status"]).items())))}
        if k > 0 and k >= len(g) / 2:  # reported unless more than half of the splits fail
            t, s2 = median_aggregate(ok["estimate"], N * ok["se"] ** 2, N)
            row.update(estimate=t, se=float(np.sqrt(s2 / N)), reported=True)
        else:
            row.update(estimate=np.nan, se=np.nan, reported=False)
        for f in ("smd_raw_max", "smd_dict_max", "ess_c", "wrong_sign_share", "max_abs_alpha", "lam"):
            v = ok[f].dropna()
            row[f"cf_{f}"] = float(np.median(v)) if len(v) else np.nan
        vecs = [json.loads(v) for v in ok["smd_raw"] if isinstance(v, str) and v.startswith("[")]
        row["cf_smd_raw"] = (json.dumps([float(x) for x in np.median(np.array(vecs), axis=0)])
                             if vecs else "")  # fmt: skip
        base = arm.split("|")[0]
        fs = full["fits"].get(base, {})
        row["full_status"] = fs.get("status", "")
        for f in ("smd_raw_max", "smd_dict_max", "ess_c", "wrong_sign_share", "max_abs_alpha", "lam"):
            row[f"full_{f}"] = fs.get(f, np.nan)
        row["full_smd_raw"] = fs.get("smd_raw", "")
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ["unweighted_smd"] = json.dumps(full["unweighted_smd"])
    summ["eb_check"] = json.dumps(full["eb_check"])
    summ["admissibility"] = json.dumps(full["admissibility"])
    summ["pi1"] = full["pi1"]
    summ["full_n_warnings"] = full["n_warnings"]
    return summ


# ---------------------------------------------------------------- tables and figure

END = " \\\\"
GEN_LABEL = {"SQ": "SQ ($C=0$)", "UKL": "UKL ($C=0$)", "BP": "BP ($\\omega=0.5$, $C=0$)",
             "BKL": "BKL ($C=0.05$)"}  # fmt: skip
DICT_LABEL = {"D1": "D1", "D2": "D2", "D3": "D3"}


def _fmt(x, d=3):
    if x is None or x != x:
        return "--"
    s = f"{float(x):.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s  # no "-0.000"


def _label(arm):
    if arm in BASELINE_ARMS:
        return arm
    base, est = arm.split("|")
    gen, dic = base.split("-")
    name = f"EB (= UKL, $C=0$, D1)" if base == EB_ARM else f"{GEN_LABEL[gen]}, {DICT_LABEL[dic]}"
    return name if est == "ARW" else f"{name}, TMLE"


def _estimate_cells(r):
    if not bool(r["reported"]):
        return [f"no exact fit ({int(r['R_s'])}/{int(r['R'])})", "--"]
    return [_fmt(r["estimate"], 1), _fmt(r["se"], 1)]


def tables(S):
    """``tab_E22_main`` (Main), ``tab_E22_full`` and ``tab_E22_smd`` (Supp; longtable
    bodies with repeated heads), ``tab_E22_diag`` (EB check and admissibility) and
    ``tab_E22_status`` (failures by status)."""
    out = {}
    by = {r["arm"]: r for _, r in S.iterrows()}
    head = ["Method", "Estimate", "SE", "Splits", "Max SMD (8)", "ESS C", "Neg.\\ share"]
    lines = ["\\begin{tabular}{lrrrrrr}", "\\hline", " & ".join(head) + END, "\\hline"]
    for arm in [f"{a}|ARW" for a in GRR_ARMS] + list(BASELINE_ARMS):
        r = by[arm]
        cells = [_label(arm), *_estimate_cells(r), f"{int(r['R_s'])}/{int(r['R'])}"]
        if arm in BASELINE_ARMS:
            cells += ["--", "--", "--"]
        else:
            cells += [_fmt(r["full_smd_raw_max"]), _fmt(r["full_ess_c"], 1),
                      _fmt(r["full_wrong_sign_share"])]  # fmt: skip
        lines.append(" & ".join(cells) + END)
    lines += [
        "\\hline",
        "\\multicolumn{7}{l}{Reference: the NSW experimental estimate is \\$1,794 (Dehejia and Wahba, 1999);}" + END,
        "\\multicolumn{7}{l}{it is not the true effect for this observational sample.}" + END,
        "\\hline", "\\end{tabular}",
    ]  # fmt: skip
    out["tab_E22_main"] = "\n".join(lines) + "\n"

    head = ["Method", "Estimate", "SE", "Splits", "$\\lambda$ (med.)", "Max SMD dict.\\ (full)",
            "Max SMD (8, full)", "Max SMD dict.\\ (cf)", "Max SMD (8, cf)", "ESS C (cf)",
            "Neg.\\ share (cf)", "Max $|\\widehat\\alpha|$ (cf)"]  # fmt: skip
    h = ["\\hline", " & ".join(head) + END, "\\hline"]
    lines = ["\\begin{tabular}{lrrrrrrrrrrr}", *h, "\\endfirsthead", *h, "\\endhead"]
    for arm in ARM_LABELS:
        r = by[arm]
        cells = [_label(arm), *_estimate_cells(r), f"{int(r['R_s'])}/{int(r['R'])}"]
        if arm in BASELINE_ARMS:
            cells += ["--"] * 8
        else:
            cells += [_fmt(r["cf_lam"], 4), _fmt(r["full_smd_dict_max"]), _fmt(r["full_smd_raw_max"]),
                      _fmt(r["cf_smd_dict_max"]), _fmt(r["cf_smd_raw_max"]), _fmt(r["cf_ess_c"], 1),
                      _fmt(r["cf_wrong_sign_share"]), _fmt(r["cf_max_abs_alpha"], 2)]  # fmt: skip
        lines.append(" & ".join(cells) + END)
    lines += ["\\hline",
              "\\multicolumn{12}{l}{--: not applicable, not reported, or not defined (the cross-fit "
              "dictionary SMD of D3, whose dictionaries are fold specific).}" + END,
              "\\hline", "\\end{tabular}"]  # fmt: skip
    out["tab_E22_full"] = "\n".join(lines) + "\n"

    head = ["Weights", *[c.replace("_", "\\_") for c in RAW]]
    h = ["\\hline", " & ".join(head) + END, "\\hline"]
    lines = ["\\begin{tabular}{l" + "r" * len(RAW) + "}", *h, "\\endfirsthead", *h, "\\endhead"]
    unw = json.loads(S["unweighted_smd"].iloc[0])
    lines.append(" & ".join(["Unweighted", *[_fmt(v) for v in unw]]) + END)
    for arm in GRR_ARMS:
        r = by[f"{arm}|ARW"]
        for col, tag in (("full_smd_raw", "full"), ("cf_smd_raw", "cf")):
            v = r[col]
            vals = json.loads(v) if isinstance(v, str) and v.startswith("[") else None
            cells = [_fmt(x) for x in vals] if vals else ["--"] * len(RAW)
            lines.append(" & ".join([f"{_label(f'{arm}|ARW')} ({tag})", *cells]) + END)
    lines += ["\\hline", "\\end{tabular}"]
    out["tab_E22_smd"] = "\n".join(lines) + "\n"

    eb = json.loads(S["eb_check"].iloc[0])
    ad = json.loads(S["admissibility"].iloc[0])
    lines = ["\\begin{tabular}{lr}", "\\hline", "Diagnostic & Value" + END, "\\hline",
             f"EB vs.\\ independent dual Newton (max.\\ relative difference) & {_sci(eb.get('max_rel_diff'))}" + END,
             f"BKL ($C=0.05$) threshold for $\\widehat e$ & {_fmt(ad['bkl_threshold'], 5)}" + END,
             f"Min.\\ logistic $\\widehat e$ (all units) & {_fmt(ad.get('logit_e_min_all'), 5)}" + END,
             f"Min.\\ logistic $\\widehat e$ among controls & {_fmt(ad.get('logit_e_min_controls'), 5)}" + END,
             f"Min.\\ $\\widehat e$ implied by UKL, D1 & {_fmt(ad.get('ukl_d1_e_min'), 5)}" + END,
             "\\hline", "\\end{tabular}"]  # fmt: skip
    out["tab_E22_diag"] = "\n".join(lines) + "\n"

    lines = ["\\begin{tabular}{lrl}", "\\hline", "Method & Successful splits & Failures by status" + END,
             "\\hline"]  # fmt: skip
    any_fail = False
    for arm in ARM_LABELS:
        r = by[arm]
        counts = {k: v for k, v in json.loads(r["status_counts"]).items() if k != "ok"}
        if counts:
            any_fail = True
            txt = ", ".join(f"{k.replace('_', chr(92) + '_')}: {v}" for k, v in counts.items())
            lines.append(f"{_label(arm)} & {int(r['R_s'])}/{int(r['R'])} & {txt}" + END)
    if not any_fail:
        lines.append("\\multicolumn{3}{l}{No split failed for any method.}" + END)
    full_fail = [(a, by[f"{a}|ARW"]["full_status"]) for a in GRR_ARMS
                 if by[f"{a}|ARW"]["full_status"] != "ok"]  # fmt: skip
    lines += ["\\hline", "\\multicolumn{3}{l}{Full-sample fits (weights for the SMD and ESS):}" + END]
    if full_fail:
        for a, st in full_fail:
            lines.append(f"{_label(f'{a}|ARW')} & -- & {str(st).replace('_', chr(92) + '_')}" + END)
    else:
        lines.append("\\multicolumn{3}{l}{Every full-sample fit succeeded.}" + END)
    lines += ["\\hline", "\\end{tabular}"]
    out["tab_E22_status"] = "\n".join(lines) + "\n"
    return out


def _sci(x):
    return "--" if x is None or x != x else f"{float(x):.1e}"


def figure(S):
    """``fig_E22_love``: |SMD| of the eight raw covariates, unweighted and under the
    full-sample weights of the D1 and D2 fits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {r["arm"]: r for _, r in S.iterrows()}
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ys = np.arange(len(RAW))
    ax.scatter(np.abs(json.loads(S["unweighted_smd"].iloc[0])), ys, marker="x", color="k",
               label="unweighted")  # fmt: skip
    markers = {"D1": "o", "D2": "s"}
    colors = {"SQ": "#4C72B0", "UKL": "#DD8452", "BP": "#55A868", "BKL": "#C44E52"}
    for arm in GRR_ARMS:
        gen, dic = arm.split("-")
        if dic not in markers:
            continue
        r = by[f"{arm}|ARW"]
        if not (isinstance(r["full_smd_raw"], str) and r["full_smd_raw"]):
            continue
        ax.scatter(np.abs(json.loads(r["full_smd_raw"])), ys, marker=markers[dic],
                   facecolors="none", edgecolors=colors[gen], label=f"{gen}, {dic}")  # fmt: skip
    ax.set_yticks(ys, list(RAW))
    ax.set_xlabel("|SMD|")
    ax.axvline(0.1, color="0.6", lw=0.8, ls="--")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    fig.tight_layout()
    return fig
