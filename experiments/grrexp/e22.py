"""E-22: LaLonde ATT with a primary partition and partition sensitivity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import env, study
from .seeds import CONFIRMATORY_ENTROPY, Seeds, fold_ids

EXP = 22
DATA_FILE = "lalonde/lalonde.csv"
RAW = ("age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75")
N, N_TREATED = 614, 185
N_VALUES = (N,)
CELLS = [[N]]
R, K, PRIMARY_PARTITION = 20, 5, 0
DICTIONARIES = ("D1", "D2")
GRR_ARMS = tuple(f"{g}-{d}" for d in DICTIONARIES for g in ("SQ", "UKL0"))
ARM_LABELS = tuple(f"{g}-{d}" for d in DICTIONARIES for g in ("SQ", "UKL0", "LogitAIPW"))
_DATA = None


def gr():
    import genriesz

    return genriesz


def sign(X):
    """Treated rows take the positive branch, control rows the negative one."""
    return np.where(np.atleast_2d(X)[:, 0] == 1, 1, -1)


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


def data():
    """The sample, loaded once per process."""
    global _DATA
    if _DATA is None:
        D, Y, Zs, raw = load()
        _DATA = {"D": D, "Y": Y, "Zs": Zs, "raw": raw, "X": design(D, Zs, raw)}
    return _DATA


sign.vectorized = True


class ATTArmBasis:
    """Treated intercept and control-arm D1 or D2 dictionary."""

    def __init__(self, dictionary):
        if dictionary not in DICTIONARIES:
            raise ValueError(dictionary)
        self.dictionary = dictionary

    def copy(self):
        return ATTArmBasis(self.dictionary)

    def fit(self, X, y=None):
        return self

    def _psi(self, X):
        X = np.atleast_2d(X)
        if self.dictionary == "D1":
            return np.column_stack([np.ones(len(X)), X[:, 1:9]])
        raw = np.column_stack([np.zeros((len(X), 6)), 1 - X[:, 9:11]])
        return np.column_stack([np.ones(len(X)), d2_features(X[:, 1:9], raw)])

    @property
    def n_features(self):
        return 10 if self.dictionary == "D1" else 16

    def __call__(self, X):
        X = np.atleast_2d(X)
        D = X[:, :1]
        return np.column_stack([D, (1 - D) * self._psi(X)])

    def control_features(self, X):
        return self._psi(X)[:, 1:]


def replicate(task):
    cell, rep, entropy = task
    sample = data()
    D, Y, Zs, raw, X = (sample[k] for k in ("D", "Y", "Zs", "raw", "X"))
    seeds = Seeds(EXP, entropy)
    folds = fold_ids(len(Y), K, seeds.folds(cell, rep))
    arms = {}
    g = gr()
    # D2 outcome regression is shared by every specification.
    mu0 = _ols_mu0(d2_features(Zs, raw), D, Y, folds)
    for dictionary in DICTIONARIES:
        basis = ATTArmBasis(dictionary)
        for name in ("SQ", "UKL0"):
            gen = (
                g.SquaredGenerator(C=0.0) if name == "SQ" else g.UKLGenerator(C=0.0, branch_fn=sign)
            )
            log = study.FitLog()
            res = g.grr_att(
                X=X,
                Y=Y,
                basis=basis,
                generator=gen,
                riesz_alpha_ref=lambda rows: np.where(rows[:, 0] == 1, 2.0, -2.0),
                riesz_penalty=None,
                riesz_lam=0.0,
                outcome_models="separate",
                outcome_basis=outcome_basis(),
                outcome_link="identity",
                outcome_penalty=None,
                outcome_lam=0.0,
                fold_ids=folds,
                estimators=("arw",),
                expose_alpha_values=True,
                evaluate_m_alpha=False,
                fit_hook=log.hook,
            )
            record = study.functional_record(res, log)
            if record["status"] == "ok":
                alpha = np.asarray(record["alpha"])
                record.update(weight_summaries(D, Zs, X, alpha, basis))
            arms[f"{name}-{dictionary}"] = record

        def logit(basis=basis):
            from sklearn.linear_model import LogisticRegression

            e = np.empty(len(Y))
            F = basis.control_features(X)
            for k in range(K):
                tr, te = folds != k, folds == k
                model = LogisticRegression(penalty=None)
                model.fit(F[tr], D[tr])
                e[te] = model.predict_proba(F[te])[:, 1]
            if not np.all((e > 0) & (e < 1)):
                return study.failed("propensity_range")
            theta, se = _att_aipw(D, Y, e, mu0)
            influence = ((D - (1 - D) * e / (1 - e)) * (Y - mu0) - D * theta) / D.mean()
            top = float(np.max(np.abs(influence)))
            scaled = influence / top if top > 0 else np.zeros_like(influence)
            mass = float(scaled @ scaled)
            alpha = (D - (1 - D) * e / (1 - e)) / D.mean()
            return {
                "estimate": theta,
                "se": se,
                "status": "ok",
                "alpha": alpha.tolist(),
                "score_concentration": float(np.max(scaled**2) / mass) if mass > 0 else np.nan,
                "max_abs_alpha": float(np.max(np.abs(alpha))),
                **weight_summaries(D, Zs, X, alpha, basis),
            }

        arms[f"LogitAIPW-{dictionary}"] = study.recorded(logit)
    return {"arms": arms, "folds": folds.tolist()}


def tasks_for(entropy=CONFIRMATORY_ENTROPY):
    import sys

    return study.tasks_for(sys.modules[__name__], entropy)


def summarise(raw):
    rows = []
    for arm, frame in raw.groupby("arm", sort=False):
        primary = frame[frame.rep == PRIMARY_PARTITION].iloc[0]
        sensitivity = frame[frame.rep != PRIMARY_PARTITION]
        ok = sensitivity.status.eq("ok")
        row = {
            "arm": arm,
            "n": N,
            "primary_partition": PRIMARY_PARTITION,
            "primary_estimate": primary.estimate,
            "primary_se": primary.se,
            "primary_status": primary.status,
            "sensitivity_partitions": len(sensitivity),
            "sensitivity_failures": int((~ok).sum()),
            "failure_rate": float(frame.status.ne("ok").mean()),
        }
        for field in (
            "estimate",
            "se",
            "ess_c",
            "max_abs_alpha",
            "smd_raw_max",
            "smd_dict_max",
            "wrong_sign_share",
            "score_concentration",
        ):
            if field in frame:
                row["primary_" + field] = primary[field]
                values = sensitivity.loc[ok, field]
                for q in (0.05, 0.5, 0.95):
                    row[f"sensitivity_{field}_q{int(q * 100):02}"] = values.quantile(q)
        for field in (
            "ess_c", "max_abs_alpha", "smd_raw_max", "smd_dict_max", "wrong_sign_share",
            "score_concentration",
        ):
            row.setdefault("primary_" + field, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def tables(summary):
    cols = [
        "arm",
        "primary_estimate",
        "primary_se",
        "primary_status",
        "sensitivity_failures",
        "primary_ess_c",
        "primary_max_abs_alpha",
        "primary_smd_raw_max",
        "primary_score_concentration",
    ]
    concentration = [
        "arm",
        "primary_score_concentration",
        *[f"sensitivity_score_concentration_q{q:02}" for q in (5, 50, 95)],
    ]
    return {
        "tab_E22": summary[cols].to_latex(index=False, escape=True, float_format="%.3f"),
        "tab_E22_score_concentration": summary.reindex(columns=concentration).to_latex(
            index=False, escape=True, float_format="%.4f"
        ),
    }


def figure(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(summary))
    ax.errorbar(
        summary.primary_estimate,
        positions,
        xerr=1.96 * summary.primary_se,
        fmt="o",
        label="Primary estimate and 95% interval",
    )
    ax.hlines(
        positions,
        summary.sensitivity_estimate_q05,
        summary.sensitivity_estimate_q95,
        color="gray",
        label="Sensitivity estimate: central 90%",
    )
    ax.set_yticks(positions, summary.arm)
    ax.set(xlabel="ATT", title="LaLonde: primary partition and partition sensitivity")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
