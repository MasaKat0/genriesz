"""E-21 support code: the IHDP replications, the functional maps, the EB check, the
AutoDML-lasso fit with zero dictionary columns, the inner cross-validation, the
records, the summaries, S1 and the tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import e21, env, outputs  # noqa: E402
from grrexp.seeds import PILOT_ENTROPY, Seeds, fold_ids  # noqa: E402

HAVE_IHDP = all((env.DATA_DIR / name).is_file() for name in env.REGISTERED_DATA[21])
needs_data = pytest.mark.skipif(not HAVE_IHDP, reason="IHDP snapshot not present (not tracked)")


def _toy(n=400, seed=0, p=3):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p))
    e = 1.0 / (1.0 + np.exp(-(0.5 * Z[:, 0] - 0.3 * Z[:, 1])))
    D = (rng.uniform(size=n) < e).astype(float)
    Y = D * (1.0 + Z[:, 0]) + Z[:, 1] + rng.normal(size=n)
    return np.column_stack([D, Z]), Y


# ---------------------------------------------------------------- data


@needs_data
def test_snapshot_hashes_match_the_registration() -> None:
    assert set(env.verify_data(env.REGISTERED_DATA[21])) == set(env.REGISTERED_DATA[21])


@needs_data
def test_replications_have_747_rows_att_four_and_standardized_continuous_columns() -> None:
    for rep in (0, 37, 99):
        D, Z, Y, ate, att = e21.replication(rep)
        assert Z.shape == (e21.N, e21.P_COV) and D.shape == Y.shape == (e21.N,)
        assert set(np.unique(D)) == {0.0, 1.0} and int(D.sum()) == 139
        assert abs(att - 4.0) <= 1e-9
        assert np.isfinite(ate)
        cont = Z[:, : e21.N_CONT]
        assert np.allclose(cont.mean(axis=0), 0.0, atol=1e-12)
        assert np.allclose(cont.std(axis=0), 1.0, atol=1e-12)
    # the 18th covariate (index 17): one treated unit in its 0 stratum
    D, Z, *_ = e21.replication(0)
    assert int(((Z[:, 17] == 0) & (D == 1)).sum()) == 1
    assert int(((Z[:, 17] == 0) & (D == 0)).sum()) == 26


# ---------------------------------------------------------------- functional maps


def test_m_of_and_centering_for_ate_and_att() -> None:
    X, _ = _toy(50)
    D = X[:, 0]

    def f(rows):
        return 2.0 * rows[:, 0] + rows[:, 1]

    assert np.allclose(e21.m_of("ATE", X, f, 0.4), 2.0)
    assert np.allclose(e21.m_of("ATT", X, f, 0.4), D * 2.0 / 0.4)
    assert e21.center("ATE", 1.5, D, 0.4) == 1.5
    assert np.allclose(e21.center("ATT", 1.5, D, 0.4), 1.5 * D / 0.4)


def test_score_record_att_uses_the_estimated_pi_correction() -> None:
    rng = np.random.default_rng(1)
    D = (rng.uniform(size=200) < 0.3).astype(float)
    parts = rng.normal(size=200) + D
    a = np.where(D == 1, 1 / 0.3, -0.5)
    rec = e21._score_record("ATT", parts, a, D, 0.3, {})
    theta = parts.mean()
    psi = parts - theta * D / 0.3
    assert rec["estimate"] == pytest.approx(theta)
    assert rec["se"] == pytest.approx(np.std(psi) / np.sqrt(200))
    bad = e21._score_record("ATE", np.array([1.0, np.inf]), np.ones(2), np.ones(2), 0.5, {})
    assert bad["status"] == "nonfinite"


# ---------------------------------------------------------------- EB


@pytest.mark.parametrize("estimand", ["ATE", "ATT"])
def test_ukl_c0_lambda0_equals_entropy_balancing(estimand) -> None:
    X, _ = _toy(600, seed=3)
    pi_hat = float(X[:, 0].mean())
    warn = e21._Warn()
    st, mdl = e21._eb_fit(estimand, X, pi_hat, warn, "t")
    assert st == "ok"
    diff = e21.eb_check(estimand, mdl, X, pi_hat)
    assert diff is not None and diff <= e21.EB_TOL


# ---------------------------------------------------------------- AutoDML-lasso


def test_autodml_zero_column_with_large_moment_is_degenerate() -> None:
    X, _ = _toy(300, seed=4)
    X[:, 1] = (X[:, 1] > 0).astype(float)
    X[X[:, 0] == 1, 1] = 0.0  # D x_1 vanishes; its ATE moment mean(x_1) ~ 0.5 is large
    st, rho, info = e21.autodml_fit("ATE", X, 0.5)
    assert st == "degenerate_functional" and rho is None
    assert info["zero_columns"]


def test_autodml_zero_column_with_small_moment_gets_coefficient_zero() -> None:
    X, _ = _toy(300, seed=5)
    X[:, 1] = 0.0
    X[0, 1] = 1e-3  # x_1 is almost zero everywhere; D x_1 vanishes on this sample
    if X[0, 0] == 1:
        X[0, 0] = 0.0
    st, rho, info = e21.autodml_fit("ATE", X, 0.5)
    assert st == "ok"
    p = len(rho)
    zero = info["zero_columns"]
    assert zero and np.all(rho[zero] == 0.0)
    assert p == 2 + 2 * 3


def test_autodml_att_moments() -> None:
    X, _ = _toy(20, seed=6)
    B, Mb = e21.autodml_dictionary("ATT", X, 0.25)
    D = X[:, 0]
    assert np.allclose(Mb[:, 1], D / 0.25)
    assert np.allclose(Mb[:, 0], 0.0)
    assert np.allclose(B[:, 1], D)


# ---------------------------------------------------------------- inner CV and outcome


def test_select_and_fit_picks_a_grid_value_and_the_criterion_is_consistent() -> None:
    X, _ = _toy(300, seed=7)
    pi_hat = float(X[:, 0].mean())
    inner = fold_ids(len(X), e21.INNER, np.random.default_rng(0))
    warn = e21._Warn()
    st, mdl, lam, _ = e21.select_and_fit("ATE", "lin", "SQ", X, inner, pi_hat, 0, warn, "t")
    assert st == "ok" and lam in e21.LAM_GRID
    status, pred = e21._predict_checked(mdl, "ATE", X)
    assert status == "ok"
    a = pred(X)
    crit = e21.riesz_criterion("ATE", X, pred, pi_hat)
    assert crit == pytest.approx(np.mean(a**2) - 2 * np.mean(e21.m_of("ATE", X, pred, pi_hat)))


def test_outcome_fit_chooses_from_the_grid_and_predicts() -> None:
    X, Y = _toy(300, seed=8)
    inner = fold_ids(len(X), e21.INNER, np.random.default_rng(1))
    pred, lam = e21.outcome_fit(X, Y, inner, 123)
    assert lam in e21.LAM_GRID
    assert np.all(np.isfinite(pred(X))) and pred(X).shape == (len(X),)


# ---------------------------------------------------------------- labels, summaries, tables


def test_arm_labels_are_unique_and_recorder_stems() -> None:
    labels = e21.ARM_LABELS
    assert len(labels) == len(set(labels)) == 58
    assert all(outputs.STEM.match(e21.record_label(a)) for a in labels)
    assert len({e21.record_label(a) for a in labels}) == len(labels)


def _fake_raw(reps=6):
    rows = []
    rng = np.random.default_rng(9)
    for rep in range(reps):
        for arm in e21.ARM_LABELS:
            est = arm.split("|")[0]
            theta0 = 4.0
            failed = arm.endswith("lin-BKL1|ARW") and rep == 0
            rows.append({"cell": 0, "rep": rep, "arm": arm, "theta0": theta0,
                         "estimate": np.nan if failed else theta0 + rng.normal(scale=0.1),
                         "se": np.nan if (failed or arm.endswith("|RA")) else 0.1,
                         "max_abs_alpha": 5.0, "ess": 100.0, "lam": 0.01, "eb_diff": np.nan,
                         "epsilon": np.nan, "n_warnings": 0.0,
                         "status": "cv_failed" if failed else "ok", "warnings": "",
                         "fold_status": "", "adml_folds": "", "gbm_params": "",
                         "_est": est})  # fmt: skip
    return pd.DataFrame(rows).drop(columns="_est")


def test_summaries_s1_and_tables() -> None:
    raw = _fake_raw()
    summ = e21.summarise(raw)
    assert len(summ) == len(e21.ARM_LABELS)
    row = summ[summ["arm"] == "ATE|lin-BKL1|ARW"].iloc[0]
    assert row["R_s"] == 5 and json.loads(row["status_counts"]) == {"cv_failed": 1, "ok": 5}
    assert summ.loc[summ["arm"] == "ATE|RA", "coverage"].isna().all()
    s1a = e21.s1_intervals(raw, Seeds(21, entropy=PILOT_ENTROPY))
    s1b = e21.s1_intervals(raw, Seeds(21, entropy=PILOT_ENTROPY))
    assert s1a.keys() == s1b.keys() and len(s1a) == 16
    for k in s1a:
        assert np.allclose(s1a[k], s1b[k], equal_nan=True)
    buf = summ.to_csv(index=False)
    S = pd.read_csv(__import__("io").StringIO(buf))
    tabs, macros, statements = e21.tables(S, s1a)
    assert set(tabs) == {"tab_E21_ate", "tab_E21_att", "tab_E21_status"}
    for name in ("tab_E21_ate", "tab_E21_att"):
        assert "\\endfirsthead" in tabs[name] and "\\endhead" in tabs[name]
        assert "-0.000" not in tabs[name]
    assert "cv\\_failed: 1" in tabs["tab_E21_status"]
    assert macros["IHDPreps"] == "6"
    assert set(statements) == {"S1", "S2"}


def test_fmt_has_no_negative_zero() -> None:
    assert e21._fmt(-0.0001) == "0.000"
    assert e21._fmt(-0.0006) == "-0.001"
    assert e21._fmt(float("nan")) == "--"


# ---------------------------------------------------------------- review fixes (round 1)


def test_att_m_of_ignores_control_counterfactuals() -> None:
    X = np.array([[0.0, 1.0], [1.0, 2.0]])

    def f(rows):  # undefined at the control unit's counterfactual rows
        return np.where(rows[:, 1] == 1.0, np.nan, 4.0 * rows[:, 0])

    assert np.array_equal(e21.m_of("ATT", X, f, 0.5), np.array([0.0, 8.0]))


def test_any_warning_invalidates_a_cv_candidate(monkeypatch) -> None:
    X, _ = _toy(200, seed=10)
    inner = fold_ids(len(X), e21.INNER, np.random.default_rng(2))
    real = e21.baselines.run_recording_warnings

    def noisy(fn):
        result, counts, converged = real(fn)
        return result, {"RuntimeWarning: injected": 1, **counts}, converged

    monkeypatch.setattr(e21.baselines, "run_recording_warnings", noisy)
    warn = e21._Warn()
    st, mdl, lam, detail = e21.select_and_fit("ATE", "lin", "SQ", X, inner, 0.5, 0, warn, "t")
    assert st == "cv_failed" and mdl is None
    assert {d[2] for d in detail} == {"warning"}


def test_records_are_finite_or_nonfinite_status() -> None:
    D = np.array([0.0, 1.0])
    big = e21._score_record("ATE", np.array([1e200, -1e200]), np.array([1e200, -1e200]), D, 0.5, {})
    assert big["status"] == "ok" and np.isfinite(big["se"]) and np.isfinite(big["ess"])
    over = e21._score_record("ATE", np.array([1e308, 1e308]), np.ones(2), D, 0.5, {})
    assert over["status"] == "nonfinite"


def test_all_failed_arm_summary_has_the_schema() -> None:
    raw = _fake_raw(4)
    raw["status"] = "cv_failed"
    raw["estimate"] = np.nan
    raw["se"] = np.nan
    summ = e21.summarise(raw)
    arw = summ[summ["arm"] == "ATE|lin-SQ|ARW"].iloc[0]
    assert arw["coverage"] == 0.0 and np.isnan(arw["coverage_conditional"])
    assert np.isnan(summ.loc[summ["arm"] == "ATE|RA", "coverage"]).all()
    assert list(summ.columns) == ["arm", *e21.SUMMARY_COLUMNS, "s2"]


def test_shared_fit_warnings_are_counted_once(monkeypatch) -> None:
    X, Y = _toy(300, seed=11)
    D = X[:, 0]
    pi_hat = float(D.mean())
    folds = fold_ids(len(X), e21.K, np.random.default_rng(3))
    inner = {
        k: fold_ids(int((folds != k).sum()), e21.INNER, np.random.default_rng(k))
        for k in range(e21.K)
    }

    def fake_select(estimand, dictionary, gen_name, X_fit, inner_k, pi, seed, warn, tag):
        warn.fits.append([tag, {"RuntimeWarning: injected": 1}])
        mdl = e21.model(estimand, "lin", "SQ", 0.01, pi, 0)
        mdl.fit(X_fit)
        return "ok", mdl, 0.01, []

    monkeypatch.setattr(e21, "select_and_fit", fake_select)
    gamma = {k: (lambda rows: np.zeros(len(rows))) for k in range(e21.K)}
    out = e21._grr_arms("ATE", X, Y, D, folds, inner, gamma, pi_hat, 0, True)
    base = "ATE|lin-SQ"
    assert out[f"{base}|ARW"]["n_warnings"] == e21.K
    assert out[f"{base}|RW"]["n_warnings"] == 0 and out[f"{base}|TMLE"]["n_warnings"] == 0
    assert json.loads(out[f"{base}|RW"]["warnings"]) == {"shared_with": "ARW"}


# ---------------------------------------------------------------- review fixes (round 2)


def test_tmle_epsilon_is_scale_stable(monkeypatch) -> None:
    X, Y = _toy(200, seed=12)
    D = X[:, 0]
    folds = fold_ids(len(X), e21.K, np.random.default_rng(4))
    inner = {k: None for k in range(e21.K)}

    class Huge:
        functional = e21.functional("ATE", 0.5)
        basis = None

    def fake_select(estimand, dictionary, gen_name, X_fit, inner_k, pi, seed, warn, tag):
        return "ok", Huge(), 0.01, []

    def fake_checked(mdl, estimand, X_eval):
        return "ok", (lambda rows: np.where(rows[:, 0] == 1, 1e160, -0.5e160))

    monkeypatch.setattr(e21, "select_and_fit", fake_select)
    monkeypatch.setattr(e21, "_predict_checked", fake_checked)
    gamma = {k: (lambda rows: np.zeros(len(rows))) for k in range(e21.K)}
    out = e21._grr_arms("ATE", X, Y, D, folds, inner, gamma, 0.5, 0, True)
    rec = out["ATE|lin-SQ|TMLE"]
    a = np.where(D == 1, 1e160, -0.5e160)
    eps = (np.sum((a / 1e160) * Y) / np.sum((a / 1e160) ** 2)) / 1e160
    assert rec["epsilon"] == pytest.approx(eps) and rec["epsilon"] != 0.0


def test_summary_and_s1_are_scale_stable() -> None:
    raw = _fake_raw(4)
    arm = raw["arm"] == "ATE|lin-SQ|ARW"
    raw.loc[arm, "estimate"] = 4.0 + np.array([1e200, -1e200, 1e200, -1e200])
    raw.loc[arm, "se"] = 1e199
    raw.loc[arm, "status"] = "ok"
    summ = e21.summarise(raw)
    row = summ[summ["arm"] == "ATE|lin-SQ|ARW"].iloc[0]
    assert row["sd"] == pytest.approx(np.std([1, -1, 1, -1], ddof=1) * 1e200)
    assert row["rmse"] == pytest.approx(1e200)
    assert np.isfinite(row["se_ratio"])
    s1 = e21.s1_intervals(raw, Seeds(21, entropy=PILOT_ENTROPY))
    assert np.isfinite(s1["ATE|lin-SQ"][0])


# ---------------------------------------------------------------- review fixes (round 3)


def test_summary_reductions_and_s1_near_the_float_limit() -> None:
    raw = _fake_raw(100)
    arm = raw["arm"] == "ATE|lin-SQ|ARW"
    raw.loc[arm, "se"] = 3.6e306
    raw.loc[arm, "max_abs_alpha"] = 1e308
    summ = e21.summarise(raw)
    row = summ[summ["arm"] == "ATE|lin-SQ|ARW"].iloc[0]
    assert row["se_mean"] == pytest.approx(3.6e306) and row["max_weight_median"] == pytest.approx(
        1e308
    )
    rw = raw["arm"] == "ATE|lin-SQ|RW"
    rng = np.random.default_rng(13)
    # one RW error of 5e303 among zeros, ARW errors of order 1e-5: the scale quotient
    # 5e303 / 1e-5 overflows, the SD ratio (about 9e307) does not
    e_rw = np.zeros(rw.sum())
    e_rw[0] = 5e303
    raw.loc[rw, "estimate"] = 4.0 + e_rw
    raw.loc[arm, "estimate"] = 4.0 + rng.uniform(-1e-5, 1e-5, size=arm.sum())
    raw.loc[arm | rw, "status"] = "ok"
    ratio, lo, hi = e21.s1_intervals(raw, Seeds(21, entropy=PILOT_ENTROPY))["ATE|lin-SQ"]
    expected = (
        np.std(e_rw / 5e303, ddof=1)
        / np.std(raw.loc[arm, "estimate"].to_numpy() - 4.0, ddof=1)
        * 5e303
    )
    assert np.isfinite(ratio) and ratio == pytest.approx(expected, rel=1e-9)
    # resamples that repeat the outlier have a ratio above the largest float: the
    # interval is then undefined (S1 is not written), never an overflowed number
    assert (np.isnan(lo) and np.isnan(hi)) or (np.isfinite(lo) and lo <= hi)
