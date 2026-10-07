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
