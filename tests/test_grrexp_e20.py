"""E-20 support code: DGP20 and the rejection sampler, Stage 0 (population ratio,
moments, predictions), the cross-fitted two-sample estimator and its three variance
estimators, the replication records, aggregation, H20 and the tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import e12, e20  # noqa: E402
from grrexp.metrics import Z_95  # noqa: E402
from grrexp.seeds import PILOT_ENTROPY, Seeds, fold_ids  # noqa: E402


@pytest.fixture(scope="module")
def stage0():
    return e20.stage0()


# ---------------------------------------------------------------- DGP


def test_dgp_check_reproduces_the_registered_second_moment() -> None:
    d = e20.dgp_check()
    assert d["ok"]
    assert abs(d["E_r0_sq_closed_form"] - 1.54378) <= 5e-6


def test_rejection_sampler_draws_the_tilted_law() -> None:
    """Target means agree with the quadrature means E_0[r0 z] within 5 standard errors."""
    rng = np.random.default_rng(11)
    Zt = e20.draw_target(rng, 200_000)
    assert Zt.shape == (200_000, 2)
    assert np.all(np.abs(Zt) <= np.sqrt(3.0))
    Z, w = e20.quadrature(64)
    rr = e20.r0(Z)
    mean = w @ (rr[:, None] * Z)
    sd = np.sqrt(w @ (rr[:, None] * (Z - mean) ** 2))
    assert np.all(np.abs(Zt.mean(axis=0) - mean) <= 5 * sd / np.sqrt(len(Zt)))


def test_draw_is_deterministic_and_has_the_requested_sizes() -> None:
    a = e20.draw(Seeds(20, entropy=PILOT_ENTROPY).data(3, 0), 800, 100)
    b = e20.draw(Seeds(20, entropy=PILOT_ENTROPY).data(3, 0), 800, 100)
    assert a[0].shape == (800, 2) and a[1].shape == (800,) and a[2].shape == (100, 2)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_gamma0_is_in_the_ols_span_and_log_r0_in_the_ratio_span() -> None:
    rng = np.random.default_rng(2)
    Z = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(50, 2))
    c, *_ = np.linalg.lstsq(e20.outcome_features(Z), e20.gamma0(Z), rcond=None)
    assert np.allclose(e20.outcome_features(Z) @ c, e20.gamma0(Z), atol=1e-12)
    b = np.array([-np.log(e20.normaliser()), *e20.TILT])
    assert np.allclose(e20.ratio_features(Z) @ b, np.log(e20.r0(Z)), atol=1e-12)


# ---------------------------------------------------------------- Stage 0


def test_stage0_certifies_the_closed_form_ratio(stage0) -> None:
    assert stage0["certified"] and all(stage0["checks"].values())
    assert stage0["beta_closed_form_max_abs_diff"] <= 1e-12
    assert stage0["support_margin_kind"] == "unbounded"
    assert stage0["sigma2_S"] == pytest.approx(1.5437799509, abs=1e-9)
    json.dumps(stage0, allow_nan=False)  # JSON-compatible


def test_predictions_follow_their_closed_forms(stage0) -> None:
    th, s2S, s2T = stage0["theta0"], stage0["sigma2_S"], stage0["sigma2_T"]
    for n, m in e20.CELLS:
        p = stage0["predictions"][f"{n},{m}"]
        assert p["c"]["V_2s"] == pytest.approx(2 * stats.norm.cdf(Z_95) - 1, abs=1e-12)
        s2 = s2S / n + s2T / m
        assert p["sd"] == pytest.approx(np.sqrt(s2), rel=1e-12)
        c_src = 2 * stats.norm.cdf(Z_95 * np.sqrt(s2S / n / s2)) - 1
        assert p["c"]["V_src"] == pytest.approx(c_src, abs=1e-6)
        pi = m / (n + m)
        # V_pool estimates (V_CS + theta0^2 (1 - pi)/pi)/N; its coverage at sd s
        se_pool = np.sqrt((s2S / (1 - pi) + s2T / pi + th**2 * (1 - pi) / pi) / (n + m))
        c_pool = 2 * stats.norm.cdf(Z_95 * se_pool / np.sqrt(s2)) - 1
        assert p["c"]["V_pool"] == pytest.approx(c_pool, abs=1e-6)


# ---------------------------------------------------------------- estimator


def _manual(Zs, Y, Zt, fs, ft, ratio_arm, freqs, outcome="OLS"):
    n, m = len(Y), len(Zt)
    folds, rw, gt = [], np.empty(n), np.empty(m)
    for k in range(e20.K):
        status, res = e20.fit_ratio(ratio_arm, Zs[fs != k], Zt[ft != k])
        assert status == "ok"
        rr = res.predict_ratio(Zs[fs == k])
        if outcome == "OLS":
            c, *_ = np.linalg.lstsq(e20.outcome_features(Zs[fs != k]), Y[fs != k], rcond=None)
            pred = lambda Z, c=c: e20.outcome_features(Z) @ c  # noqa: E731
        else:
            pred = e20._rff(Zs[fs != k], Y[fs != k], freqs)
        rw[fs == k] = rr * (Y[fs == k] - pred(Zs[fs == k]))
        gt[ft == k] = pred(Zt[ft == k])
        folds.append(np.mean(pred(Zt[ft == k])) + np.mean(rw[fs == k]))
    theta = np.mean(folds)
    return theta, rw, gt


@pytest.mark.parametrize("arm", e20.ARMS)
def test_cross_fit_matches_a_direct_computation(arm) -> None:
    seeds = Seeds(20, entropy=PILOT_ENTROPY)
    Zs, Y, Zt = e20.draw(seeds.data(1, 0), 600, 400)
    fg = seeds.folds(1, 0)
    fs, ft = fold_ids(600, e20.K, fg), fold_ids(400, e20.K, fg)
    freqs = e20.rff_frequencies(seeds.baseline(1, 0))
    ratio_arm, outcome = arm.split("+")
    rec, detail, _ = e20._cross_fit(Zs, Y, Zt, fs, ft, ratio_arm, outcome, freqs)
    assert isinstance(rec, dict), detail
    theta, rw, gt = _manual(Zs, Y, Zt, fs, ft, ratio_arm, freqs, outcome)
    assert rec["estimate"] == pytest.approx(theta, abs=1e-12)
    n, m = 600, 400
    s2S, s2T = np.mean(rw**2), np.mean((gt - theta) ** 2)
    assert rec["se_V_2s"] == pytest.approx(np.sqrt(s2S / n + s2T / m), rel=1e-12)
    assert rec["se_V_src"] == pytest.approx(np.sqrt(s2S / n), rel=1e-12)
    pi = m / (n + m)
    # the pooled influence function, written out per observation
    v_pool = (np.sum((gt / pi - theta) ** 2) + np.sum((rw / (1 - pi) - theta) ** 2)) / (n + m)
    assert rec["se_V_pool"] == pytest.approx(np.sqrt(v_pool / (n + m)), rel=1e-12)
    assert rec["ratio_gradient_max"] <= e20.SAMPLE_TOL


def test_ukl_ratio_is_positive_and_sq_ratio_is_linear() -> None:
    seeds = Seeds(20, entropy=PILOT_ENTROPY)
    Zs, _, Zt = e20.draw(seeds.data(2, 0), 3000, 3000)
    _, ukl = e20.fit_ratio("UKL", Zs, Zt)
    _, sq = e20.fit_ratio("SQ", Zs, Zt)
    assert ukl.status == "ok" and sq.status == "ok"
    assert np.all(ukl.predict_ratio(Zs) > 0)
    assert np.allclose(np.log(ukl.predict_ratio(Zs)), e20.ratio_features(Zs) @ ukl.beta)
    assert np.allclose(sq.predict_ratio(Zs), e20.ratio_features(Zs) @ sq.beta / 2.0)
    # both satisfy the two-sample balance equations on the regressors at the tolerance
    for res in (ukl, sq):
        imb = np.mean(res.predict_ratio(Zs)[:, None] * e20.ratio_features(Zs), axis=0) - np.mean(
            e20.ratio_features(Zt), axis=0
        )
        assert np.max(np.abs(imb)) <= e20.SAMPLE_TOL


def test_a_failed_ratio_fit_fails_the_arm_with_its_status(monkeypatch) -> None:
    real = e20.fit_ratio

    def fake(arm, Zs, Zt):
        if arm == "SQ":
            status, res = real(arm, Zs, Zt)
            return "linesearch", res
        return real(arm, Zs, Zt)

    monkeypatch.setattr(e20, "fit_ratio", fake)
    out = e20.replicate((1, 0, PILOT_ENTROPY))
    assert out["SQ+OLS"]["status"] == "linesearch" and np.isnan(out["SQ+OLS"]["estimate"])
    assert out["UKL+OLS"]["status"] == "ok" and out["UKL+RFF"]["status"] == "ok"
    assert json.loads(out["SQ+OLS"]["fold_status"])[0] == ["0", "ratio", "linesearch"]


def test_a_prediction_outside_the_domain_fails_the_arm(monkeypatch) -> None:
    real = e20.fit_ratio

    class Wrapped:
        def __init__(self, res):
            self._res = res
            self.fit = res.fit

        def classify(self, Z):
            r, outside, nonfinite = self._res.classify(Z)
            outside = outside.copy()
            outside[0] = True
            return r, outside, nonfinite

    def fake(arm, Zs, Zt):
        status, res = real(arm, Zs, Zt)
        return status, Wrapped(res)

    monkeypatch.setattr(e20, "fit_ratio", fake)
    out = e20.replicate((1, 0, PILOT_ENTROPY))
    assert all(out[a]["status"] == "domain_prediction" for a in e20.ARMS)


# ---------------------------------------------------------------- records and aggregation


def test_replicate_is_deterministic_and_records_every_arm() -> None:
    a = e20.replicate((3, 1, PILOT_ENTROPY))
    b = e20.replicate((3, 1, PILOT_ENTROPY))
    assert set(a) == set(e20.ARMS)
    for arm in e20.ARMS:
        assert a[arm]["status"] == "ok"
        assert a[arm]["estimate"] == b[arm]["estimate"]
        assert a[arm]["se_V_pool"] == b[arm]["se_V_pool"]


def test_ratio_balance_passes_and_stops_on_a_violation() -> None:
    tasks = e20.tasks_for(PILOT_ENTROPY, 1, cells=[0])
    raw = e20.raw_frame(tasks, [e20.replicate(t) for t in tasks])
    out = e20.ratio_balance(raw)
    assert out["records_checked"] == len(e20.ARMS)
    bad = raw.copy()
    bad.loc[0, "ratio_gradient_max"] = 2e-8
    with pytest.raises(AssertionError):
        e20.ratio_balance(bad)


def _aggregate(raw, stage0, seeds):
    e20.ratio_balance(raw)
    summ = e20.summarise(raw, stage0)
    summ = summ.assign(**e12.family1_mcse(raw, summ, seeds))
    verdict, tests = e20.family_h20(summ)
    summ["family_verdict"] = json.dumps(verdict.sentence() if verdict else None)
    return summ, verdict, tests


def test_aggregation_h20_and_tables(stage0) -> None:
    tasks = e20.tasks_for(PILOT_ENTROPY, 3)
    raw = e20.raw_frame(tasks, [e20.replicate(t) for t in tasks])
    assert len(raw) == 5 * 3 * len(e20.ARM_LABELS)
    summ, verdict, tests = _aggregate(raw, stage0, Seeds(20, entropy=PILOT_ENTROPY))
    assert len(summ) == 5 * len(e20.ARM_LABELS)
    assert sum(k.startswith("(a)") for k in tests) == 4  # min(n, m) >= 1000
    assert sum(k.startswith("(b)") for k in tests) == 5
    assert sum(k.startswith("(c)") for k in tests) == 5
    assert verdict.hypothesis == "H20"
    assert summ.loc[summ["estimator"] != e20.THEORY_ARM, "c_pred"].isna().all()
    tabs, _ = e20.tables(summ)
    body = tabs["tab_E20"]
    assert "\\endfirsthead" in body and "\\endhead" in body
    assert body.count(" \\\\") == 2 + 5 * len(e20.ARMS)  # the head twice, then the rows
    assert "No replication failed" in tabs["tab_E20_status"]
    assert "-0.000" not in body


def test_summarise_aggregates_one_cell_at_a_time(stage0) -> None:
    for ci in range(len(e20.CELLS)):
        tasks = e20.tasks_for(PILOT_ENTROPY, 2, cells=[ci])
        raw = e20.raw_frame(tasks, [e20.replicate(t) for t in tasks])
        summ, verdict, tests = _aggregate(raw, stage0, Seeds(20, entropy=PILOT_ENTROPY))
        assert set(summ["cell"]) == {ci} and len(summ) == len(e20.ARM_LABELS)
        assert tests and verdict is not None


def test_status_table_lists_failed_cells(stage0, monkeypatch) -> None:
    real = e20.fit_ratio

    def fake(arm, Zs, Zt):
        status, res = real(arm, Zs, Zt)
        return ("maxit", res) if arm == "SQ" else (status, res)

    monkeypatch.setattr(e20, "fit_ratio", fake)
    tasks = e20.tasks_for(PILOT_ENTROPY, 2, cells=[0])
    raw = e20.raw_frame(tasks, [e20.replicate(t) for t in tasks])
    summ, _, _ = _aggregate(raw, stage0, Seeds(20, entropy=PILOT_ENTROPY))
    tabs, _ = e20.tables(summ)
    assert "maxit: 2" in tabs["tab_E20_status"]
    assert "SQ + OLS" in tabs["tab_E20_status"]


def test_fmt_has_no_negative_zero() -> None:
    assert e20._fmt(-0.00001) == "0.000"
    assert e20._fmt(-0.0011) == "-0.001"
    assert e20._fmt(float("nan")) == "--"
