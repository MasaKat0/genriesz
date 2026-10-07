"""E-12 support code: warnings, statuses, balance records, Stage 0 JSON, aggregation."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import baselines, bootstrap, e12, outputs  # noqa: E402
from grrexp.seeds import PILOT_ENTROPY, Seeds  # noqa: E402

import genriesz as gr  # noqa: E402

# ------------------------------------------------ warnings and statuses (§1.1, §1.3 B)


def test_run_recording_warnings_counts_and_flags_convergence() -> None:
    from sklearn.exceptions import ConvergenceWarning

    def fit():
        warnings.warn("plain", UserWarning, stacklevel=1)
        warnings.warn("plain", UserWarning, stacklevel=1)
        warnings.warn("divide by zero encountered in matmul", RuntimeWarning, stacklevel=1)
        return 7

    res, counts, converged = baselines.run_recording_warnings(fit)
    assert res == 7 and converged
    assert counts == {"UserWarning: plain": 2}  # the Accelerate matmul warning is the only filter

    def failing():
        warnings.warn("lbfgs failed", ConvergenceWarning, stacklevel=1)

    _, counts, converged = baselines.run_recording_warnings(failing)
    assert not converged and counts == {"ConvergenceWarning: lbfgs failed": 1}


def test_recorded_statuses_cover_baselines_and_outcome_failures() -> None:
    statuses = outputs.recorded_statuses()
    for s in (
        *gr.STATUSES,
        *baselines.BASELINE_STATUSES,
        "outcome_optimizer_failure",
        "outcome_maxit",
    ):
        assert s in statuses
    assert "outcome_ok" not in statuses


# ---------------------------------------------------------------- genriesz records


def _ate_sample(n=400, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-1, 1, size=(n, 2))
    D = (rng.uniform(size=n) < 1 / (1 + np.exp(-Z[:, 0]))).astype(float)
    Y = D + Z[:, 0] + rng.standard_normal(n)
    X = np.column_stack([D, Z])
    basis = gr.TreatmentInteractionBasis(
        base_basis=gr.CallableBasis(lambda z: np.column_stack([np.ones(len(z)), z]))
    )
    return X, Y, basis


def test_grr_functional_records_training_balance_per_fold() -> None:
    X, Y, basis = _ate_sample()
    basis.fit(X)
    folds = np.arange(len(Y)) % 5
    res = gr.grr_functional(
        X=X,
        Y=Y,
        m=gr.ATEFunctional(0),
        basis=basis,
        generator=gr.SquaredGenerator(C=0.0),
        riesz_penalty=None,
        riesz_lam=0.0,
        riesz_tol=1e-8,
        outcome_models="shared",
        outcome_link="identity",
        outcome_penalty="l2",
        outcome_lam=0.0,
        fold_ids=folds,
        estimators=("arw",),
        expose_alpha_values=True,
    )
    assert res.success
    tb = res.diagnostics["train_imbalance"]
    assert len(tb["max"]) == len(tb["scale"]) == 5
    for imb, scale, k in zip(tb["max"], tb["scale"], range(5), strict=True):
        tr = folds != k
        M = gr.ATEFunctional(0).m_basis_matrix(X[tr], basis)
        assert scale == pytest.approx(max(1.0, float(np.max(np.abs(np.mean(M, axis=0))))))
        assert imb <= 1e-8 * scale


def test_rw_full_inference_matches_least_squares() -> None:
    X, Y, basis = _ate_sample(seed=1)
    gen = gr.SquaredGenerator(C=0.0)
    grr = gr.GRRGLM(
        basis=basis, generator=gen, functional=gr.ATEFunctional(0), penalty=None, lam=0.0
    )
    assert grr.fit(X, tol=1e-8).status == "ok"
    est = gr.rw_full_inference(grr=grr, X=X, Y=Y)
    Phi = np.asarray(basis(X))
    coef, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    M = gr.ATEFunctional(0).m_basis_matrix(X, basis)
    a = grr.predict_alpha(X)
    theta = float(np.mean(a * Y))
    psi = M @ coef + a * (Y - Phi @ coef) - theta
    assert est.estimate == pytest.approx(theta, abs=1e-14)
    assert est.se == pytest.approx(float(np.std(psi, ddof=1) / np.sqrt(len(Y))), rel=1e-10)


# ---------------------------------------------------------------- Stage 0 JSON (§1.9)


def test_stage0_rows_are_json_compatible_with_unbounded_and_failed_cells(monkeypatch) -> None:
    def fake_cell(arm, rs, s, points):
        if arm == "BKL1":
            return {
                "status": "maxit",
                "max_gradient": 1.0,
                "min_hessian_eig": np.nan,
                "beta": np.full(10, np.nan),
                **{k: np.nan for k in e12.POPULATION_KEYS},
            }
        ref = {("UKL1", "Omit", 1.25): (7.2906549, 7.6013032)}.get((arm, rs, s), (4.0, 4.0))
        e2 = {("SQ", "Include", 0.5): 4.3664466, ("UKL1", "Include", 0.5): 4.4173369}
        return {
            "status": "ok",
            "max_gradient": 1e-14,
            "min_hessian_eig": 1.0,
            "beta": np.zeros(10),
            "theta_star": 1.0,
            "b": 0.0,
            "sigma2_true": ref[0],
            "sigma2_se": ref[1],
            "E_alpha2": e2.get((arm, rs, s), 4.0),
        }

    monkeypatch.setattr(e12, "population_cell", fake_cell)
    monkeypatch.setattr(
        e12, "support_dual_margin", lambda arm, rs, beta: 0.5 if arm.startswith("BP") else np.inf
    )
    rows, gate = e12.stage0()
    json.dumps({"cells": rows, "reference_gate": gate}, allow_nan=False)
    by = {(r["arm"], r["set"], r["s"]): r for r in rows}
    sq = by[("SQ", "Include", 0.5)]
    assert sq["certified"] and sq["support_dual_margin"] is None
    assert sq["support_dual_margin_kind"] == "unbounded" and sq["c_n"] is not None
    bkl = by[("BKL1", "Include", 0.5)]
    assert not bkl["certified"] and bkl["support_dual_margin_kind"] == "unavailable"
    assert bkl["c_n"] is None and bkl["b"] is None and bkl["beta_diff_48_64"] is None
    assert all(v["ok"] for v in gate.values())


# ---------------------------------------------------------------- aggregation


def _raw(status, estimate=1.0, ratio=1e-10, arm="SQ|RW_full", reps=4):
    rows = []
    for rep in range(reps):
        rows.append(
            {
                "cell": 0,
                "s": 0.5,
                "set": "Include",
                "n": 500,
                "rep": rep,
                "arm": arm,
                "estimate": estimate + 0.01 * rep if status == "ok" else np.nan,
                "se": 0.1 if status == "ok" else np.nan,
                "max_abs_alpha": 3.0 if status == "ok" else np.nan,
                "ess": 400.0,
                "train_imbalance": 1e-12,
                "train_balance_ratio": ratio if status == "ok" else np.nan,
                "train_fits": 1 if status == "ok" else 0,
                "eval_imbalance": np.nan,
                "n_warnings": 0.0,
                "wp_gap": np.nan,
                "wp_n_at_boundary": np.nan,
                "wp_min_margin": np.nan,
                "status": status,
                "warnings": "{}",
                "fold_status": "[]",
                "wp_verdict": "",
                "wp_solver_status": "",
                "n_check_warnings": 0,
                "check_sq_rw_equals_ols": 1e-13,
                "check_sq_rw_equals_ols_bound": 1e-10,
            }
        )
    return pd.DataFrame(rows)


def test_balance_checks_stop_on_training_imbalance() -> None:
    assert e12.balance_checks(_raw("ok"))["fits_checked"] == 4
    with pytest.raises(AssertionError, match="training imbalance"):
        e12.balance_checks(_raw("ok", ratio=1e-6))
    raw = _raw("ok")
    raw.loc[2, "check_sq_rw_equals_ols"] = 2e-10  # above its own bound (Amendment 5)
    with pytest.raises(AssertionError, match="check_sq_rw_equals_ols"):
        e12.balance_checks(raw)


def _population(certified):
    return [
        {
            "arm": a,
            "set": rs,
            "s": s,
            "certified": certified,
            "b": 0.0,
            "sigma2_true": 4.0,
            "c_n": {str(n): 0.95 for n in e12.N_VALUES} if certified else None,
        }
        for a in e12.ALL_ARMS
        for rs in e12.SETS
        for s in e12.S_VALUES
    ]


def test_summarise_reports_coverage_and_failures_without_successes() -> None:
    summ = e12.summarise(_raw("maxit"), _population(True))
    r = summ.iloc[0]
    assert r["R_s"] == 0 and r["coverage"] == 0.0 and r["failure_rate"] == 1.0
    assert r["moment_status"] == "fewer_than_two" and "bias" not in summ.columns


def test_moment_status_marks_negative_radicand_without_truncating() -> None:
    assert e12.moment_status(np.array([0.0, 1.0])) == "negative_radicand"  # m4 - s^4 < 0
    assert e12.moment_status(np.array([1.0, 1.0, 1.0])) == "zero_variance"
    assert e12.moment_status(np.random.default_rng(0).standard_normal(500)) == "ok"
    summ = e12.summarise(_raw("ok", reps=2), _population(True))
    r = summ.iloc[0]
    assert r["moment_status"] == "negative_radicand" and np.isnan(r.get("root_n_sd_mcse", np.nan))
    fam, unavailable = e12.families(summ.assign(n=2000), _raw("ok", reps=2).assign(n=2000))
    assert any("sd_ratio (negative_radicand)" in u for u in unavailable["H12-Inf"])


def test_predictions_only_for_certified_cells_and_predicted_estimators() -> None:
    raw = pd.concat([_raw("ok"), _raw("ok", arm="SQ|TMLE_cf")])
    summ = e12.summarise(raw, _population(True)).set_index("arm")
    assert summ.loc["SQ|RW_full", "c_pred"] == 0.95
    assert np.isnan(summ.loc["SQ|TMLE_cf", "c_pred"])
    summ = e12.summarise(raw, _population(False))
    assert "c_pred" not in summ.columns


def test_family1_order_and_undefined_se_ratio() -> None:
    raw = _raw("ok", reps=2)  # two estimates: half of the resamples repeat one, SD 0
    summ = e12.summarise(raw, _population(True))
    out = e12.family1_mcse(raw, summ, Seeds(12, entropy=PILOT_ENTROPY))
    assert out["ci_length_mean_mcse_lo"][0] == pytest.approx(2 * 1.96 * 0.1)
    assert np.isnan(out["se_ratio_mcse_lo"][0])


def test_bootstrap_nonfinite_is_reported_only_on_request() -> None:
    fb = bootstrap.FamilyBootstrap(Seeds(12, entropy=PILOT_ENTROPY), 1)
    with pytest.raises(FloatingPointError):
        fb.replicate(lambda idx: np.full(idx.shape[0], np.inf), 5)
    fb = bootstrap.FamilyBootstrap(Seeds(12, entropy=PILOT_ENTROPY), 1)
    assert np.all(
        np.isinf(fb.replicate(lambda idx: np.full(idx.shape[0], np.inf), 5, finite=False))
    )


def test_vectorized_branch_fn_gives_the_row_by_row_signs() -> None:
    X = np.column_stack([np.array([1.0, 0.0, 1.0, 0.0]), np.arange(4.0)])
    v = np.array([0.3, -0.2, 1.1, -0.7])

    def rowwise(x):
        return int(np.atleast_1d(x)[0] == 1)

    for make in (
        lambda b: gr.UKLGenerator(C=1.0, branch_fn=b),
        lambda b: gr.BPGenerator(omega=0.5, C=0.0, branch_fn=b),
    ):
        a_vec = make(e12.sign).inv_grad(X, v)
        a_row = make(rowwise).inv_grad(X, v)
        assert np.array_equal(a_vec, a_row)


def test_balance_checks_cover_successful_folds_of_failed_replications() -> None:
    raw = _raw("maxit")
    raw.loc[1, ["train_fits", "train_balance_ratio"]] = [2, 1e-6]  # 2 folds fitted, then a failure
    with pytest.raises(AssertionError, match="training imbalance"):
        e12.balance_checks(raw)
    raw = _raw("ok")
    raw.loc[0, "train_fits"] = 0  # a successful RW_full record without its fit's balance
    with pytest.raises(AssertionError, match="without the balance"):
        e12.balance_checks(raw)


def test_undefined_tests_stay_in_the_family_and_zero_radicand_still_tests_coverage(
    monkeypatch,
) -> None:
    raw = _raw("ok").assign(n=2000)
    monkeypatch.setattr(e12.metrics, "moment_terms", lambda x: (1.0, 1.0))  # m4 - s^4 = 0
    summ = e12.summarise(raw, _population(True)).assign(n=2000)
    r = summ.iloc[0]
    assert r["moment_status"] == "zero_radicand" and r["root_n_sd_mcse"] == 0.0
    fam, unavailable = e12.families(summ, raw)
    inf = fam["H12-Inf"]
    key = "SQ|RW_full|s=0.5|Include|n=2000"
    assert set(inf.labels) == {f"{key}|{t}" for t in ("coverage", "failure", "bias", "sd_ratio")}
    assert dict(zip(inf.labels, inf.pvalues, strict=True))[f"{key}|sd_ratio"] == e12.UNAVAILABLE_P
    assert unavailable["H12-Inf"] == [f"{key}|sd_ratio (zero_radicand)"]
