"""Numerical discrepancies are recorded; library failures never supply replacement fits."""

import ast
import collections
import itertools
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import baselines, e12, e13, e14, e18, e19, e20, e21, outputs, population  # noqa: E402

import genriesz as gr  # noqa: E402

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks" / "experiments"


def cell(name, index):
    return "".join(json.loads((NOTEBOOKS / name).read_text())["cells"][index]["source"])


def test_population_uses_the_analytic_hessian_and_library_defaults(monkeypatch):
    actual = optimize.minimize
    calls = []

    def recording(fun, x0, **kwargs):
        calls.append(kwargs)
        np.testing.assert_allclose(kwargs["hess"](x0), np.diag([0.25, 1.0]))
        return actual(fun, x0, **kwargs)

    monkeypatch.setattr(optimize, "minimize", recording)
    Phi = np.diag([1.0, 2.0])
    result = population.solve(generator=gr.SquaredGenerator(), X=Phi, w=np.full(2, 0.5),
                              Phi=Phi, M=np.eye(2), offset=np.zeros(2))
    assert result.status == "ok"
    assert calls[0]["method"] == "trust-ncg"
    assert calls[0]["tol"] is None and calls[0]["options"] == {}
    np.testing.assert_allclose(result.beta, [2.0, 0.5])


@pytest.mark.parametrize("nonfinite", [False, True])
def test_eb_library_failure_preserves_message_without_weights(monkeypatch, nonfinite):
    message = "Library optimization failure."
    monkeypatch.setattr(optimize, "minimize", lambda *a, **kw: optimize.OptimizeResult(
        x=np.array([np.nan if nonfinite else 0.0]), fun=np.nan if nonfinite else 1.0,
        jac=np.array([np.nan if nonfinite else 1.0]), success=nonfinite, status=2,
        message=message, nit=1))
    weights, diagnostic = baselines.eb_dual_newton(np.ones((2, 1)), np.ones(1), 2,
                                                  return_diagnostics=True)
    assert weights is None
    assert diagnostic["status"] == ("nonfinite" if nonfinite else "optimizer_failure")
    assert diagnostic["message"] == message
    json.dumps(diagnostic, allow_nan=False)


def test_eb_unexpected_exception_still_propagates(monkeypatch):
    def broken(*args, **kwargs):
        raise TypeError("Unexpected library exception.")

    monkeypatch.setattr(optimize, "minimize", broken)
    with pytest.raises(TypeError, match="Unexpected"):
        baselines.eb_dual_newton(np.ones((2, 1)), np.ones(1), 2)


def test_e12_all_failed_independent_eb_fits_are_recorded():
    fits = {"treated": {"status": "optimizer_failure", "message": "Library failure."},
            "control": {"status": "optimizer_failure", "message": "Library failure."}}
    result = {"EB|RW_full": {"status": "ok", "train_fits": 1, "train_balance_ratio": 0.1},
              "_checks": {"eb_dual_newton_max_diff": None, "eb_fits": json.dumps(fits)},
              "_check_warnings": 0}
    raw = e12.raw_frame([(0, 0, 20261008, None)], [result])
    diagnostic = e12.balance_checks(raw)
    assert not diagnostic["eb_ok"] and diagnostic["eb_max"] is None
    assert diagnostic["eb_fit_status_counts"] == {"optimizer_failure": 2}
    # the per-replication records stay in raw
    assert json.loads(raw.loc[raw["arm"] == "EB|RW_full", "check_eb_fits"].iloc[0]) == fits
    json.dumps(diagnostic, allow_nan=False)


def test_e21_records_failed_independent_eb_fit_and_continues(monkeypatch):
    X = np.array([[1.0, 0.0], [0.0, 0.0]])
    mdl = SimpleNamespace(predict_alpha=lambda X: np.where(X[:, 0] == 1, 2.0, -2.0))
    diagnostic = {"status": "optimizer_failure", "message": "Library failure."}
    monkeypatch.setattr(baselines, "eb_dual_newton", lambda *a, **kw: (None, diagnostic))
    result = e21.eb_check("ATE", mdl, X, 0.5)
    assert not result["ok"] and result["max_relative_difference"] is None
    assert result["fits"] == [diagnostic, diagnostic]


def test_e14_nonfinite_design_records_failed_fits_before_optimization(monkeypatch):
    row = {"generator": "SQ"}
    monkeypatch.setattr(e14, "Design", lambda _: SimpleNamespace(name="SQ", pi1=0.5))
    monkeypatch.setattr(e14, "draw", lambda *a: (np.array([0.0, 1.0]), np.zeros((2, 4)),
                                               np.ones(2)))
    monkeypatch.setattr(e14, "design_matrix", lambda *a: np.full((2, 6), np.nan))
    fit = Mock(side_effect=AssertionError("Non-finite input must not be fitted."))
    monkeypatch.setattr(e14, "fit", fit)
    result = e14._replicate((0, 0, 20261008, row))
    assert not result["design_diagnostics"]["finite"]
    assert all(result[p]["status"] == "nonfinite" for p in e14.PENALTIES)
    fit.assert_not_called()


def test_e18_radius_discrepancy_does_not_discard_a_library_success(monkeypatch):
    radius = e18.RADIUS["SQ"]
    beta = np.array([radius + 1.0, 0.0, 0.0, 0.0])

    class Model:
        beta_ = beta

        def fit(self, X):
            return SimpleNamespace(status="ok", beta=beta, message="Library success.")

    monkeypatch.setattr(gr, "GRRGLM", lambda **kwargs: Model())
    status, fitted, diagnostic = e18.fit_candidate(
        0, np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        return_diagnostics=True)
    assert status == "ok" and fitted is not None
    assert all(not d["radius_ok"] for d in diagnostic)
    assert diagnostic[0]["l1_beta"] == radius + 1.0


def test_e18_selection_domain_failure_is_recorded(monkeypatch):
    beta = np.zeros(e18.N_COLS[1])
    beta[0] = 2.0  # BKL's treated dual coordinate is outside its domain
    fitted = e18._Fit("BKL1", 1, {1.0: beta, 0.0: beta})
    monkeypatch.setattr(e18, "fit_candidate", lambda *a, **kw: ("ok", fitted, []))
    X = np.zeros((12, 4))
    X[:, 0] = np.arange(12) % 2
    P = e18.legendre_features(X[:, 1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        record, fits = e18._fold_selection("18B", X, X[:, 1:], np.arange(6),
                                           np.arange(6, 12), P, P, np.full(12, 0.5), 1e6)
    assert record["n_failed"] == e18.A0 and all(f is None for f in fits)
    assert record["diagnostics"][0]["status"] in ("domain_prediction", "nonfinite")
    assert not record["diagnostics"][0]["predictions"]["domain_ok"]


def test_recorder_retains_failed_checks_as_json():
    rec = outputs.RunRecorder.__new__(outputs.RunRecorder)
    rec.closed, rec.diagnostics = False, {}
    rec.record_diagnostics("finite", {"value": np.inf, "ok": False})
    assert rec.diagnostics == {"finite": [{"value": None, "ok": False}]}
    json.dumps(rec.diagnostics, allow_nan=False)


@pytest.mark.parametrize("values", [[0.0, np.inf], [0.0, 1e308]])
def test_e18_nonfinite_risk_moments_are_unavailable(values):
    with np.errstate(all="ignore"):
        mean, lo, hi, diagnostic = e18.bernstein_interval(
            values, 4.0, 0.01, return_diagnostics=True)
    assert not diagnostic["finite"] and np.isnan([mean, lo, hi]).all()


def test_e18_failed_candidate_cannot_enter_the_oracle_risk(monkeypatch):
    treated = np.zeros(e18.N_COLS[1])
    treated[0] = 4.0
    control = np.zeros(e18.N_COLS[1])
    control[0] = 4.0 / 3.0
    fitted = e18._Fit("SQ", 1, {1.0: treated, 0.0: control})
    monkeypatch.setattr(fitted, "conj", lambda d, P: np.full(len(P), np.nan))
    monkeypatch.setattr(e18, "fit_candidate", lambda *a, **kw: ("ok", fitted, []))
    X = np.zeros((12, 4))
    X[:, 0] = np.arange(12) % 2
    P = e18.legendre_features(X[:, 1:])
    prop = np.full(12, 0.25)
    result, fits = e18._fold_selection("18B", X, X[:, 1:], np.arange(6),
                                       np.arange(6, 12), P, P, prop, 1e6)
    assert result["n_failed"] == e18.A0 and all(f is None for f in fits)
    assert result["risk_star"] == pytest.approx(float(e18._risk_values(2.0, -2.0, prop).mean()))


@pytest.mark.parametrize("number,payload", [
    (10, ([], {"value": {"computed": 2.0, "reference": 1.0, "ok": False}})),
    (11, []),
    (12, {"generators": [], "bkl_zero_offset_infeasible": False}),
    (13, {"part_a_ok": False, "part_a": [], "generators": []}),
    (16, {"inference": {"certified": False, "checks": {}, "theta0": 0.0,
                        "V_star": 1.0, "c_n": {}}, "counterexamples": {}}),
    (17, {"certified": False, "checks": {}, "predictions": {}}),
    (20, {"ok": False, "designs": {}, "radius": {}, "ukl_18b": {"checks": {}}}),
    (22, {"ok": False, "checks": {}, "differences": {}}),
])
def test_stage0_notebook_records_failed_numerical_checks(number, payload):
    path = next(NOTEBOOKS.glob(f"{number:02d}_*.ipynb"))
    rec = Mock()
    exp = {10: 12, 11: 13, 12: 14, 13: 15, 16: 19, 17: 20, 20: 18, 22: 24}[number]
    module = SimpleNamespace(stage0=lambda: payload, dgp_check=lambda: {"ok": False},
                             PART_A_TOL=1e-12, CELLS=[], tune_task=lambda _: None)
    scope = {"STAGE": "stage0", "rec": rec, f"e{exp}": module, "json": json,
             "baselines": baselines, "parallel": Mock(run_tasks=lambda *a: []),
             "time": Mock(perf_counter=lambda: 0.0), "CONFIRMATORY_ENTROPY": 123}
    exec(cell(path.name, 3), scope)
    rec.write_stage0.assert_called_once()
    rec.record_diagnostics.assert_called_once()
    rec.finalize.assert_called_once()


def test_e22_notebook_records_failed_eb_comparison():
    rec = Mock()
    full = {"eb_check": {"agrees": False, "fit": {"status": "optimizer_failure",
                                                 "message": "Library failure."}}, "n_warnings": 0}
    scope = {"rec": rec, "baselines": baselines, "json": json,
             "e22": SimpleNamespace(full_sample=lambda _: full, summarise=lambda *a: {})}
    exec(cell("19_E22_lalonde.ipynb", 3), scope)
    summary, _ = scope["aggregate"](None, 123)
    rec.record_diagnostics.assert_called_once_with("eb_check", full["eb_check"])
    assert "full_sample_warnings" in summary


def test_e17_curvature_discrepancy_is_recorded():
    source = ast.parse(cell("15_E17_misspecification_bounds.ipynb", 2))
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                    and n.name == "curvature")
    rec = Mock()

    def g2(gen, X, alpha):
        return 1.0 + 4.0 * alpha * (1.0 - alpha)

    scope = {"np": np, "rec": rec, "g2": g2, "sign": lambda X: np.ones(len(X))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "curvature", "exec"), scope)
    # Both branches must be represented for the range computation.
    scope["sign"] = lambda X: X[:, 0]
    scope["curvature"](None, np.array([[1.0], [-1.0]]), np.zeros(2), np.ones(2))
    diagnostic = rec.record_diagnostics.call_args.args[1]
    assert all(not d["ok"] and d["maximum"] == 2.0 for d in diagnostic)


def test_e13_population_general_link_uses_hessian_method(monkeypatch):
    model = Mock()
    monkeypatch.setattr(e13, "model", lambda _: model)
    e13._general_fit("unused", np.ones((2, 1)), np.full(2, 0.5))
    assert model.fit.call_args.kwargs["method"] == "trust-ncg"


def test_e20_failed_population_is_serializable_without_predictions(monkeypatch):
    failed = population.PopulationSolution(np.full(3, np.nan), "nonfinite", 1, np.nan,
                                           np.nan, np.full(2, np.nan), np.nan,
                                           "Library failure.")
    monkeypatch.setattr(e20, "population_ratio", lambda _: failed)
    monkeypatch.setattr(e20, "population_moments", lambda _: {
        "theta0": 1.0, "sigma2_S": 1.0, "sigma2_T": 1.0})
    result = e20.stage0()
    assert result["status_64"] == "nonfinite" and result["message_64"] == failed.message
    assert result["beta_64"] == [None] * 3 and result["predictions"] is None
    assert result["support_margin_kind"] == "unavailable"
    json.dumps(result, allow_nan=False)


def test_e19_failed_population_is_serializable(monkeypatch):
    monkeypatch.setattr(e19, "population_q", lambda _: {
        "status": "nonfinite", "message": "Library failure.", "method": "trust-ncg",
        "beta_sq": np.full(5, np.nan), "max_gradient": np.nan, "min_hessian_eig": 1.0,
        "theta0": 1.0, "V_star": 1.0, "E_alpha0_sq": 1.0,
        "riesz_identity_gap": 0.0, "f_D_at_0": 0.1})
    monkeypatch.setattr(e19, "population_l", lambda _: {
        "theta0": 1.0, "V_star": 1.0, "E_alpha0_sq": 1.0, "mass": 1.0})
    monkeypatch.setattr(e19, "cex_i_variance", lambda *a: {"variance": 1.0, "mean": 0.0})
    result = e19.stage0()
    row = result["inference"]
    assert not row["certified"] and row["status_64"] == "nonfinite"
    assert row["message_64"] == "Library failure." and row["beta_sq"] == [None] * 5
    assert row["max_gradient_64"] is None
    json.dumps(result, allow_nan=False)


def test_e17_failed_population_is_recorded_without_curvature():
    failed = population.PopulationSolution(np.full(1, np.nan), "nonfinite", 1, np.nan,
                                           np.nan, np.full(2, np.nan), np.nan,
                                           "Library failure.")
    X = np.ones((2, 1))
    curvature = Mock(side_effect=AssertionError("Failed fits must not compute curvature."))
    scope = {"np": np, "pd": pd, "warnings": warnings, "collections": collections,
             "itertools": itertools, "SPURIOUS_MATMUL": "unused", "rec": Mock(),
             "DESIGNS": [(0.0, 0.0, 0.0, 0.0)], "SPANS": {"lin": 1},
             "GENERATORS": {"failed": lambda: SimpleNamespace(grad=lambda X, a: np.zeros(len(X)))},
             "population_of": lambda *a: (X, np.full(2, 0.5), np.ones(2), np.ones(2), 1.0),
             "design_matrices": lambda *a: (X, X, X, X), "sign": lambda X: np.ones(len(X)),
             "population": SimpleNamespace(solve=lambda **kw: failed), "curvature": curvature,
             "correctly_specified": lambda *a: False}
    exec(cell("15_E17_misspecification_bounds.ipynb", 3), scope)
    row = scope["cells_df"].iloc[0]
    assert row["status"] == "nonfinite" and row["message"] == failed.message
    assert not row["certified"] and np.isnan(row["norm_e"])
    curvature.assert_not_called()
