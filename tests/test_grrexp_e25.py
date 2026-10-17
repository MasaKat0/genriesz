"""Matched exponential odds, moment regimes and feasible ATT correction."""

import json
import sys
from functools import wraps
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import digamma

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from grrexp import e25, study


def test_design_and_moment_frontier():
    assert len(e25.CELLS) == 6 and e25.N_VALUES == (500, 2000, 8000) and e25.R == 500 and e25.K == 2
    assert e25.moment_facts(0.2)["SQ_score_variance_finite"]
    weak = e25.moment_facts(0.6)
    assert not weak["SQ_score_variance_finite"]
    assert weak["compatible_score_variance_finite"] and weak["representer_L2_finite"]
    assert weak["SQ_stable_index"] == pytest.approx(4 / 3)


@pytest.mark.parametrize("arm", e25.ARM_LABELS)
def test_objective_gradients_and_default_optimizer(monkeypatch, arm):
    U, D, _ = e25.draw(np.random.default_rng(12), 500, 0.6)
    F = np.column_stack([np.ones(len(U)), -np.log(U)])
    c = np.array([0.1, 0.55])
    numerical = approx_derivative(lambda c: e25.objective(c, F, D, arm)[0], c).ravel()
    np.testing.assert_allclose(e25.objective(c, F, D, arm)[1], numerical, rtol=1e-5)
    actual = e25.optimize.minimize

    def track(*args, **kwargs):
        assert "tol" not in kwargs and kwargs["options"] is None and "method" not in kwargs
        return actual(*args, **kwargs)

    monkeypatch.setattr(e25.optimize, "minimize", track)
    coef, info = e25.fit(U, D, 0.6, arm)
    assert "message" in info and "stationarity_scaled" in info
    if arm != "Logistic":
        assert all(lo <= v <= hi for v, (lo, hi) in zip(coef, e25.BOXES[0.6], strict=True))


def test_l2_integral_and_empirical_share_correction():
    assert e25.representer_l2([0, 0.6], 0.6) == 0
    assert e25.representer_l2([0, 0.81], 0.6) == np.inf
    D = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    Y = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    odds = np.arange(1, 6.0)
    g0 = np.zeros(5)
    g1 = np.ones(5)
    g = D.copy()
    rec = e25.att_record(D, Y, odds, g, g1, g0)
    uncentered = (D * (g1 - g0) + (D - (1 - D) * odds) * (Y - g)) / D.mean()
    theta = uncentered.mean()
    assert rec["estimate"] == pytest.approx(theta)
    assert rec["se"] == pytest.approx((uncentered - D * theta / D.mean()).std() / np.sqrt(5))


@pytest.mark.parametrize(
    "a, b, b0",
    [
        (0.1, 0.35, 0.2),
        (-0.3, 0.4, 0.6),
        (-0.20776702766243574, 0.75, 0.6),
        (0.0, 0.79999, 0.6),
    ],
)
def test_l2_matches_integrated_power_moments(a, b, b0):
    def moment(c):
        return (digamma((c + b0) / (2 * b0)) - digamma(c / (2 * b0))) / (2 * b0)

    squared = (
        np.exp(2 * a) * moment(1 + b0 - 2 * b)
        - 2 * np.exp(a) * moment(1 - b)
        + moment(1 - b0)
    )
    expected = np.sqrt(squared) / e25.treated_share(b0)
    assert e25.representer_l2([a, b], b0) == pytest.approx(expected, rel=1e-10)


def test_l2_remains_accurate_near_truth_and_at_integrability_boundary():
    b0, a = 0.6, 1e-10
    c = 1 - b0
    moment = (digamma((c + b0) / (2 * b0)) - digamma(c / (2 * b0))) / (2 * b0)
    expected = np.expm1(a) * np.sqrt(moment) / e25.treated_share(b0)
    assert e25.representer_l2([a, b0], b0) == pytest.approx(expected, rel=1e-10, abs=0)
    assert e25.representer_l2([0.0, 0.8], b0) == np.inf
    assert np.isfinite(e25.representer_l2([0.0, 0.799], b0))


def test_l2_quadrature_failure_is_reported(monkeypatch):
    monkeypatch.setattr(e25, "treated_share", lambda b0: 0.6)

    @wraps(e25.integrate.quad)
    def no_convergence(*args, **kwargs):
        return 1.0, 1.0, {}, "no convergence"

    monkeypatch.setattr(e25.integrate, "quad", no_convergence)
    with pytest.raises(RuntimeError, match="no convergence"):
        e25.representer_l2([0.0, 0.75], 0.6)


def test_l2_uses_default_quadrature_options(monkeypatch):
    quad = e25.integrate.quad
    calls = []

    @wraps(quad)
    def track(*args, **kwargs):
        calls.append(kwargs)
        return quad(*args, **kwargs)

    monkeypatch.setattr(e25.integrate, "quad", track)
    assert np.isfinite(e25.representer_l2([0.1, 0.75], 0.6))
    assert calls and all(set(kwargs) <= {"full_output"} for kwargs in calls)


@pytest.mark.parametrize("cell, scale", [(0, np.sqrt(500)), (3, 500**0.25 / np.log(500))])
def test_sq_reference_normalization(cell, scale):
    import pandas as pd

    errors = np.array([-0.1, 0.03, 0.15])
    raw = pd.DataFrame({
        "cell": cell,
        "n": e25.CELLS[cell][-1],
        "arm": "SQ",
        "estimate": 1.0,
        "se": 0.1,
        "status": "ok",
        "slope_error": errors,
        "full_fit": json.dumps({"status": "ok", "boundary": False, "stationarity_scaled": 0.0}),
    })
    row = e25.summarise(raw).iloc[0]
    assert row.SQ_reference_median_abs_slope_error == pytest.approx(
        np.median(np.abs(errors)) * scale
    )
    for q in (5, 50, 95):
        assert row[f"SQ_reference_scaled_error_q{q:02}"] == pytest.approx(
            np.quantile(errors * scale, q / 100)
        )


def test_replication_and_coefficient_quantiles():
    tasks = [(0, r, 20261007) for r in range(2)]
    raw = study.raw_frame(e25, tasks, [e25.replicate(t) for t in tasks])
    summary = e25.summarise(raw)
    assert len(raw) == 6 and set(raw.arm) == set(e25.ARM_LABELS)
    assert {
        "slope_error_q05",
        "slope_error_q95",
        "root_n_median_abs_slope_error",
        "optimizer_status_counts",
        "boundary_frequency_conditional",
    } <= set(summary)
    assert (summary.R == 2).all()


def test_figures_accept_a_missing_regime_or_no_returned_coefficients(tmp_path):
    import matplotlib.pyplot as plt
    import pandas as pd

    base = {
        "b0": 0.6,
        "n": 500,
        "arm": "SQ",
        "median_abs_slope_error": float("nan"),
        "root_n_slope_error_q50": float("nan"),
        "root_n_slope_error_q05": float("nan"),
        "root_n_slope_error_q95": float("nan"),
    }
    fig = e25.figure(pd.DataFrame([base]))
    fig.savefig(tmp_path / "missing.pdf")
    plt.close(fig)
    assert (tmp_path / "missing.pdf").stat().st_size > 0
