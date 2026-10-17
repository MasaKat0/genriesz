"""Matched exponential odds, moment regimes and feasible ATT correction."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks/experiments"))
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
