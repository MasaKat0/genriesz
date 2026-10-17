"""The derivative functional and its fitted procedure under DGP19Q."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks/experiments"))
from grrexp import e19


def test_design_and_analytic_derivatives():
    assert e19.CELLS == [[1000], [4000]] and e19.R == 500 and e19.K == 5
    X = np.random.default_rng(0).normal(size=(30, 3))
    step = np.zeros_like(X)
    step[:, 0] = 1e-5
    for f, df in ((e19.feat_gamma, e19.dfeat_gamma), (e19.feat_alpha, e19.dfeat_alpha)):
        np.testing.assert_allclose((f(X + step) - f(X - step)) / 2e-5, df(X), atol=1e-8)
    np.testing.assert_allclose(
        e19.feat_alpha(X) @ np.array([0, 1, 0.4, -0.5, -0.3]), e19.alpha0_q(X)
    )


def test_target_integration_and_procedure():
    # Independent higher-order quadrature of the target, with no population nuisance fit.
    nodes, weights = e19._gl(-10, 10, 220)
    X, w = e19.quadrature_q(32, (nodes, weights))
    assert e19.true_target() == pytest.approx(w @ e19.dgamma0(X), abs=1e-10)
    result = e19.replicate((0, 0, 20261007))
    rec = result["arms"]["SQ|ARW_cf"]
    assert rec["status"] == "ok" and abs(rec["estimate"] - e19.true_target()) < 0.2
    assert len(rec["fits"]) == 10
