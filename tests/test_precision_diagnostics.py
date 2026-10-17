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

from grrexp import baselines, e12, e19, outputs, population  # noqa: E402

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






def test_recorder_retains_failed_checks_as_json():
    rec = outputs.RunRecorder.__new__(outputs.RunRecorder)
    rec.closed, rec.diagnostics = False, {}
    rec.record_diagnostics("finite", {"value": np.inf, "ok": False})
    assert rec.diagnostics == {"finite": [{"value": None, "ok": False}]}
    json.dumps(rec.diagnostics, allow_nan=False)







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


