"""Numerical discrepancies are recorded; library failures never supply replacement fits."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))

from grrexp import outputs, population  # noqa: E402

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
    result = population.solve(
        generator=gr.SquaredGenerator(),
        X=Phi,
        w=np.full(2, 0.5),
        Phi=Phi,
        M=np.eye(2),
        offset=np.zeros(2),
    )
    assert result.status == "ok"
    assert calls[0]["method"] == "trust-ncg"
    assert calls[0]["tol"] is None and calls[0]["options"] == {}
    np.testing.assert_allclose(result.beta, [2.0, 0.5])


def test_nonfinite_diagnostics_are_serializable():
    result = outputs.clean({"finite": {"value": np.inf, "ok": False}})
    assert result == {"finite": {"value": None, "ok": False}}
    json.dumps(result, allow_nan=False)
