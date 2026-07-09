from __future__ import annotations

import pytest

from genriesz.results import FunctionalEstimate, SingleEstimate


def _single(name: str, value: float) -> SingleEstimate:
    return SingleEstimate(
        name=name.upper(),
        estimate=value,
        se=0.1,
        ci_low=value - 0.2,
        ci_high=value + 0.2,
        p_value=0.0,
    )


def test_functional_estimate_named_accessors() -> None:
    estimates = {
        "ra": _single("ra", 1.0),
        "rw": _single("rw", 2.0),
        "arw": _single("arw", 3.0),
        "tmle": _single("tmle", 4.0),
    }
    res = FunctionalEstimate(
        estimand="ATE",
        n=10,
        alpha=0.05,
        null=0.0,
        estimates=estimates,
        diagnostics={"alpha_abs_max": 5.0},
    )

    assert res["ra"] is estimates["ra"]
    assert res.rw is estimates["rw"]
    assert res.arw.estimate == 3.0
    assert res.tmle.estimate == 4.0
    assert res.diagnostics["alpha_abs_max"] == 5.0

    with pytest.raises(KeyError):
        _ = res["missing"]


def test_functional_estimate_prefers_shared_variant() -> None:
    estimates = {
        "arw (shared)": _single("arw", 3.0),
        "arw (separate)": _single("arw", 4.0),
    }
    res = FunctionalEstimate(
        estimand="ATE",
        n=10,
        alpha=0.05,
        null=0.0,
        estimates=estimates,
        diagnostics={},
    )

    assert res["arw"].estimate == 3.0
    assert res.arw.estimate == 3.0
