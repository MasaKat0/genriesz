"""Matched E-24 procedures, inner training rows and solver defaults."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from grrexp import baselines, e24


def test_design_and_polynomial_dictionary():
    assert e24.N_VALUES == (1000, 4000) and e24.R == 500 and e24.K == 5 and e24.CV_FOLDS == 3
    assert e24.LAMBDA_GRID == (1e-3, 1e-2, 1e-1) and e24.C_GRID == (0.1, 1.0, 10.0)
    assert len(e24.ARMS) == 8 and "RieszNet" in e24.ARMS
    assert not any(a in e24.ARMS for a in ("AutoDML", "EB", "TMLE", "Nystrom"))
    X = np.random.default_rng(0).normal(size=(100, 7))
    X[:, 0] = np.arange(100) % 2
    assert e24.cns_dictionary(X).shape == (100, 56)
    assert e24.interaction(e24.poly_base, X).shape == (100, 56)
    assert not hasattr(e24, "fit_autodml") and not hasattr(e24, "Nystrom")


def test_inner_cv_uses_only_outer_training_rows(monkeypatch):
    X = np.column_stack([np.arange(30) % 2, np.arange(30), np.zeros((30, 5))])
    cv = np.arange(30) % 3
    seen = []

    def fit(arm, base, lam, rows):
        training = set(rows[:, 1])
        seen.append((lam, training))

        def predict(test, *rest):
            assert not training.intersection(test[:, 1]) or len(training) == 30
            n = len(test)
            return np.ones(n), np.ones(n), -np.ones(n), "ok"

        return e24.Rep("ok", predict)

    monkeypatch.setattr(e24, "fit_glm", fit)
    result = e24.tune_representer("SQ", X, cv)
    assert result.status == "ok" and len(seen) == 10
    for j, (_, rows) in enumerate(seen[:-1]):
        assert rows == set(X[cv != j % 3, 1])
    assert seen[-1][1] == set(X[:, 1])
    assert len(result.info["cv"]) == 3


def test_tuning_failure_has_no_substitute(monkeypatch):
    X = np.column_stack([np.arange(30) % 2, np.zeros((30, 6))])
    monkeypatch.setattr(e24, "fit_glm", lambda *args: e24.Rep("optimizer_failure"))
    result = e24.tune_representer("SQ", X, np.arange(30) % 3)
    assert result.status == "cv_failed" and len(result.info["cv"]) == 3
    assert all(r["status"] == "optimizer_failure" for r in result.info["cv"])


def test_warning_without_library_failure_does_not_fail_tuning():
    import warnings

    def evaluate(v):
        warnings.warn("diagnostic", UserWarning, stacklevel=2)
        return v, "ok"

    choice, record = e24.select((0.1, 1, 10), evaluate)
    assert choice == 0.1 and record[0]["warnings"]


def test_logit_uses_major_library_defaults(monkeypatch):
    from sklearn.linear_model import LogisticRegression

    original = LogisticRegression.fit
    seen = []

    def fit(model, X, y):
        seen.append(model.get_params())
        return original(model, X, y)

    monkeypatch.setattr(LogisticRegression, "fit", fit)
    D, Z, Y = e24.draw(np.random.default_rng(4), 500)
    rep = e24.fit_logit(1.0, np.column_stack([D, Z]))
    assert rep.status == "ok"
    assert seen[0]["max_iter"] == LogisticRegression().max_iter
    assert seen[0]["tol"] == LogisticRegression().tol


def test_baseline_propensity_does_not_reject_small_finite_values():
    D = np.array([0.0, 1.0])
    Y = np.array([1.0, 2.0])
    e = np.array([1e-14, 1 - 1e-14])
    theta, se = baselines._aipw(D, Y, e, np.ones(2), np.zeros(2))
    assert np.isfinite([theta, se]).all()


def test_outcome_cv_fits_only_its_inner_training_rows(monkeypatch):
    X = np.column_stack([np.arange(30) % 2, np.arange(30), np.zeros((30, 5))])
    cv = np.arange(30) % 3
    seen = []

    class Outcome:
        def __init__(self, base, lam, rows, Y):
            self.training = set(rows[:, 1])
            seen.append(self.training)

        def __call__(self, test):
            assert not self.training.intersection(test[:, 1]) or len(self.training) == 30
            return np.zeros(len(test))

    monkeypatch.setattr(e24, "Ridge", Outcome)
    model, info = e24.tune_outcome(X, np.ones(30), cv)
    assert len(seen) == 10
    assert all(rows == set(X[cv != j % 3, 1]) for j, rows in enumerate(seen[:-1]))
    assert seen[-1] == set(X[:, 1]) and info["choice"] == max(e24.LAMBDA_GRID)
