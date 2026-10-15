"""Library solver defaults, equivalent objectives, and failure propagation."""

import numpy as np
import pytest
from scipy import optimize

from genriesz import BPGenerator, SquaredGenerator
from genriesz.solvers import DualProblem, fista_solve, minimize_coefficients, newton_solve


def test_smooth_default_is_the_direct_library_result():
    X = np.eye(3)
    problem = DualProblem(SquaredGenerator(), X, X, np.zeros(3), np.array([1., -2., 3.]))
    result = newton_solve(problem, beta0=np.zeros(3), lam=0.2, q=2.)
    reference = optimize.minimize(
        lambda b: problem.evaluate(b).loss + 0.1 * (b @ b), np.zeros(3),
        jac=lambda b: problem.evaluate(b).grad + 0.2 * b,
    )
    assert result.success == reference.success
    assert result.message == str(reference.message)
    np.testing.assert_array_equal(result.beta, reference.x)
    np.testing.assert_allclose(result.beta, problem.target / (1/6 + 0.2), atol=1e-4)


@pytest.mark.parametrize("q", [1.5, 3.])
def test_lq_objective_preserves_the_known_stationary_solution(q):
    X = np.eye(2)
    expected = np.array([0.7, -1.2])
    lam = 0.3
    target = expected / 4 + lam * np.sign(expected) * np.abs(expected)**(q-1)
    problem = DualProblem(SquaredGenerator(), X, X, np.zeros(2), target)
    solve = fista_solve if q < 2 else newton_solve
    result = solve(problem, beta0=np.zeros(2), lam=lam, q=q)
    assert result.success, result.message
    np.testing.assert_allclose(result.beta, expected, atol=1e-4)


def test_weighted_l1_preserves_the_closed_form_solution():
    target = np.array([3., -2., 0.1])
    weights = np.array([0.5, 0.25, 0.2])
    res = minimize_coefficients(lambda b: 0.5 * np.sum((b-target)**2),
                                lambda b: b-target, np.zeros(3), l1=weights)
    assert res.success, res.message
    expected = np.sign(target) * np.maximum(np.abs(target)-weights, 0.)
    np.testing.assert_allclose(res.x, expected, atol=1e-5)


def test_scipy_defaults_and_explicit_options_are_forwarded(monkeypatch):
    captured = []
    actual = optimize.minimize

    def recording(*args, **kwargs):
        captured.append(kwargs.copy())
        return actual(*args, **kwargs)

    monkeypatch.setattr(optimize, "minimize", recording)
    minimize_coefficients(lambda b: float(b@b), lambda b: 2*b, np.ones(2))
    assert captured[-1]["tol"] is None and captured[-1]["options"] == {}
    assert "method" not in captured[-1]
    res = minimize_coefficients(lambda b: float(b@b), lambda b: 2*b, np.ones(2),
                                max_iter=0, tol=0.01, options={"gtol": 0.02})
    assert captured[-1]["tol"] == 0.01
    assert captured[-1]["options"] == {"maxiter": 0, "gtol": 0.02}
    assert not res.success


def test_domain_is_infinite_and_library_failure_is_preserved(monkeypatch):
    X = np.array([[1.], [-1.]])
    problem = DualProblem(BPGenerator(C=0., omega=1., branch_fn=lambda x: 1),
                          X, X, np.zeros(2), np.array([1.5]))
    actual = optimize.minimize
    observed = []

    def recording(fun, x0, **kwargs):
        assert np.isposinf(fun(np.array([3.])))
        result = actual(fun, x0, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(optimize, "minimize", recording)
    result = newton_solve(problem, beta0=np.zeros(1))
    assert not result.success and not observed[0].success
    assert result.message == str(observed[0].message)
    assert result.extra["scipy_status"] == observed[0].status


def test_nonfinite_library_success_is_still_a_failure(monkeypatch):
    X = np.ones((1, 1))
    problem = DualProblem(SquaredGenerator(), X, X, np.zeros(1), np.ones(1))
    message = "library message retained"
    monkeypatch.setattr(optimize, "minimize", lambda *args, **kwargs: optimize.OptimizeResult(
        x=np.array([np.nan]), fun=np.nan, jac=np.array([np.nan]), success=True,
        status=0, message=message, nit=1,
    ))
    result = newton_solve(problem, beta0=np.zeros(1))
    assert not result.success and result.status == "nonfinite"
    assert result.message == message
