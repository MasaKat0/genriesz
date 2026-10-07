from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grrexp import population  # noqa: E402

import genriesz as gr  # noqa: E402


def ate_population(c0: float, c1: float):
    z = np.linspace(-1.0, 1.0, 9)
    e = expit(c0 + c1 * z)
    X = np.array([(d, zz) for zz in z for d in (1.0, 0.0)])
    w = np.array([p for ee in e for p in (ee / z.size, (1 - ee) / z.size)])
    alpha0 = np.where(X[:, 0] == 1, 1 / np.repeat(e, 2), -1 / (1 - np.repeat(e, 2)))
    basis = gr.TreatmentInteractionBasis(
        base_basis=gr.CallableBasis(
            lambda Z: np.column_stack([np.ones(len(np.atleast_2d(Z))), np.atleast_2d(Z)[:, 0]])
        )
    )
    basis.fit(X)
    Phi = np.asarray(basis(X))
    M = np.asarray(gr.ATEFunctional(0).m_basis_matrix(X, basis))
    return X, w, Phi, M, alpha0


def test_sq_population_solution_is_the_weighted_projection() -> None:
    X, w, Phi, M, alpha0 = ate_population(0.3, 0.8)
    sol = population.solve(
        generator=gr.SquaredGenerator(C=0.0), X=X, w=w, Phi=Phi, M=M, offset=np.zeros(len(X))
    )
    assert sol.status == "ok" and sol.certified(margin_tol=-np.inf)
    np.testing.assert_allclose(
        sol.alpha, population.weighted_projection(alpha0, Phi, w), atol=1e-12
    )


def test_ukl_recovers_a_representer_in_the_model() -> None:
    X, w, Phi, M, alpha0 = ate_population(0.3, 0.8)
    sign = lambda A: np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)  # noqa: E731
    ukl = gr.UKLGenerator(C=1.0, branch_fn=sign)
    off = gr.offset_from_alpha(ukl, lambda A: np.where(A[:, 0] == 1, 2.0, -2.0))(X)
    sol = population.solve(generator=ukl, X=X, w=w, Phi=Phi, M=M, offset=off)
    assert sol.certified()
    np.testing.assert_allclose(sol.alpha, alpha0, rtol=1e-11, atol=0)


def test_start_outside_domain_and_bad_weights_stop() -> None:
    X, w, Phi, M, _ = ate_population(0.0, 0.5)
    sign = lambda A: np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)  # noqa: E731
    bkl = gr.BKLGenerator(C=1.0, branch_fn=sign)
    with pytest.raises(ValueError, match="outside the link domain"):
        population.solve(generator=bkl, X=X, w=w, Phi=Phi, M=M, offset=np.zeros(len(X)))
    with pytest.raises(ValueError, match="probabilities"):
        population.solve(
            generator=gr.SquaredGenerator(), X=X, w=2 * w, Phi=Phi, M=M, offset=np.zeros(len(X))
        )


def sign(A):
    return np.where(np.atleast_2d(A)[:, 0] == 1, 1, -1)


GENERATORS = {
    "UKL": lambda: gr.UKLGenerator(C=1.0, branch_fn=sign),
    "BKL": lambda: gr.BKLGenerator(C=1.0, branch_fn=sign),
    "BP": lambda: gr.BPGenerator(omega=0.5, C=0.0, branch_fn=sign),
}


@pytest.mark.parametrize("name", list(GENERATORS))
def test_objective_derivatives_match_finite_differences(name) -> None:
    gen = GENERATORS[name]()
    X, w, Phi, M, _ = ate_population(0.2, 0.6)
    off = np.asarray(gen.grad(X, np.where(X[:, 0] == 1, 2.0, -2.0)))
    b = w @ M

    def F(beta):
        g_star, alpha, dalpha = gen.dual_eval(X, off + Phi @ beta)
        return (
            float(w @ g_star) - beta @ b,
            Phi.T @ (w * alpha) - b,
            Phi.T @ (Phi * (w * dalpha)[:, None]),
        )

    beta = np.array([0.01, -0.02, 0.015, 0.005])
    _, grad, H = F(beta)
    h = 1e-6
    for j in range(4):
        e = np.zeros(4)
        e[j] = h
        np.testing.assert_allclose(
            (F(beta + e)[0] - F(beta - e)[0]) / (2 * h), grad[j], rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            (F(beta + e)[1] - F(beta - e)[1]) / (2 * h), H[:, j], rtol=0, atol=1e-7
        )
    sol = population.solve(generator=gen, X=X, w=w, Phi=Phi, M=M, offset=off)
    assert sol.status == "ok" and sol.max_gradient <= 1e-12


def test_certificate_uses_the_dual_margin() -> None:
    gen = gr.BKLGenerator(C=1.0, branch_fn=sign)
    X, w, Phi, M, _ = ate_population(0.0, 0.5)
    off = np.asarray(gen.grad(X, np.where(X[:, 0] == 1, 2.0, -2.0)))
    sol = population.solve(generator=gen, X=X, w=w, Phi=Phi, M=M, offset=off)
    v = off + Phi @ sol.beta
    assert sol.min_dual_margin == pytest.approx(float(np.min(gen.dual_margin(X, v))), rel=0, abs=0)
    assert sol.certified(margin_tol=sol.min_dual_margin)
    assert not sol.certified(margin_tol=sol.min_dual_margin * 1.0001)


def test_counterfactual_rows_must_stay_admissible() -> None:
    gen = gr.BKLGenerator(C=1.0, branch_fn=sign)
    X, w, Phi, M, _ = ate_population(0.0, 0.5)
    off = np.asarray(gen.grad(X, np.where(X[:, 0] == 1, 2.0, -2.0)))
    bad_X = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="outside the link domain"):
        population.solve(
            generator=gen,
            X=X,
            w=w,
            Phi=Phi,
            M=M,
            offset=off,
            check_X=bad_X,
            check_Phi=np.zeros((1, 4)),
            check_offset=np.array([5.0]),
        )


def test_singular_system_stops_without_an_exception() -> None:
    X, w, Phi, M, _ = ate_population(0.0, 0.5)
    Phi2 = np.column_stack([Phi, Phi[:, :1]])
    M2 = np.column_stack([M, M[:, :1]])
    sol = population.solve(
        generator=gr.UKLGenerator(C=1.0, branch_fn=sign),
        X=X,
        w=w,
        Phi=Phi2,
        M=M2,
        offset=np.asarray(
            gr.UKLGenerator(C=1.0, branch_fn=sign).grad(X, np.where(X[:, 0] == 1, 2.0, -2.0))
        ),
    )
    assert sol.status == "singular" and not sol.certified()


def test_counterfactual_overflow_of_a_finite_ukl_coordinate_is_rejected() -> None:
    ukl = gr.UKLGenerator(C=1.0, branch_fn=sign)
    X, w, Phi, M, _ = ate_population(0.0, 0.5)
    off = np.asarray(ukl.grad(X, np.where(X[:, 0] == 1, 2.0, -2.0)))
    with pytest.raises(ValueError, match="outside the link domain"):
        population.solve(
            generator=ukl,
            X=X,
            w=w,
            Phi=Phi,
            M=M,
            offset=off,
            check_X=np.array([[1.0, 0.0]]),
            check_Phi=np.zeros((1, 4)),
            check_offset=np.array([800.0]),
        )  # exp(800) overflows to inf


def test_backtracking_and_failed_acceptance() -> None:
    # starting next to the BKL boundary (|alpha_ref| = 1.01), the full Newton step leaves the
    # domain:
    # with halvings the solve converges, without them it stops with "linesearch"
    bkl = gr.BKLGenerator(C=1.0, branch_fn=sign)
    X, w, Phi, M, _ = ate_population(0.8, 1.5)
    off = np.asarray(bkl.grad(X, np.where(X[:, 0] == 1, 1.01, -1.01)))
    sol = population.solve(generator=bkl, X=X, w=w, Phi=Phi, M=M, offset=off)
    assert sol.status == "ok" and sol.certified()
    stuck = population.solve(generator=bkl, X=X, w=w, Phi=Phi, M=M, offset=off, max_halvings=0)
    assert stuck.status == "linesearch" and stuck.n_iter == 1 and not stuck.certified()
