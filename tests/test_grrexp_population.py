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
    np.testing.assert_allclose(sol.alpha, alpha0, rtol=1e-11)


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
