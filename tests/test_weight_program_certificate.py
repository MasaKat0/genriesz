"""Sample weight program (registration §1.3 A-6, E-9; re-review item 5).

The numerical check never reports nonexistence: a near-boundary solution with a
small duality gap is "uncertified_boundary". On PR-4a's BP example (where the
GRR estimator provably does not exist) it returns the boundary solution
(3, 0) with dual coefficient 4 and a zero gap.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy.special import expit

from genriesz import (
    GRRGLM,
    ATEFunctional,
    BPGenerator,
    CallableBasis,
    UKLGenerator,
    offset_from_alpha,
    weight_program_certificate,
)
from genriesz.certificates import VERDICTS
from genriesz.functionals import LinearFunctional

HAS_CVXPY = importlib.util.find_spec("cvxpy") is not None
BACKENDS = ["scipy"] + (["cvxpy"] if HAS_CVXPY else [])


class _FixedTarget(LinearFunctional):
    """A functional with mean m(W, phi) = b (each row contributes b)."""

    def __init__(self, b):
        super().__init__(name="fixed")
        object.__setattr__(self, "b", np.asarray(b, dtype=float))

    def m_basis_matrix(self, X, basis):
        return np.tile(self.b, (len(np.atleast_2d(X)), 1))

    def m_from_predictor(self, X, predict):  # pragma: no cover - unused
        raise NotImplementedError


def _pos(_x):
    return 1


def _treated(x):
    return int(x[0] == 1.0)


def test_verdicts_never_claim_nonexistence():
    assert set(VERDICTS) == {
        "interior", "uncertified_boundary", "uncertified_infeasible", "inconclusive"
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_pr4a_bp_example_is_an_uncertified_boundary_solution(backend):
    X = np.array([[1.0], [-1.0]])
    cert = weight_program_certificate(
        X=X,
        basis=CallableBasis(lambda X_: np.atleast_2d(X_)[:, :1]),
        functional=_FixedTarget([1.5]),
        generator=BPGenerator(C=0.0, omega=1.0, branch_fn=_pos),
        backend=backend,
    )
    assert cert.verdict == "uncertified_boundary"
    assert cert.backend == backend
    assert np.allclose(cert.alpha, [3.0, 0.0], atol=1e-6)
    assert cert.beta is not None and abs(cert.beta[0] - 4.0) < 1e-5
    assert abs(cert.gap) <= 1e-9
    assert cert.primal_value == pytest.approx(1.5, abs=1e-8)
    assert cert.dual_value == pytest.approx(1.5, abs=1e-8)


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
def test_interior_solution_matches_the_grr_weights():
    rng = np.random.default_rng(0)
    n = 400
    a = np.sqrt(3.0)
    Z = rng.uniform(-a, a, size=(n, 2))
    D = rng.binomial(1, expit(0.5 * Z[:, 0])).astype(float)
    X = np.column_stack([D, Z])

    def cols(X_):
        X_ = np.atleast_2d(X_)
        d = X_[:, [0]]
        F = np.column_stack([np.ones(len(X_)), X_[:, 1:]])
        return np.concatenate([d * F, (1.0 - d) * F], axis=1)

    gen = UKLGenerator(C=1.0, branch_fn=_treated)
    off = offset_from_alpha(gen, lambda X_: np.where(X_[:, 0] == 1.0, 3.0, -3.0))
    basis = CallableBasis(cols)
    model = GRRGLM(basis=basis, generator=gen, functional=ATEFunctional(0), penalty=None,
                   offset=off)
    assert model.fit(X).status == "ok"
    cert = weight_program_certificate(
        X=X, basis=basis, functional=ATEFunctional(0), generator=gen, offset=off,
        backend="cvxpy",
    )
    assert cert.verdict == "interior"
    # Conic-solver accuracy (exponential cone) is ~1e-6 relative; the duality gap
    # is second order in that error and stays below 1e-9.
    assert np.max(np.abs(cert.alpha - model.predict_alpha(X))) < 1e-3
    assert np.max(np.abs(cert.beta - model.beta_)) < 1e-3
    assert abs(cert.gap) <= 1e-9


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
def test_balance_set_outside_the_closed_domain_is_uncertified_infeasible():
    """PR-4a: UKL, n = p = 1, phi = 1, b < C, lambda = 0 -- balance needs alpha = b < C."""

    cert = weight_program_certificate(
        X=np.ones((1, 1)),
        basis=CallableBasis(lambda X_: np.ones((len(np.atleast_2d(X_)), 1))),
        functional=_FixedTarget([0.5]),
        generator=UKLGenerator(C=1.0, branch_fn=_pos),
        backend="cvxpy",
    )
    assert cert.verdict == "uncertified_infeasible"
    assert cert.alpha is None


def test_branchwise_generator_needs_branch_fn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen = UKLGenerator(C=1.0)
    with pytest.raises(ValueError, match="branch_fn"):
        weight_program_certificate(
            X=np.ones((2, 1)),
            basis=CallableBasis(lambda X_: np.ones((len(np.atleast_2d(X_)), 1))),
            functional=_FixedTarget([2.0]),
            generator=gen,
        )


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
@pytest.mark.parametrize(
    "C, grr_status, verdict",
    [(1.0, "uncertified_numerical_boundary", "uncertified_boundary"), (0.0, "ok", "interior")],
)
def test_solver_status_and_weight_program_agree_on_the_probe_design(C, grr_status, verdict):
    """BP(C=1) under weak overlap: the GRR solver stops at the boundary and the
    weight program's numerical solution has weights on the boundary. Neither is
    read as nonexistence."""

    import warnings

    rng = np.random.default_rng(1)
    n = 1000
    a = np.sqrt(3.0)
    Z = rng.uniform(-a, a, size=(n, 3))
    q = Z[:, 1] ** 2 - 1.0
    D = rng.binomial(1, expit(1.25 * (Z[:, 0] - 0.5 * Z[:, 1] + 0.6 * q))).astype(float)
    X = np.column_stack([D, Z])

    def cols(X_):
        X_ = np.atleast_2d(X_)
        d = X_[:, [0]]
        z = X_[:, 1:]
        F = np.concatenate([np.ones((len(X_), 1)), z, z[:, [1]] ** 2 - 1.0], axis=1)
        return np.concatenate([d * F, (1.0 - d) * F], axis=1)

    gen = BPGenerator(C=C, omega=0.5, branch_fn=_treated)
    basis = CallableBasis(cols)
    r = GRRGLM(basis=basis, generator=gen, functional=ATEFunctional(0), penalty=None).fit(X)
    assert r.status == grr_status
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # cvxpy "may be inaccurate"
        cert = weight_program_certificate(
            X=X, basis=basis, functional=ATEFunctional(0), generator=gen, backend="cvxpy"
        )
    assert cert.verdict == verdict
    assert abs(cert.gap) <= 1e-9
