import numpy as np
import pytest

from genriesz import (
    ATEFunctional,
    BPGenerator,
    SquaredGenerator,
    grr_functional,
)
from genriesz.basis import coerce_basis
from genriesz.model_selection import GRRCVConfig, select_grr_hyperparams


def _make_synthetic_ate(n: int = 240, d: int = 2, seed: int = 3):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    logits = 0.7 * Z[:, 0] - 0.3 * Z[:, 1]
    e = 1.0 / (1.0 + np.exp(-logits))
    D = rng.binomial(1, e, size=n)
    mu0 = Z[:, 0] + 0.25 * Z[:, 1] ** 2
    Y = mu0 + 1.0 * D + rng.normal(scale=1.0, size=n)
    X = np.concatenate([D.reshape(-1, 1), Z], axis=1)
    return X, Y


def phi(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        d = X[0]
        z = X[1:]
        return np.concatenate([[1.0], [d], z, d * z])
    d = X[:, [0]]
    z = X[:, 1:]
    return np.concatenate([np.ones((len(X), 1)), d, z, d * z], axis=1)


def _branch_treated(x: np.ndarray) -> int:
    return int(float(x[0]) >= 0.5)


def _bp(omega: float) -> object:
    return BPGenerator(C=1.0, omega=omega, branch_fn=_branch_treated).as_generator()


def test_config_rejects_generator_grid_without_squared_loss_score():
    with pytest.raises(ValueError, match="squared_loss_validation"):
        GRRCVConfig(generator_grid=[_bp(0.5)])


def test_config_rejects_empty_or_non_generator_grid():
    with pytest.raises(ValueError, match="at least one"):
        GRRCVConfig(generator_grid=[], selection_score="squared_loss_validation")
    with pytest.raises(TypeError, match="BregmanGenerator"):
        GRRCVConfig(generator_grid=["bp"], selection_score="squared_loss_validation")


def test_select_grr_hyperparams_picks_a_named_generator():
    X, Y = _make_synthetic_ate()
    grid = [_bp(0.25), _bp(0.75)]
    config = GRRCVConfig(
        generator_grid=grid,
        selection_score="squared_loss_validation",
        return_path=True,
        random_state=0,
    )
    sel = select_grr_hyperparams(
        X_train=X,
        y_train=Y,
        m=ATEFunctional(treatment_index=0),
        basis=coerce_basis(phi),
        generator=grid[0],
        config=config,
        riesz_lam=1e-2,
    )
    assert sel.generator_name in {g.name for g in grid}
    assert sel.generator is grid[0] or sel.generator is grid[1]
    assert np.isfinite(sel.best_score)
    assert len(sel.path) == len(grid)
    assert all("generator_name" in row for row in sel.path)


def test_grr_functional_generator_grid_end_to_end_and_deterministic():
    X, Y = _make_synthetic_ate()
    kwargs = dict(
        X=X,
        Y=Y,
        m=ATEFunctional(treatment_index=0),
        basis=phi,
        cross_fit=True,
        folds=3,
        random_state=0,
        riesz_lam=1e-2,
        outcome_models="none",
        estimators=("rw",),
        riesz_selection_score="squared_loss_validation",
    )
    res1 = grr_functional(
        generator=_bp(0.5), riesz_generator_grid=[_bp(0.25), _bp(0.5), _bp(0.75)], **kwargs
    )
    res2 = grr_functional(
        generator=_bp(0.5), riesz_generator_grid=[_bp(0.25), _bp(0.5), _bp(0.75)], **kwargs
    )

    selected = res1.diagnostics["riesz_cv"]["selected"]
    assert len(selected) == 3
    assert all(s["generator"] is not None for s in selected)
    assert np.isfinite(res1["rw"].estimate)
    # The plain BP family does not modify the estimand, and the per-fold
    # generator selection must not flip the flag.
    assert res1.diagnostics["riesz_modifies_estimand"] is False
    # Same data, folds, and grid: the selection and the estimate are deterministic.
    assert res1["rw"].estimate == res2["rw"].estimate
    assert [s["generator"] for s in selected] == [
        s["generator"] for s in res2.diagnostics["riesz_cv"]["selected"]
    ]


def test_single_candidate_grid_matches_fixed_generator_path():
    X, Y = _make_synthetic_ate()
    gen = SquaredGenerator(C=0.0).as_generator()
    kwargs = dict(
        X=X,
        Y=Y,
        m=ATEFunctional(treatment_index=0),
        basis=phi,
        cross_fit=True,
        folds=3,
        random_state=0,
        riesz_lam=1e-2,
        outcome_models="none",
        estimators=("rw",),
    )
    res_fixed = grr_functional(generator=gen, **kwargs)
    res_grid = grr_functional(
        generator=gen,
        riesz_generator_grid=[gen],
        riesz_selection_score="squared_loss_validation",
        **kwargs,
    )
    # With one candidate and no other grid dimension, the outer refits use the
    # same basis, lambda, and generator, so the estimates must coincide exactly.
    assert res_grid["rw"].estimate == res_fixed["rw"].estimate
    assert res_grid["rw"].se == res_fixed["rw"].se
