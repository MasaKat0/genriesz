from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "experiments"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from reference_selection import (  # noqa: E402
    CandidateSpec,
    ExperimentBasis,
    bias_aware_critical_value,
    fit_candidate,
    gaussian_multiplier_mean_radii,
    generate_data,
    make_fold_roles,
)


def test_low_dimensional_dgp_has_unit_ate_and_finite_values() -> None:
    data = generate_data(n=50_000, design="low", overlap_scale=0.5, seed=17)
    assert abs(float(np.mean(data.tau)) - 1.0) < 0.02
    assert np.all(np.isfinite(data.X))
    assert np.all(np.isfinite(data.y))
    assert np.all((data.propensity > 0.0) & (data.propensity < 1.0))


def test_rotating_folds_are_disjoint_and_cover_the_sample() -> None:
    roles = make_fold_roles(n=103, n_folds=5, seed=19)
    for role in roles:
        assert np.intersect1d(role.training, role.diagnostic).size == 0
        assert np.intersect1d(role.training, role.evaluation).size == 0
        assert np.intersect1d(role.diagnostic, role.evaluation).size == 0
        combined = np.concatenate((role.training, role.diagnostic, role.evaluation))
        assert np.array_equal(np.sort(combined), np.arange(103))


def test_basis_standardization_uses_only_fitting_observations() -> None:
    train = generate_data(n=300, design="low", overlap_scale=0.5, seed=23)
    evaluation = generate_data(n=100, design="low", overlap_scale=0.5, seed=29)
    basis = ExperimentBasis("rich").fit(train.X)
    phi_before = basis(evaluation.X)
    shifted = evaluation.X.copy()
    shifted[:, 1:] += 100.0
    phi_after = basis(shifted)
    assert phi_before.shape == phi_after.shape
    assert not np.allclose(phi_before, phi_after)
    assert np.allclose(basis(train.X).mean(axis=0), 0.0, atol=1e-10)


def test_squared_candidate_returns_finite_signed_weights() -> None:
    data = generate_data(n=1000, design="low", overlap_scale=0.5, seed=31)
    fit = fit_candidate(
        data.X,
        CandidateSpec("SQ", "rich", 0.5),
        max_iter=1000,
        tolerance=1e-8,
        gradient_tolerance=1e-2,
    )
    assert fit.success
    alpha = fit.predict(data.X)
    assert np.all(np.isfinite(alpha))
    assert np.mean(alpha[data.X[:, 0] == 1.0]) > 0.0
    assert np.mean(alpha[data.X[:, 0] == 0.0]) < 0.0


def test_multiplier_radii_are_invariant_to_column_permutation() -> None:
    rng = np.random.default_rng(37)
    values = rng.normal(size=(500, 7))
    order = np.array([6, 2, 0, 5, 1, 4, 3])
    radii = gaussian_multiplier_mean_radii(values, delta=0.01, draws=500, seed=41)
    permuted = gaussian_multiplier_mean_radii(
        values[:, order], delta=0.01, draws=500, seed=41
    )
    assert np.allclose(radii[order], permuted)


def test_bias_aware_critical_value_increases_with_bias_bound() -> None:
    c0 = bias_aware_critical_value(0.0, coverage=0.95)
    c1 = bias_aware_critical_value(1.0, coverage=0.95)
    assert abs(c0 - 1.959963984540054) < 1e-8
    assert c1 > c0
