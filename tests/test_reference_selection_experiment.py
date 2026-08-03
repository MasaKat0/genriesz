"""Tests for the reference-based loss--link selection experiment.

The properties checked here are the ones the design in
``notebooks/experiments/REFERENCE_SELECTION_PLAN.md`` relies on. Several of them
were promised by the earlier design documents but never implemented: the
rescaling invariance behind experiment E1a, the ``BP(1)`` identity that
justifies excluding it from the candidate grid, the analytic audit formulas, and
the near-orthogonality of the hidden direction that makes the drifting
misspecification family work.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from genriesz.experiments.reference_selection.audit import audit_from_values  # noqa: E402
from genriesz.experiments.reference_selection.candidates import (  # noqa: E402
    BENCHMARK_SPEC,
    CandidateSpec,
    ExperimentBasis,
    FoldLibrary,
    ScaledGenerator,
    ate_branch,
    candidate_grid,
    make_generator,
)
from genriesz.experiments.reference_selection.dgp import (  # noqa: E402
    NOISE_SD,
    THETA0,
    generate_data,
    hidden_direction,
    make_fold_roles,
    stable_seed,
)
from genriesz.experiments.reference_selection.inference import (
    bias_aware_critical_value,  # noqa: E402
)
from genriesz.experiments.reference_selection.selection import (  # noqa: E402
    DeltaBudget,
    candidate_scores,
    effective_sample_ratio,
    gaussian_multiplier_mean_radii,
    minimum_bias_upper_bound,
)
from genriesz.functionals import ATEFunctional  # noqa: E402
from genriesz.generators import BPGenerator, SquaredGenerator  # noqa: E402
from genriesz.glm import GRRGLM  # noqa: E402

PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
PARQUET_TEST = pytest.mark.skipif(
    not PYARROW_AVAILABLE, reason="pyarrow is required for Parquet persistence tests."
)

# --------------------------------------------------------------------------- #
# Data-generating processes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hidden_scale", [0.0, 1.0, 2.0])
def test_ate_is_one_for_every_hidden_scale(hidden_scale: float) -> None:
    """The hidden direction must not move the estimand.

    It enters the treatment index and the untreated regression but never the
    conditional treatment effect, so the calibration sweep of experiment E4
    changes the bias without changing what is being estimated.
    """

    data = generate_data(
        n=200_000, design="low", overlap_scale=1.5, hidden_scale=hidden_scale, seed=17
    )
    assert abs(float(np.mean(data.tau)) - THETA0) < 0.01
    assert np.all(np.isfinite(data.X))
    assert np.all(np.isfinite(data.outcomes()))
    assert np.all((data.propensity > 0.0) & (data.propensity < 1.0))


def _projection_r_squared(target: np.ndarray, features: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(features, target, rcond=None)
    residual = target - features @ beta
    return 1.0 - float(np.mean(residual**2)) / float(np.var(target))


def test_hidden_direction_asymmetry_holds_in_the_spaces_actually_used() -> None:
    """The invariant experiment E4 rests on, checked on the real bases.

    The candidate basis is treatment specific, and treatment depends on the
    hidden direction through the propensity, so the interaction absorbs part of
    it: the unconditional projection understates what the candidates can fit.
    The test therefore uses the treatment-specific bases at the largest
    calibrated scale, where absorption is worst.

    What must hold is the asymmetry, not orthogonality: the candidates leave most
    of the direction unexplained while the correct reference spans it exactly. If
    the reference did not, its own allowance would fail and every candidate bound
    would fail with it.
    """

    from genriesz.experiments.reference_selection.reference import (
        MisspecifiedOutcomeBasis,
        ReferenceOutcomeBasis,
        _correct_features,
    )

    # The largest hidden scale in the committed calibration table.
    data = generate_data(
        n=100_000, design="low", overlap_scale=1.5, hidden_scale=1.383, seed=13
    )
    psi = hidden_direction(data.X[:, 1:])
    intercept = np.ones((data.X.shape[0], 1))

    candidate = _projection_r_squared(psi, ExperimentBasis("rich").raw_features(data.X))
    assert candidate < 0.5, "candidates must leave most of the hidden direction unfitted"

    assert _projection_r_squared(psi, ReferenceOutcomeBasis()(data.X)) > 0.999
    assert (
        _projection_r_squared(psi, np.column_stack((intercept, _correct_features(data.X))))
        > 0.999
    )
    assert _projection_r_squared(psi, MisspecifiedOutcomeBasis()(data.X)) < 0.5


def test_fold_rotation_is_disjoint_and_covers_the_sample() -> None:
    roles = make_fold_roles(n=103, n_folds=5, seed=19)
    counts = np.zeros(103, dtype=int)
    for role in roles:
        assert np.intersect1d(role.training, role.diagnostic).size == 0
        assert np.intersect1d(role.training, role.evaluation).size == 0
        assert np.intersect1d(role.diagnostic, role.evaluation).size == 0
        combined = np.concatenate((role.training, role.diagnostic, role.evaluation))
        assert np.array_equal(np.sort(combined), np.arange(103))
        counts[role.evaluation] += 1
    assert np.all(counts == 1), "each observation evaluates exactly once"


def test_stable_seed_is_order_independent_and_distinct() -> None:
    assert stable_seed(1, "a", 2) == stable_seed(1, "a", 2)
    assert stable_seed(1, "a", 2) != stable_seed(1, "a", 3)
    assert stable_seed(1, "a", 2) != stable_seed(2, "a", 2)


# --------------------------------------------------------------------------- #
# Experiment E1a: raw Bregman objectives are not comparable
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("loss", ["SQ", "UKL"])
def test_rescaling_preserves_the_representer_but_scales_the_objective(loss: str) -> None:
    """The claim underlying the whole selection criterion.

    Multiplying the generator by ``kappa`` leaves the unpenalized fitted
    representer unchanged and multiplies the objective by ``kappa``. A
    cross-validated Bregman objective therefore ranks generators arbitrarily.
    """

    data = generate_data(n=1500, design="low", overlap_scale=1.5, hidden_scale=0.0, seed=31)
    spec = CandidateSpec(loss=loss, dictionary="second_order", penalty_multiplier=0.0)  # type: ignore[arg-type]
    inner = make_generator(spec)

    fitted: dict[float, tuple[np.ndarray, float]] = {}
    for kappa in (0.5, 1.0, 2.0):
        generator = inner if kappa == 1.0 else ScaledGenerator(inner, kappa)
        model = GRRGLM(
            basis=ExperimentBasis("second_order").fit(data.X),
            generator=generator,
            functional=ATEFunctional(treatment_index=0),
            penalty="l1",
            lam=0.0,
            p_norm=1.0,
        )
        fit = model.fit(data.X, max_iter=5000, tol=1e-10)
        assert fit.success
        fitted[kappa] = (
            np.asarray(model.predict_alpha(data.X), dtype=float),
            float(fit.objective_value),
        )

    baseline_alpha, baseline_objective = fitted[1.0]
    scale = float(np.max(np.abs(baseline_alpha)))
    for kappa, (alpha, objective) in fitted.items():
        assert np.max(np.abs(alpha - baseline_alpha)) < 1e-3 * scale
        assert objective == pytest.approx(kappa * baseline_objective, rel=1e-6)


def test_bp_one_matches_the_squared_curvature() -> None:
    """Why ``BP(1)`` is left out of the candidate grid.

    Its second derivative is constantly two on a fixed branch, the same as the
    squared generator, so it carries no additional Bregman geometry.
    """

    rng = np.random.default_rng(11)
    n = 4000
    D = rng.binomial(1, 0.5, n).astype(float)
    X = np.column_stack((D, rng.normal(size=(n, 3))))
    magnitude = 1.0 + np.abs(rng.normal(size=n))
    alpha = np.where(D >= 0.5, magnitude, -magnitude)

    bp_one = BPGenerator(C=1.0, omega=1.0, branch_fn=ate_branch)
    squared = SquaredGenerator(C=0.0)
    assert np.allclose(bp_one.grad2(X, alpha), squared.grad2(X, alpha))
    assert np.allclose(bp_one.grad2(X, alpha), 2.0)


def test_candidate_grid_has_ninety_distinct_labels() -> None:
    specs = candidate_grid()
    assert len(specs) == 90
    assert len({spec.label for spec in specs}) == 90
    assert all(spec.omega != 1.0 for spec in specs), "BP(1) must not be in the grid"


# --------------------------------------------------------------------------- #
# Dictionaries and fitting
# --------------------------------------------------------------------------- #


def test_standardization_uses_only_the_fitting_observations() -> None:
    train = generate_data(n=600, design="low", overlap_scale=0.5, hidden_scale=0.0, seed=23)
    evaluation = generate_data(n=200, design="low", overlap_scale=0.5, hidden_scale=0.0, seed=29)
    basis = ExperimentBasis("rich").fit(train.X)
    before = basis(evaluation.X)
    shifted = evaluation.X.copy()
    shifted[:, 1:] += 100.0
    assert not np.allclose(before, basis(shifted))
    assert np.allclose(basis(train.X).mean(axis=0), 0.0, atol=1e-10)


def test_library_shares_one_basis_and_one_generator_per_configuration() -> None:
    """The efficiency invariant the runtime budget depends on."""

    data = generate_data(n=240, design="low", overlap_scale=1.5, hidden_scale=0.0, seed=41)
    specs = (
        CandidateSpec("SQ", "linear", 0.0),
        CandidateSpec("SQ", "second_order", 0.0),
        CandidateSpec("SQ", "rich", 0.0),
        CandidateSpec("UKL", "linear", 0.0),
        CandidateSpec("BKL", "linear", 0.0),
        CandidateSpec("BP", "linear", 0.0, omega=0.25),
        CandidateSpec("BP", "linear", 0.0, omega=0.5),
    )
    library = FoldLibrary(data.X, specs, max_iter=200, tolerance=1e-8)
    assert len(library.bases) == 3
    assert len({id(generator) for generator in library.generators}) == 5
    sq_generators = [
        generator
        for spec, generator in zip(library.specs, library.generators, strict=True)
        if spec.loss == "SQ"
    ]
    assert len({id(generator) for generator in sq_generators}) == 1


def test_squared_candidate_returns_finite_signed_weights() -> None:
    data = generate_data(n=1500, design="low", overlap_scale=0.5, hidden_scale=0.0, seed=31)
    library = FoldLibrary(data.X, (BENCHMARK_SPEC,), max_iter=1000, tolerance=1e-8)
    assert library.fits[0].success
    alpha = library.alpha_matrix(data.X)[:, 0]
    assert np.all(np.isfinite(alpha))
    assert np.mean(alpha[data.X[:, 0] == 1.0]) > 0.0
    assert np.mean(alpha[data.X[:, 0] == 0.0]) < 0.0


def test_alpha_matrix_agrees_with_individual_prediction() -> None:
    """The batched evaluation must match a per-candidate ``predict_alpha``."""

    data = generate_data(n=800, design="low", overlap_scale=1.5, hidden_scale=0.0, seed=53)
    specs = (
        CandidateSpec("SQ", "linear", 0.0),
        CandidateSpec("UKL", "linear", 0.5),
        CandidateSpec("BP", "second_order", 0.0, omega=0.5),
    )
    library = FoldLibrary(data.X, specs, max_iter=1000, tolerance=1e-8)
    batched = library.alpha_matrix(data.X)
    for j, spec in enumerate(specs):
        if not library.fits[j].success:
            continue
        basis = library.bases[spec.dictionary]
        v = np.asarray(basis(data.X), dtype=float) @ library.fits[j].beta
        expected = np.asarray(library.generators[j].inv_grad(data.X, v), dtype=float)
        assert np.allclose(batched[:, j], expected)


def test_effective_sample_ratio_is_scale_free_and_bracketed() -> None:
    """Uniform weight gives one, a single spike gives ``1/n``, and scale is irrelevant."""

    n = 50
    uniform = np.full(n, 2.0)
    spike = np.zeros(n)
    spike[0] = 1.0
    alternating = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    ratios = effective_sample_ratio(np.column_stack([uniform, spike, alternating]))
    assert np.isclose(ratios[0], 1.0)
    assert np.isclose(ratios[1], 1.0 / n)
    assert np.isclose(ratios[2], 1.0)
    # The Kish ratio is homogeneous of degree zero, so screening on it cannot
    # depend on the units of the target parameter. That must hold to the edge of
    # the floating-point range: without peak normalization, uniform weights of
    # ``2e307`` make ``(sum |a|)^2`` overflow to ``inf / inf = nan`` and uniform
    # tiny weights underflow to zero, both of which would silently change
    # admissibility. (``1e308`` would overflow in the test input itself.)
    for scale in (137.0, 1e307, 1e-300, np.nextafter(0.0, 1.0)):
        scaled = effective_sample_ratio(np.column_stack([uniform, spike]) * scale)
        assert np.allclose(scaled, ratios[:2]), scale
    assert np.isclose(effective_sample_ratio(np.zeros((n, 1)))[0], 0.0)
    # A non-finite entry of any kind is a failed fit, not a concentrated one.
    for bad in (np.inf, -np.inf, np.nan):
        with_bad = np.column_stack([uniform, uniform])
        with_bad[3, 1] = bad
        ratios_bad = effective_sample_ratio(with_bad)
        assert np.isclose(ratios_bad[0], 1.0)
        assert np.isnan(ratios_bad[1])


def test_weight_screen_shrinks_the_admissible_set_monotonically() -> None:
    """A stricter pre-specified restriction may only remove candidates."""

    data = generate_data(n=600, design="low", overlap_scale=1.5, hidden_scale=0.0, seed=97)
    specs = candidate_grid()[:20]
    library = FoldLibrary(data.X, specs, max_iter=1000, tolerance=1e-8)
    gamma = np.zeros(data.X.shape[0])
    contrast = np.zeros(data.X.shape[0])

    def admissible_at(threshold: float | None) -> np.ndarray:
        _, admissible, _, _ = candidate_scores(
            library, data.X, data.y, contrast, gamma, min_ess_ratio=threshold
        )
        return admissible

    unrestricted = admissible_at(None)
    # Guard against a degenerate screen: monotonicity alone would also hold for
    # an implementation that rejects everything at any positive threshold.
    assert np.sum(unrestricted) > 0
    assert np.sum(admissible_at(0.05)) > 0
    assert np.array_equal(admissible_at(0.0), unrestricted)
    previous = unrestricted
    for threshold in (0.05, 0.2, 0.5):
        current = admissible_at(threshold)
        assert np.all(current <= previous)
        previous = current
    # The strictest threshold below one still cannot admit anything new.
    assert np.sum(admissible_at(0.99)) <= np.sum(unrestricted)


def test_screened_candidate_keeps_its_diagnostics_but_loses_its_scores() -> None:
    """The screen must remove a candidate from selection without erasing the record.

    ``scores`` drive selection, so they become ``nan``; ``ess_ratio`` and
    ``max_abs_alpha`` describe the fit and must survive so that the candidate
    table can say what was screened and why.
    """

    data = generate_data(n=600, design="low", overlap_scale=1.5, hidden_scale=0.0, seed=43)
    specs = candidate_grid()[:20]
    library = FoldLibrary(data.X, specs, max_iter=1000, tolerance=1e-8)
    zeros = np.zeros(data.X.shape[0])
    _, unrestricted, _, ess = candidate_scores(library, data.X, data.y, zeros, zeros)
    values = ess[unrestricted]
    assert values.min() < values.max(), "grid too homogeneous for this test"
    threshold = float((values.min() + values.max()) / 2.0)
    scores, admissible, max_weight, ess_after = candidate_scores(
        library, data.X, data.y, zeros, zeros, min_ess_ratio=threshold
    )
    screened = unrestricted & ~admissible
    kept = unrestricted & admissible
    assert screened.any() and kept.any()
    assert np.all(np.isnan(scores[:, screened]))
    assert np.all(np.isfinite(ess_after[screened]))
    assert np.all(ess_after[screened] < threshold)
    assert np.all(np.isfinite(max_weight[screened]))
    assert np.all(np.isfinite(scores[:, kept]))


def test_min_ess_ratio_changes_the_configuration_digest() -> None:
    """A screened run must never silently reuse batches from an unscreened one."""

    from genriesz.experiments.reference_selection.grids import experiment_config
    from genriesz.experiments.reference_selection.runner import (
        Numerics,
        Scenario,
        configuration_digest,
        configuration_record,
    )

    scenario = Scenario(
        grid="A", design="low", sample_size=400, overlap_scale=1.5, target_t=0.0, hidden_scale=0.0
    )
    common = {"scenarios": (scenario,), "replications_by_grid": {"A": 1}}
    plain = experiment_config(**common)
    screened = experiment_config(**common, numerics=Numerics(min_ess_ratio=0.5))
    assert configuration_record(screened)["numerics"]["min_ess_ratio"] == 0.5
    assert configuration_digest(plain) != configuration_digest(screened)


def test_weight_screen_rejects_a_threshold_outside_the_unit_interval() -> None:
    data = generate_data(n=300, design="low", overlap_scale=1.5, hidden_scale=0.0, seed=11)
    library = FoldLibrary(data.X, (BENCHMARK_SPEC,), max_iter=500, tolerance=1e-8)
    zeros = np.zeros(data.X.shape[0])
    with pytest.raises(ValueError, match="min_ess_ratio"):
        candidate_scores(library, data.X, data.y, zeros, zeros, min_ess_ratio=1.0)
    with pytest.raises(ValueError, match="min_ess_ratio"):
        candidate_scores(library, data.X, data.y, zeros, zeros, min_ess_ratio=-0.1)


# --------------------------------------------------------------------------- #
# Analytic audit
# --------------------------------------------------------------------------- #


def test_analytic_audit_matches_a_noisy_monte_carlo_average() -> None:
    """The closed forms of plan section 10 must reproduce the score moments.

    ``B = E[(alpha_0 - alpha_hat)(gamma_hat - gamma_0)]`` should equal the mean
    of the realized score contribution minus the estimand, and the analytic
    variance should equal its sampling variance.
    """

    rng = np.random.default_rng(97)
    n = 400_000
    Z = rng.normal(size=(n, 2))
    D = rng.binomial(1, 0.5, n).astype(float)
    e = np.full(n, 0.5)
    alpha0 = D / e - (1.0 - D) / (1.0 - e)
    tau = 1.0 + 0.3 * Z[:, 0]
    gamma0 = Z[:, 0] + D * tau
    # A deliberately wrong nuisance pair so the bias is not degenerate.
    alpha_hat = alpha0 * 0.8 + 0.3 * Z[:, 1]
    gamma_hat = gamma0 + 0.25 * Z[:, 0] - 0.1
    m_hat = tau + 0.0

    bias, variance, _ = audit_from_values(
        alpha_hat=alpha_hat,
        alpha0=alpha0,
        gamma_hat=gamma_hat,
        gamma0=gamma0,
        m_hat=m_hat,
        n_evaluation=100,
    )
    y = gamma0 + NOISE_SD * rng.normal(size=n)
    realized = m_hat + alpha_hat * (y - gamma_hat)
    assert bias == pytest.approx(float(np.mean(realized)) - THETA0, abs=0.01)
    assert variance == pytest.approx(float(np.var(realized)), rel=0.01)


def test_audit_of_the_truth_reference_has_zero_bias() -> None:
    bias, variance, _ = audit_from_values(
        alpha_hat=np.array([1.0, -2.0, 3.0]),
        alpha0=np.array([1.0, -2.0, 3.0]),
        gamma_hat=np.array([0.5, 0.25, -1.0]),
        gamma0=np.array([0.5, 0.25, -1.0]),
        m_hat=np.array([1.0, 1.0, 1.0]),
        n_evaluation=10,
    )
    assert bias == pytest.approx(0.0, abs=1e-15)
    assert variance > 0.0


# --------------------------------------------------------------------------- #
# Diagnostics, budget, and inference
# --------------------------------------------------------------------------- #


def test_multiplier_radii_are_invariant_to_column_permutation() -> None:
    rng = np.random.default_rng(37)
    values = rng.normal(size=(500, 7))
    order = np.array([6, 2, 0, 5, 1, 4, 3])
    radii = gaussian_multiplier_mean_radii(values, delta=0.01, draws=500, seed=41)
    permuted = gaussian_multiplier_mean_radii(values[:, order], delta=0.01, draws=500, seed=41)
    assert np.allclose(radii[order], permuted)


def test_delta_budget_does_not_overspend() -> None:
    """The allocation error that the earlier implementation carried.

    Charging the variance bound to the coverage budget made the within-fold
    allocations sum to one and a half times ``delta``. The variance bound
    belongs to the risk statement, not to the coverage statement, so it has its
    own budget.
    """

    budget = DeltaBudget()
    assert budget.bias_budget_is_exhausted()
    assert budget.mean_radius_delta + budget.reference_delta == pytest.approx(budget.fold_delta)
    assert budget.fold_delta * budget.n_folds == pytest.approx(budget.delta)
    ellipsoid_spend = 2.0 * (1.0 - budget.ellipsoid_probability)
    assert ellipsoid_spend == pytest.approx(budget.reference_delta)
    assert budget.normal_coverage == pytest.approx(1.0 - (budget.tau - budget.delta))


def test_delta_budget_rejects_an_inconsistent_allocation() -> None:
    with pytest.raises(ValueError):
        DeltaBudget(tau=0.05, delta=0.05)


def test_minimum_bias_bound_is_no_larger_than_either_bound() -> None:
    first = np.array([1.0, 2.0, np.nan])
    second = np.array([0.5, 3.0, 4.0])
    merged = minimum_bias_upper_bound({"a": first, "b": second})
    assert np.allclose(merged, np.array([0.5, 2.0, 4.0]), equal_nan=True)


def test_bias_aware_critical_value_increases_with_the_bias_bound() -> None:
    c0 = bias_aware_critical_value(0.0, coverage=0.95)
    assert abs(c0 - 1.959963984540054) < 1e-8
    values = [bias_aware_critical_value(t, coverage=0.95) for t in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert all(
        later > earlier for earlier, later in zip(values, values[1:], strict=False)
    )


def test_bias_aware_critical_value_attains_its_nominal_coverage() -> None:
    """The critical value must solve the bounded-normal-mean problem exactly."""

    from scipy import stats

    for t in (0.0, 0.75, 2.5):
        c = bias_aware_critical_value(t, coverage=0.96)
        attained = stats.norm.cdf(c - t) - stats.norm.cdf(-c - t)
        assert attained == pytest.approx(0.96, abs=1e-9)


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #


def _fit_low_reference(name: str, hidden_scale: float, seed: int = 5):
    from genriesz.experiments.reference_selection.audit import audit_from_values as _audit
    from genriesz.experiments.reference_selection.candidates import fit_outcome
    from genriesz.experiments.reference_selection.reference import fit_logistic_reference

    data = generate_data(
        n=3000, design="low", overlap_scale=1.5, hidden_scale=hidden_scale, seed=seed
    )
    roles = make_fold_roles(3000, 5, seed + 1)[0]
    X_train = data.X[roles.training]
    y_train = data.outcomes()[roles.training]
    fit_outcome(X_train, y_train, design="low", seed=seed)
    reference = fit_logistic_reference(
        X_train,
        y_train,
        name=name,  # type: ignore[arg-type]
        ellipsoid_probability=DeltaBudget().ellipsoid_probability,
        max_iter=1000,
        tolerance=1e-8,
    )
    audit = generate_data(
        n=100_000,
        design="low",
        overlap_scale=1.5,
        hidden_scale=hidden_scale,
        seed=seed + 2,
        with_outcome=False,
    )
    drift, _, _ = _audit(
        alpha_hat=reference.alpha(audit.X),
        alpha0=audit.alpha0,
        gamma_hat=reference.gamma(audit.X),
        gamma0=audit.gamma0,
        m_hat=reference.contrast(audit.X),
        n_evaluation=600,
    )
    return reference, drift


@pytest.mark.parametrize("hidden_scale", [0.0, 1.0])
def test_correct_reference_respects_its_allowance(hidden_scale: float) -> None:
    """The premise of Theorem ``data_dependent_bias`` must hold for the reference.

    The correct reference includes the hidden direction in both nuisances, so it
    stays valid as the calibrated bias grows. If it did not, every candidate
    bound would fail at once and the sweep would measure nothing.
    """

    reference, drift = _fit_low_reference("correct", hidden_scale)
    assert abs(drift) <= reference.honest_allowance


def test_misspecified_reference_breaks_its_allowance_when_the_hidden_term_is_active() -> None:
    """Experiment E5 needs a reference that is genuinely invalid, not merely noisy."""

    _, drift_null = _fit_low_reference("misspecified", 0.0)
    reference, drift_active = _fit_low_reference("misspecified", 1.0)
    assert abs(drift_null) < abs(drift_active)
    assert abs(drift_active) > reference.honest_allowance


def test_truth_reference_has_no_allowance() -> None:
    from genriesz.experiments.reference_selection.reference import TruthReference

    reference = TruthReference(design="low", overlap_scale=1.5, hidden_scale=0.7)
    assert reference.honest_allowance == 0.0
    assert reference.allowance(2.0) == 0.0


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def _unit_config(max_workers: int):
    from genriesz.experiments.reference_selection.candidates import FIXED_BENCHMARKS
    from genriesz.experiments.reference_selection.grids import experiment_config
    from genriesz.experiments.reference_selection.runner import Numerics, Scenario

    scenario = Scenario(
        grid="A",
        design="low",
        sample_size=180,
        overlap_scale=1.5,
        target_t=0.0,
        hidden_scale=0.0,
    )
    return experiment_config(
        scenarios=(scenario,),
        replications_by_grid={"A": 1},
        candidate_specs=tuple(FIXED_BENCHMARKS.values()),
        numerics=Numerics(
            low_integration_size=500,
            multiplier_draws=20,
            allowance_scales=(1.0,),
            reference_constants=(1.0,),
        ),
        max_workers=max_workers,
    )


def test_unit_run_produces_every_table_and_rule() -> None:
    from genriesz.experiments.reference_selection.runner import TABLES, run_repetition
    from genriesz.experiments.reference_selection.selection import RULES

    config = _unit_config(1)
    tables = run_repetition((config, config.scenarios[0], 0))
    assert set(tables) == set(TABLES)
    assert not tables["repetition"].empty
    produced = set(tables["repetition"]["rule"])
    assert set(RULES) - {"proposed"} <= produced | {"proposed"}
    assert {"proposed", "bregman_cv", "lsif_cv", "oracle"} <= produced
    # Failures must be recorded, not dropped.
    assert set(tables["repetition"]["complete"]) <= {True, False}


def test_random_number_allocation_does_not_depend_on_worker_count() -> None:
    """Changing the worker count leaves the job list and seed allocation unchanged."""

    from dataclasses import replace as dataclass_replace

    from genriesz.experiments.reference_selection.runner import (
        configuration_digest,
        expand_jobs,
    )

    serial = _unit_config(1)
    parallel = dataclass_replace(serial, max_workers=4)
    serial_jobs = expand_jobs(serial)
    parallel_jobs = expand_jobs(parallel)
    assert [job[1:] for job in serial_jobs] == [job[1:] for job in parallel_jobs]
    assert configuration_digest(serial) == configuration_digest(parallel)


@PARQUET_TEST
def test_batched_run_writes_and_reloads(tmp_path) -> None:
    from genriesz.experiments.reference_selection.runner import (
        TABLES,
        load_experiment,
        run_experiment,
    )

    config = _unit_config(1)
    run_experiment(config, tmp_path / "unit_run")
    loaded = load_experiment(tmp_path / "unit_run")
    assert set(loaded) == set(TABLES)
    assert not loaded["repetition"].empty
    assert (tmp_path / "unit_run" / "run_manifest.json").exists()


@PARQUET_TEST
def test_load_experiment_refuses_an_incomplete_publication_run(tmp_path) -> None:
    """Reporting must not read a partial set of manifest batches."""

    from genriesz.experiments.reference_selection.runner import load_experiment, run_experiment

    config = _unit_config(1)
    output = tmp_path / "unit_run"
    run_experiment(config, output)
    missing = next(output.glob("oracle_*.parquet"))
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_experiment(output)


@PARQUET_TEST
def test_a_changed_configuration_refuses_to_reuse_batches(tmp_path) -> None:
    """Resuming must not read another configuration's Parquet as this run's output.

    Skipping a batch on file existence alone would let a changed calibration
    table, multiplier count, or scenario list report stale numbers under the new
    configuration's name.
    """

    from dataclasses import replace as dataclass_replace

    from genriesz.experiments.reference_selection.runner import Numerics, run_experiment

    config = _unit_config(1)
    run_experiment(config, tmp_path / "run")
    changed = dataclass_replace(
        config,
        numerics=dataclass_replace(config.numerics, multiplier_draws=123),
    )
    with pytest.raises(RuntimeError, match="different configuration"):
        run_experiment(changed, tmp_path / "run")
    assert Numerics is not None


def test_rescaling_invariance_survives_the_l1_penalty() -> None:
    """The escape route a referee will look for, closed.

    The invariance holds whenever the penalty is positively homogeneous of
    degree one, and the experiments use an l1 penalty. So raw Bregman objectives
    are incomparable for the penalized estimator that is actually fitted, not
    only for an unpenalized idealization. Under l2 the fitted representer does
    move, which is recorded here so the l1 result is not mistaken for a
    universal one.
    """

    from genriesz.experiments.reference_selection.rescaling import rescaling_table

    table = rescaling_table(n=1200, losses=("SQ",))
    penalized = table.loc[
        table["penalty"].eq("l1")
        & np.isclose(table["penalty_multiplier"], 1.0)
        & ~np.isclose(table["kappa"], 1.0)
    ]
    assert not penalized.empty
    assert (penalized["alpha_max_deviation"] < 1e-2).all()
    assert np.allclose(penalized["objective_ratio"], penalized["kappa"], rtol=1e-4)
    assert np.allclose(penalized["heldout_ratio"], penalized["kappa"], rtol=1e-4)

    ridge = table.loc[table["penalty"].eq("l2") & ~np.isclose(table["kappa"], 1.0)]
    assert not ridge.empty
    assert (ridge["alpha_max_deviation"] > 1e-2).any()


def test_rescaling_records_restricted_holdout_domain_without_substitution() -> None:
    """A held-out BP domain failure is data, not an exception or a clipped value."""

    from genriesz.experiments.reference_selection.rescaling import rescaling_table

    table = rescaling_table(
        n=2000,
        losses=("BP",),
        penalties=(("l1", 1.0, 0.0),),
    )
    baseline = table.loc[np.isclose(table["kappa"], 1.0)].iloc[0]
    assert baseline["status"] == "converged"
    assert baseline["heldout_status"] == "dual_domain_failure"
    assert 0 < int(baseline["heldout_valid_rows"]) < int(
        baseline["heldout_total_rows"]
    )
    assert np.isnan(float(baseline["heldout_bregman"]))
    assert np.isnan(float(baseline["heldout_ratio"]))


def test_minimum_bias_bound_keeps_all_nan_columns_missing() -> None:
    merged = minimum_bias_upper_bound(
        {"a": np.array([1.0, np.nan]), "b": np.array([2.0, np.nan])}
    )
    assert merged[0] == pytest.approx(1.0)
    assert np.isnan(merged[1])


def test_delta_budget_spends_exactly_delta_on_the_bias_event() -> None:
    """The invariant the earlier implementation violated.

    Charging the variance bound to the coverage budget made the total spend one
    and a half times ``delta``. The variance bound belongs to the risk statement,
    so adding it here must overspend -- that is what makes the separate budget
    necessary rather than cosmetic.
    """

    budget = DeltaBudget()
    assert budget.total_bias_spend() == pytest.approx(budget.delta)
    assert budget.bias_budget_is_exhausted()
    # The earlier allocation charged the variance bound at the same delta/(2K)
    # rate to the same budget, on top of the mean radius and the reference
    # allowance: three events at delta/(2K) per fold, or 1.5 delta in total.
    historical = budget.n_folds * (
        budget.mean_radius_delta + budget.reference_delta + budget.mean_radius_delta
    )
    assert historical == pytest.approx(1.5 * budget.delta)
    assert historical > budget.delta
    with pytest.raises(ValueError):
        DeltaBudget(delta_variance=0.0)


def test_split_interval_survives_a_failure_in_another_fold() -> None:
    """A failure elsewhere must not be scored as a split-interval coverage failure.

    The single-split intervals verify a single-split theorem, so gating them on
    the cross-fitted completeness flag would depress the headline uniform
    coverage for reasons unrelated to the theorem.
    """

    import pandas as pd

    from genriesz.experiments.reference_selection import report

    repetitions = pd.DataFrame(
        [
            {
                "repetition": 0,
                "rule": "proposed",
                "reference": "correct",
                "allowance_scale": 1.0,
                "grid": "B",
                "design": "low",
                "sample_size": 1000,
                "overlap_scale": 1.5,
                "target_t": 1.0,
                "complete": False,
                "split_available": True,
                "bias": 0.0,
                "squared_error": 0.0,
                "wald_split_covers": True,
                "wald_split_length": 0.4,
                "bias_aware_split_covers": True,
                "bias_aware_split_length": 0.9,
                "conservative_cf_covers": False,
                "conservative_cf_length": np.nan,
            }
        ]
    )
    table = report._coverage_frame(repetitions, ["rule"])
    assert table["bias_aware_split_coverage"].iloc[0] == pytest.approx(1.0)
    assert table["wald_split_coverage"].iloc[0] == pytest.approx(1.0)
    assert table["conservative_cf_coverage"].iloc[0] == pytest.approx(0.0)


def test_audit_caches_can_be_cleared() -> None:
    from genriesz.experiments.reference_selection import audit

    sample = audit.integration_sample(
        design="low", overlap_scale=1.5, hidden_scale=0.0, size=1000, seed=3
    )
    again = audit.integration_sample(
        design="low", overlap_scale=1.5, hidden_scale=0.0, size=1000, seed=3
    )
    assert again is sample
    audit.clear_caches()
    fresh = audit.integration_sample(
        design="low", overlap_scale=1.5, hidden_scale=0.0, size=1000, seed=3
    )
    assert fresh is not sample
    assert np.array_equal(fresh.X, sample.X)


def test_numerics_rejects_a_fold_count_that_disagrees_with_the_budget() -> None:
    """The budget is allocated per fold, so the two counts must not drift apart.

    With ten folds and a five-fold budget the bias event would be charged at
    delta/(2*5) ten times over, doubling the declared spend while every
    within-budget check still passed.
    """

    from genriesz.experiments.reference_selection.runner import Numerics

    with pytest.raises(ValueError, match="must equal"):
        Numerics(n_folds=10)
    assert Numerics(n_folds=10, budget=DeltaBudget(n_folds=10)).n_folds == 10


def test_reference_check_reports_a_nonfinite_comparison_as_undecidable() -> None:
    """A blown-up reference score must not be recorded as a passing check."""

    from genriesz.experiments.reference_selection.reference import ReferenceCheck

    clean = ReferenceCheck("a", "b", difference=0.1, radius=0.05, allowance_sum=0.01)
    assert clean.checkable
    assert clean.violated is True

    broken = ReferenceCheck("a", "b", difference=np.nan, radius=0.05, allowance_sum=0.01)
    assert not broken.checkable
    assert broken.violated is None


def test_a_failed_reference_is_not_used_for_selection_or_inference(monkeypatch) -> None:
    """A reference without a valid allowance carries no guarantee.

    Recording the failure in a status column while still forming its bias bound,
    selecting with it, and reporting its interval would be a fail-open on the
    premise of Theorem ``data_dependent_bias``.
    """

    from genriesz.experiments.reference_selection import runner as runner_module
    from genriesz.experiments.reference_selection.audit import integration_sample, scenario_key
    from genriesz.experiments.reference_selection.candidates import FIXED_BENCHMARKS
    from genriesz.experiments.reference_selection.runner import Numerics, Scenario, run_fold

    scenario = Scenario(
        grid="A",
        design="low",
        sample_size=240,
        overlap_scale=1.5,
        target_t=0.0,
        hidden_scale=0.0,
    )
    numerics = Numerics(low_integration_size=1000, multiplier_draws=20)
    data = generate_data(
        n=scenario.sample_size,
        design="low",
        overlap_scale=1.5,
        hidden_scale=0.0,
        seed=101,
    )
    shared = dict(
        design="low", overlap_scale=1.5, hidden_scale=0.0, size=1000, seed=102
    )
    kwargs = dict(
        scenario=scenario,
        numerics=numerics,
        data=data,
        integration=integration_sample(**shared),
        integration_key=scenario_key(**shared),
        roles=make_fold_roles(scenario.sample_size, 5, 103)[0],
        specs=tuple(FIXED_BENCHMARKS.values()),
        scenario_seed=104,
        repetition=0,
        fold_index=0,
    )
    healthy = run_fold(**kwargs)
    assert any(key[1] == "misspecified" for key in healthy.selection)

    original = runner_module._build_references

    def break_misspecified(*args, **kw):
        references = original(*args, **kw)
        references["misspecified"].success = False
        references["misspecified"].status = "logistic_separation"
        return references

    monkeypatch.setattr(runner_module, "_build_references", break_misspecified)
    degraded = run_fold(**kwargs)

    assert not any(key[1] == "misspecified" for key in degraded.selection)
    assert not any(row["reference"] == "misspecified" for row in degraded.rows_bound)
    # The minimum bound of Proposition several_references needs both members.
    assert not any(key[1] == "min" for key in degraded.selection)
    # The surviving references are unaffected.
    assert any(key[1] == "correct" for key in degraded.selection)


def test_every_configured_procedure_appears_even_when_it_always_fails() -> None:
    """Failures must stay in the denominator, including total ones.

    Building the row set from the folds that succeeded would delete a procedure
    from its own denominator precisely when it failed everywhere, which is the
    reporting convention the plan forbids.
    """

    from genriesz.experiments.reference_selection.runner import expected_procedures, run_repetition

    config = _unit_config(1)
    tables = run_repetition((config, config.scenarios[0], 0))
    expected = set(expected_procedures(config.scenarios[0].design, config.numerics))
    produced = set(
        map(tuple, tables["repetition"][["rule", "reference", "allowance_scale"]].to_numpy())
    )
    assert produced == expected
    # fixed_bkl leaves its domain on every ATE fold, so it must be present and
    # marked incomplete rather than missing.
    bkl = tables["repetition"].loc[tables["repetition"]["rule"].eq("fixed_bkl")]
    assert len(bkl) == 1
    assert not bool(bkl["complete"].iloc[0])


@PARQUET_TEST
def test_batches_without_a_manifest_are_refused(tmp_path) -> None:
    """Parquet of unknown provenance must not be adopted by a new run."""

    from genriesz.experiments.reference_selection.runner import load_experiment, run_experiment

    config = _unit_config(1)
    run_experiment(config, tmp_path / "run")
    (tmp_path / "run" / "run_manifest.json").unlink()
    with pytest.raises(RuntimeError, match="no run_manifest"):
        run_experiment(config, tmp_path / "run")
    with pytest.raises(FileNotFoundError, match="no run_manifest"):
        load_experiment(tmp_path / "run")


@PARQUET_TEST
def test_batch_files_beyond_the_run_are_refused(tmp_path) -> None:
    """A shorter re-run must not read the longer run's leftover batches."""

    import shutil

    from genriesz.experiments.reference_selection.runner import load_experiment, run_experiment

    config = _unit_config(1)
    run_experiment(config, tmp_path / "run")
    source = next((tmp_path / "run").glob("candidate_*.parquet"))
    shutil.copy(source, tmp_path / "run" / "candidate_00099.parquet")
    with pytest.raises(RuntimeError, match="beyond"):
        load_experiment(tmp_path / "run")


def test_digest_tracks_the_expanded_job_list(tmp_path) -> None:
    """Batch contents depend on the job list, not only on the declared settings.

    The expanded job list depends on the replication counts, so the digest must
    change when those counts change.
    """

    from dataclasses import replace as dataclass_replace

    from genriesz.experiments.reference_selection.runner import configuration_digest

    config = _unit_config(1)
    baseline = configuration_digest(config)
    assert configuration_digest(dataclass_replace(config, batch_size=3)) != baseline

    changed_counts = dataclass_replace(config, replications_by_grid={"A": 2})
    assert configuration_digest(changed_counts) != baseline
    assert configuration_digest(config) == baseline


def test_reference_check_rate_is_fold_level_with_a_clustered_error() -> None:
    """The reported quantity must stay the fold-level violation rate.

    Collapsing each replication to "did it fire at least once" would silently
    change the estimand: two replications of two folds violating once and never
    give a fold rate of 0.25 but an any-fold rate of 0.5. Undecidable folds are
    excluded from the rate rather than counted as passes, and the standard error
    is clustered by replication because folds share a sample.
    """

    import pandas as pd

    from genriesz.experiments.reference_selection import report

    scenario = {
        "grid": "A",
        "design": "low",
        "sample_size": 400,
        "overlap_scale": 1.5,
        "target_t": 0.0,
        "first": "correct",
        "second": "misspecified",
        "difference": 0.1,
        "radius": 0.05,
        "allowance_sum": 0.01,
    }
    rows = [
        {**scenario, "repetition": 0, "fold": 0, "checkable": True, "violated": True},
        {**scenario, "repetition": 0, "fold": 1, "checkable": True, "violated": False},
        {**scenario, "repetition": 1, "fold": 0, "checkable": True, "violated": False},
        {**scenario, "repetition": 1, "fold": 1, "checkable": True, "violated": False},
    ]
    table = report.reference_check_table({"check": pd.DataFrame(rows)})
    assert table["violation_rate"].iloc[0] == pytest.approx(0.25)
    # Two replication rates, 0.5 and 0.0: sd 0.3536 over sqrt(2).
    assert table["violation_mcse"].iloc[0] == pytest.approx(0.25)
    assert table["undecidable_rate"].iloc[0] == pytest.approx(0.0)

    # An undecidable fold is excluded from the rate, not counted as a pass. The
    # rate is the ratio of totals: one violation among three decidable folds.
    # Averaging the per-replication rates would give 0.5 instead.
    rows[1] = {**scenario, "repetition": 0, "fold": 1, "checkable": False, "violated": None}
    partial = report.reference_check_table({"check": pd.DataFrame(rows)})
    assert partial["violation_rate"].iloc[0] == pytest.approx(1.0 / 3.0)
    assert partial["undecidable_rate"].iloc[0] == pytest.approx(0.25)
    assert partial["decidable_folds"].iloc[0] == pytest.approx(3.0)

    # With nothing decidable the rate is missing, not zero.
    blind = [{**row, "checkable": False, "violated": None} for row in rows]
    empty = report.reference_check_table({"check": pd.DataFrame(blind)})
    assert np.isnan(empty["violation_rate"].iloc[0])
    assert empty["decidable_replications"].iloc[0] == pytest.approx(0.0)


def test_scaled_bp_uses_the_scaled_exact_dual_domain() -> None:
    """Generator rescaling must also rescale BP's linear dual constraints."""

    from genriesz.experiments.reference_selection.rescaling import rescaling_table

    table = rescaling_table(
        n=2000,
        losses=("BP",),
        penalties=(("l1", 1.0, 0.0),),
    )
    assert len(table) == 3
    assert table["status"].isin(("converged", "closed_form")).all()
    assert (table["alpha_max_deviation"] < 1e-2).all()
    assert np.allclose(table["objective_ratio"], table["kappa"], rtol=1e-4)


def test_rescaling_records_heldout_domain_failure_without_substitution() -> None:
    """An exact BP link that fails on held-out rows has no validation loss."""

    from genriesz.experiments.reference_selection.rescaling import rescaling_table

    table = rescaling_table(
        n=2000,
        losses=("BP",),
        penalties=(("l1", 1.0, 0.0),),
    )
    failed = table.loc[table["heldout_status"].eq("dual_domain_failure")]
    assert not failed.empty
    assert (failed["heldout_valid_fraction"] < 1.0).all()
    assert failed["heldout_bregman"].isna().all()
    assert failed["heldout_ratio"].isna().all()


def test_candidate_admissibility_uses_the_l1_kkt_residual(monkeypatch) -> None:
    """A sparse l1 fit is judged by its KKT condition, not raw gradient size."""

    from genriesz.experiments.reference_selection.candidates import (
        CandidateSpec,
        ExperimentBasis,
        fit_candidate_beta,
        make_generator,
    )
    from genriesz.glm import GRRGLM, FitResult

    data = generate_data(n=120, design="low", overlap_scale=1.5, hidden_scale=0.0, seed=91)
    spec = CandidateSpec(loss="SQ", dictionary="linear", penalty_multiplier=1.0)
    basis = ExperimentBasis("linear").fit(data.X)
    beta = np.zeros(basis.n_features, dtype=float)

    def fitted_with_valid_kkt(self, X, **kwargs):
        return FitResult(
            beta=beta,
            success=True,
            message="",
            n_iter=1,
            status="converged",
            objective_value=0.0,
            gradient_norm=1.0,
            kkt_residual=1e-6,
            clip_binding_rate=float("nan"),
            fit_time=0.0,
        )

    monkeypatch.setattr(GRRGLM, "fit", fitted_with_valid_kkt)
    fitted = fit_candidate_beta(
        data.X,
        spec,
        basis,
        make_generator(spec),
        max_iter=10,
        tolerance=1e-8,
        kkt_tolerance=1e-2,
    )
    assert fitted.success
    assert fitted.gradient_norm == pytest.approx(1.0)
    assert fitted.kkt_residual == pytest.approx(1e-6)


def test_candidate_with_excessive_kkt_residual_is_rejected(monkeypatch) -> None:
    """Optimizer success does not override a failed KKT diagnostic."""

    from genriesz.experiments.reference_selection.candidates import (
        CandidateSpec,
        ExperimentBasis,
        fit_candidate_beta,
        make_generator,
    )
    from genriesz.glm import GRRGLM, FitResult

    data = generate_data(n=120, design="low", overlap_scale=1.5, hidden_scale=0.0, seed=92)
    spec = CandidateSpec(loss="SQ", dictionary="linear", penalty_multiplier=1.0)
    basis = ExperimentBasis("linear").fit(data.X)
    beta = np.zeros(basis.n_features, dtype=float)

    def fitted_with_invalid_kkt(self, X, **kwargs):
        return FitResult(
            beta=beta,
            success=True,
            message="",
            n_iter=1,
            status="converged",
            objective_value=0.0,
            gradient_norm=1e-6,
            kkt_residual=0.1,
            clip_binding_rate=float("nan"),
            fit_time=0.0,
        )

    monkeypatch.setattr(GRRGLM, "fit", fitted_with_invalid_kkt)
    fitted = fit_candidate_beta(
        data.X,
        spec,
        basis,
        make_generator(spec),
        max_iter=10,
        tolerance=1e-8,
        kkt_tolerance=1e-2,
    )
    assert not fitted.success
    assert fitted.status == "diagnostic_failure"
    assert fitted.beta is None
