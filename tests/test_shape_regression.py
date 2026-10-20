"""Direct sample checks of the histogram, GRR, ridge and score calculations."""

import numpy as np
import pytest

from genriesz.shape_regression import (
    SourceMoments,
    TargetMoments,
    basis,
    fit_ridge,
    fit_scales,
    fit_shape,
)


def data(size=80):
    rng = np.random.default_rng(127)
    u, z = rng.uniform(size=(2, size))
    coefficient = np.array([0.3, 0.2, -0.4, 0.6, 0.1, 0.5])
    y = basis(u, z) @ coefficient + rng.normal(size=size)
    return u, z, y


def test_moments_and_all_score_terms_equal_direct_arrays():
    u, z, y = data()
    tu, tz = u[::-1], z**0.7
    h, b, m, penalty = 4, 0.35, len(y) // 2, 0.17
    source, target = SourceMoments(h, b), TargetMoments(h, b)
    for sl, pilot in ((slice(0, m), True), (slice(m, None), False)):
        source.add(u[sl], z[sl], y[sl], pilot=pilot)
        target.add(tu[sl], tz[sl], pilot=pilot)
    phi, target_phi = basis(u, z), basis(tu, tz)
    np.testing.assert_allclose(source.gram, phi.T @ phi, atol=1e-13)
    np.testing.assert_allclose(source.xy, phi.T @ y, atol=1e-13)
    assert source.n == target.n == source.cell_counts.sum() == target.cell_counts.sum() == len(y)
    assert source.pilot_n == source.scale_n == target.pilot_n == target.scale_n == m
    shape, info = fit_shape(source.pilot_counts, target.pilot_counts)
    scales, scale_info = fit_scales(shape, source.scale_v, source.scale_v2, target.scale_v, m)
    coefficient, ridge_info = fit_ridge(source.gram, source.xy, len(y), penalty)
    direct_coefficient = np.linalg.solve(
        phi.T @ phi / len(y) + np.diag([0, *([penalty] * 5)]), phi.T @ y / len(y)
    )
    np.testing.assert_allclose(coefficient, direct_coefficient, atol=1e-15)
    w = u[m:] ** (-b) * shape[(z[m:] * h).astype(int)]
    tw = tu[m:] ** (-b) * shape[(tz[m:] * h).astype(int)]
    assert scales["SQ"] == pytest.approx(tw.mean() / np.mean(w**2))
    assert scales["UKL"] == pytest.approx(1 / w.mean())
    for scale in scales.values():
        r = scale * u ** (-b) * shape[(z * h).astype(int)]
        score = r * (y - phi @ coefficient)
        first, second = source.score(coefficient, shape, scale)
        assert first == pytest.approx(score.mean(), abs=2e-14)
        assert second == pytest.approx(np.mean(score**2), rel=3e-14)
    center = 0.4
    first, second = target.score(coefficient, center)
    assert first == pytest.approx((target_phi @ coefficient).mean())
    assert second == pytest.approx(np.mean((target_phi @ coefficient - center) ** 2))
    assert info["shape_min"] > 0 and info["shape_integral_error"] < 1e-15
    assert scale_info["SQ_stationarity_scaled"] < 1e-15
    assert scale_info["UKL_stationarity_scaled"] < 1e-15
    assert ridge_info["normal_equation_scaled"] < 1e-15
    assert ridge_info["ridge_min_eigenvalue"] > 0


def test_empty_cells_are_positive_and_scalar_reparameterization_is_exact():
    source, target = np.array([30, 0, 0, 2]), np.array([0, 0, 1, 31])
    shape, _ = fit_shape(source, target)
    q = (target + 1) / (source + 1)
    scale_v, scale_v2, target_v = np.arange(1, 5), np.arange(3, 7), np.arange(4, 8)
    normalized, _ = fit_scales(shape, scale_v, scale_v2, target_v, 9)
    unnormalized, _ = fit_scales(q, scale_v, scale_v2, target_v, 9)
    for arm in normalized:
        np.testing.assert_allclose(
            normalized[arm] * shape, unnormalized[arm] * q, rtol=4 * np.finfo(float).eps
        )
    assert np.all(shape > 0) and shape.mean() == pytest.approx(1)


def test_unpenalized_intercept_and_rank_deficient_design():
    phi = basis(np.full(7, 0.3), np.full(7, 0.7))
    fit, info = fit_ridge(phi.T @ phi, phi.T @ np.full(7, 4.25), 7, 0.2)
    np.testing.assert_allclose(fit, [4.25, 0, 0, 0, 0, 0], atol=3e-14)
    assert info["ridge_min_eigenvalue"] > 0


@pytest.mark.parametrize("bad", [0, 1, np.nan, -0.1])
def test_noninterior_covariates_raise(bad):
    with pytest.raises(ValueError):
        basis(np.array([bad]), np.array([0.5]))


def test_invalid_fit_inputs_raise():
    with pytest.raises(ValueError):
        fit_shape([1, -1], [0, 0])
    with pytest.raises(ValueError):
        fit_scales([1, 0], [1, 1], [1, 1], [1, 1], 2)


def test_unknown_shape_function_components_and_signed_drift_decompositions():
    import sys
    from pathlib import Path

    from scipy import integrate

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from grrexp import e27

    facts = e27.population(0.35, 2048)
    shape = np.array([0.4, 0.7, 1.2, 1.7])
    coefficient = (1 - facts["q"]) * e27.BETA0 + np.array([0.01, 0.02, -0.03, 0.01, 0.02, -0.01])
    scale = facts["k"] + 0.03
    true_v2_mean = 1.07 * facts["A0"]
    d = e27.fold_diagnostics(shape, scale, coefficient, facts, true_v2_mean)
    u_integral = integrate.quad(lambda u: u ** (-2 * facts["b"]), 0, 1, epsabs=1e-11, epsrel=1e-11)[
        0
    ]
    shape_error_integral = sum(
        integrate.quad(
            lambda z, cell=cell: (shape[cell] - 3 * (1 + z) ** 2 / 7) ** 2,
            cell / 4,
            (cell + 1) / 4,
            epsabs=1e-13,
        )[0]
        for cell in range(4)
    )
    true_shape_integral = integrate.quad(lambda z: (3 * (1 + z) ** 2 / 7) ** 2, 0, 1, epsabs=1e-13)[
        0
    ]
    expected_cross_norm = abs(scale - facts["k"]) * np.sqrt(u_integral * shape_error_integral)
    assert d["cross_l2"] == pytest.approx(expected_cross_norm, rel=2e-12)
    assert d["coefficient_error_l2"] == pytest.approx(
        abs(scale - facts["k"]) * np.sqrt(u_integral * true_shape_integral), rel=2e-12
    )
    assert d["shape_term_l2"] == pytest.approx(
        facts["k"] * np.sqrt(u_integral * shape_error_integral), rel=2e-12
    )
    delta = coefficient - e27.BETA0
    # Three-point Gaussian integration is exact for the squared quadratic regression error.
    nodes, weights = np.polynomial.legendre.leggauss(3)
    u, z = np.meshgrid((nodes + 1) / 2, (nodes + 1) / 2, indexing="ij")
    regression_norm2 = (
        np.sum(
            (basis(u.ravel(), z.ravel()) @ delta).reshape(3, 3) ** 2
            * weights[:, None]
            * weights[None, :]
        )
        / 4
    )
    assert d["cross_norm_product"] == pytest.approx(
        np.sqrt(facts["n"] * regression_norm2) * expected_cross_norm, rel=2e-12
    )
    ridge_error = coefficient - (1 - facts["q"]) * e27.BETA0
    j = np.zeros(6)
    for cell in range(4):
        for coordinate in range(6):
            j[coordinate] += integrate.dblquad(
                lambda z, x, cell=cell, coordinate=coordinate: (
                    (shape[cell] - 3 * (1 + z) ** 2 / 7)
                    * basis(np.array([x ** (1 / facts["k"])]), np.array([z]))[0, coordinate]
                    / facts["k"]
                ),
                0,
                1,
                lambda _, cell=cell: cell / 4,
                lambda _, cell=cell: (cell + 1) / 4,
                epsabs=1e-11,
            )[0]
    assert d["shape_bias"] == pytest.approx(facts["k"] * facts["q"] * j @ e27.BETA0, abs=1e-11)
    assert d["shape_sampling"] == pytest.approx(-facts["k"] * j @ ridge_error, abs=1e-11)
    assert d["cross_bias"] == pytest.approx(
        (scale - facts["k"]) * facts["q"] * j @ e27.BETA0, abs=1e-11
    )
    assert d["cross_sampling"] == pytest.approx(-(scale - facts["k"]) * j @ ridge_error, abs=1e-11)
    assert d["pilot_shape_drift"] == pytest.approx(d["shape_bias"] + d["shape_sampling"])
    assert d["coefficient_shape_cross"] == pytest.approx(d["cross_bias"] + d["cross_sampling"])
    assert abs(d["shape_decomposition_identity_error"]) < 1e-14
    assert abs(d["cross_decomposition_identity_error"]) < 1e-14
    expected_da = -facts["k"] / facts["A0"] * (true_v2_mean - facts["A0"])
    assert d["da_source"] == pytest.approx(expected_da)
    assert d["da_remainder"] == pytest.approx(scale - facts["k"] - expected_da)
    assert d["source_stable_leading_drift"] == pytest.approx(facts["q"] * facts["mu"] * expected_da)
    assert d["leading_drift"] == pytest.approx(
        d["source_stable_leading_drift"] + d["coefficient_expansion_drift"]
    )
    assert abs(d["coefficient_decomposition_identity_error"]) < 1e-14


def test_true_source_scale_second_moment_uses_every_scale_observation_only():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from grrexp import e27

    cell, rep, fold = 0, 12, 1
    facts = e27.population(*e27.CELLS[cell])
    source, target, oracle, _, _ = e27.collect_fold(cell, rep, fold, chunk_size=57)
    u = e27.interior_uniform(e27.generator(cell, rep, 0, fold), facts["L"])
    z = e27.interior_uniform(e27.generator(cell, rep, 1, fold), facts["L"])
    expected = np.sum((u[facts["M"] :] ** (-facts["b"]) * 3 * (1 + z[facts["M"] :]) ** 2 / 7) ** 2)
    assert len(oracle) == 3
    assert oracle[2] == pytest.approx(expected, rel=3e-15)
    assert source.scale_n == target.scale_n == facts["M"]
    assert "true_shape" not in source.__dict__ and "true_shape" not in target.__dict__
    collections = [e27.collect_fold(cell, rep, j, chunk_size=57) for j in range(2)]
    sources, targets, oracles = ([item[index] for item in collections] for index in range(3))
    fits = [e27.nuisance(sources[j], targets[j], facts["lambda"]) for j in range(2)]
    rows = e27.score_records(sources, targets, fits, oracles, facts)
    for row in rows.values():
        for evaluation in range(2):
            assert row[f"fold{evaluation}_scale_true_v2_mean"] == pytest.approx(
                oracles[1 - evaluation][2] / facts["M"]
            )
        for aggregate, first, second in (
            ("pilot_shape_drift", "shape_bias", "shape_sampling"),
            ("coefficient_shape_cross", "cross_bias", "cross_sampling"),
            ("leading_drift", "source_stable_leading_drift", "coefficient_expansion_drift"),
        ):
            assert row[aggregate] == pytest.approx(row[first] + row[second], abs=1e-14)
            for name in (first, second):
                assert row["root_n_" + name] == pytest.approx(np.sqrt(facts["n"]) * row[name])
                assert row["stable_" + name] == pytest.approx(
                    row[name] / (facts["rho"] * facts["q"])
                )
                assert row["combined_scale_" + name] == pytest.approx(
                    row[name] / (facts["n"] ** (-0.5) + facts["rho"] * facts["q"])
                )
        assert abs(row["finite_sample_identity_error"]) < 1e-14


def test_added_diagnostics_remain_numerical_raw_and_summary_columns():
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from grrexp import e27

    added = (
        "shape_bias",
        "shape_sampling",
        "cross_bias",
        "cross_sampling",
        "source_stable_leading_drift",
        "coefficient_expansion_drift",
    )
    frames = []
    for cell, (b, n) in enumerate(e27.CELLS):
        facts = e27.population(b, n)
        for arm in e27.ARM_LABELS:
            frame = pd.DataFrame(
                {
                    "cell": cell,
                    "rep": np.arange(e27.R),
                    "n": n,
                    "arm": arm,
                    "status": "ok",
                    "estimate": facts["theta0"],
                    "se": 0.1,
                    "oracle_estimate": facts["theta0"],
                    "oracle_se": 0.1,
                    "oracle_known_variance_se": np.sqrt(facts["V"] / n),
                }
            )
            for index, name in enumerate(added, start=1):
                frame[name] = index / 10
                frame["root_n_" + name] = np.sqrt(n) * frame[name]
                frame["stable_" + name] = frame[name] / (facts["rho"] * facts["q"])
                frame["combined_scale_" + name] = frame[name] / (
                    n ** (-0.5) + facts["rho"] * facts["q"]
                )
            for fold in range(2):
                for name in (
                    "cross_l2",
                    "cross_norm_product",
                    "coefficient_error_l2",
                    "shape_term_l2",
                    "da_source",
                    "da_remainder",
                    "da_source_over_rho",
                    "da_remainder_over_rho",
                ):
                    frame[f"fold{fold}_{name}"] = 0.25
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    summary = e27.summarise(raw)
    assert len(summary) == len(e27.CELLS) * len(e27.ARM_LABELS)
    for index, name in enumerate(added, start=1):
        assert summary[name + "_q50"].eq(index / 10).all()
        for prefix in ("root_n_", "stable_", "combined_scale_"):
            assert np.isfinite(summary[prefix + name + "_q50"]).all()
    for fold in range(2):
        for name in (
            "cross_l2",
            "cross_norm_product",
            "coefficient_error_l2",
            "shape_term_l2",
            "da_source",
            "da_remainder",
            "da_source_over_rho",
            "da_remainder_over_rho",
        ):
            assert summary[f"fold{fold}_{name}_q50"].eq(0.25).all()
