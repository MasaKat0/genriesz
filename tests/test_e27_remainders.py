"""Independent scalar decompositions and strict saved-result validation."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from grrexp import e27
from grrexp import e27_remainders as analysis
from grrexp.e25 import quantile_interval
from grrexp.seeds import CONFIRMATORY_ENTROPY


def saved_row(rep=0):
    """A few exact moment inputs for ordinary algebra and validation tests."""
    b, n = e27.CELLS[0]
    facts = e27.population(b, n)
    row = {
        "cell": 0,
        "rep": rep,
        "b": b,
        "n": n,
        "arm": "SQ",
        "status": "ok",
        "entropy": CONFIRMATORY_ENTROPY,
        **{key: facts[key] for key in analysis.CONSTANTS},
    }
    for j in range(2):
        prefix = f"fold{j}_"
        shape = facts["projected_shape"] + np.linspace(-0.03, 0.03, facts["H"]) * (j + 1)
        shape /= shape.mean()
        bar_s = float(np.mean(shape**2) / (1 - 2 * b))
        bar_t = float((1 - b) * np.mean(shape * facts["projected_shape"]) / (1 - 2 * b))
        s0 = facts["A0"] * (0.9 + 0.04 * rep + 0.03 * j)
        source = bar_s * (0.8 + 0.07 * rep + 0.15 * j)
        target = bar_t * (1.15 - 0.04 * rep + 0.1 * j)
        coefficient = target / source
        leading = -(1 - b) / facts["A0"] * (s0 - facts["A0"])
        remainder = coefficient - (1 - b) - leading
        row.update(
            {
                prefix + "scale_source_v2": source,
                prefix + "scale_target_v": target,
                prefix + "scale_true_v2_mean": s0,
                prefix + "scale_denominator_center": bar_s,
                prefix + "scale_numerator_center": bar_t,
                prefix + "coefficient": coefficient,
                prefix + "da_source": leading,
                prefix + "da_remainder": remainder,
                prefix + "coefficient_expansion_drift": facts["q"] * facts["mu"] * remainder,
                prefix + "shape_coefficient": json.dumps(shape.tolist()),
            }
        )
    row["combined_scale_coefficient_expansion_drift"] = (
        sum(row[f"fold{j}_coefficient_expansion_drift"] for j in range(2))
        / 2
        / (n ** (-0.5) + facts["rho"] * facts["q"])
    )
    return row


def scalar_decomposition(record, fold):
    """Evaluate the mathematical identities directly, one fold at a time."""
    p = f"fold{fold}_"
    s, t, s0, abar, bbar = (
        record[p + field]
        for field in (
            "scale_source_v2",
            "scale_target_v",
            "scale_true_v2_mean",
            "scale_denominator_center",
            "scale_numerator_center",
        )
    )
    k, a0 = 1 - record["b"], record["A0"]
    coefficient = t / s
    return {
        "coefficient_remainder": coefficient + k * s0 / a0 - 2 * k,
        "source_denominator": k * (s0 - a0) * (s - a0) / (a0 * s),
        "target_fluctuation": (t - bbar) / s,
        "pilot_centering": (bbar - k * abar) / s,
        "pilot_source_fluctuation": -k * (s - s0 - abar + a0) / s,
        "retained_denominator_remainder": (t - k * a0 - k * (s - s0)) / s,
        "conditional_linear_remainder": coefficient - k + k * (s - abar) / abar,
        "conditional_denominator": k * (s - abar) ** 2 / (abar * s),
        "conditional_rational_remainder": (t - k * abar) / s,
    }


def test_vectorized_decomposition_against_independent_scalar_equations():
    records = [saved_row(rep) for rep in range(4)]
    result = analysis.decompose(pd.DataFrame(records))
    assert len(result) == 8
    for record in records:
        for fold in range(2):
            row = result[(result.rep == record["rep"]) & (result.fold == fold)].iloc[0]
            direct = scalar_decomposition(record, fold)
            for name, expected in direct.items():
                assert row[name] == pytest.approx(expected, abs=8e-16)
            assert row.conditional_denominator >= 0
            assert row.normalization == pytest.approx(
                record["n"] ** (-0.5) + record["rho"] * record["q"]
            )
    assert result[list(analysis.IDENTITIES)].abs().to_numpy().max() < 2e-15


def test_large_source_moment_has_bounded_scale_error_and_positive_inverse_remainder():
    record = saved_row()
    k, a0 = 1 - record["b"], record["A0"]
    for fold in range(2):
        p = f"fold{fold}_"
        record[p + "scale_source_v2"] = record[p + "scale_true_v2_mean"] = 100 * a0
        record[p + "scale_denominator_center"] = a0
        record[p + "scale_numerator_center"] = record[p + "scale_target_v"] = k * a0
        record[p + "coefficient"] = k / 100
    result = analysis.decompose(pd.DataFrame([record]))
    assert (result.coefficient - k > -k).all()
    assert (result.coefficient_remainder > 90 * k).all()
    np.testing.assert_allclose(result.coefficient_remainder, result.source_denominator)
    np.testing.assert_allclose(result.conditional_denominator, result.source_denominator)
    np.testing.assert_allclose(result.conditional_rational_remainder, 0, atol=2e-16)


def test_signed_folds_are_averaged_before_absolute_quantiles():
    record = saved_row()
    k, a0 = 1 - record["b"], record["A0"]
    for fold, desired in enumerate((0.1, -0.1)):
        p = f"fold{fold}_"
        coefficient = k - k * (record[p + "scale_true_v2_mean"] - a0) / a0 + desired
        record[p + "coefficient"] = coefficient
        record[p + "scale_target_v"] = record[p + "scale_source_v2"] * coefficient
    folds = analysis.decompose(pd.DataFrame([record]))
    averaged = analysis.replication_values(folds)
    assert averaged.coefficient_remainder.iloc[0] == pytest.approx(0, abs=8e-16)
    assert np.mean(np.abs(folds.coefficient_remainder)) > 0.09


def test_summary_quantiles_intervals_and_tail_fractions_use_the_same_replications():
    records = [saved_row(rep) for rep in range(4)]
    folds = analysis.decompose(pd.DataFrame(records))
    summary = analysis.summarise(folds).iloc[0]
    expected = {name: [] for name in analysis.TERMS}
    for record in records:
        multiplier = (
            record["q"]
            * record["mu"]
            / (record["n"] ** (-0.5) + record["rho"] * record["q"])
        )
        scalar = [scalar_decomposition(record, fold) for fold in range(2)]
        for name in analysis.TERMS:
            expected[name].append(multiplier * (scalar[0][name] + scalar[1][name]) / 2)
    for name, values in expected.items():
        for q in (0.95, 0.99):
            key = f"{name}_abs_q{round(100 * q)}"
            assert summary[key] == pytest.approx(np.quantile(np.abs(values), q))
            lo, hi = quantile_interval(np.abs(values), q)
            assert summary[key + "_mc_lo"] == pytest.approx(lo)
            assert summary[key + "_mc_hi"] == pytest.approx(hi)
    for q in (0.95, 0.99, 0.999):
        assert summary[f"denominator_ratio_q{round(1000 * q)}"] == pytest.approx(
            np.quantile(folds.denominator_ratio, q)
        )
    old, source = (
        np.asarray(expected["coefficient_remainder"]),
        np.asarray(expected["source_denominator"]),
    )
    for q, label in ((0.95, "top_05"), (0.99, "top_01")):
        selected = np.abs(old) >= np.quantile(np.abs(old), q)
        assert summary["source_fraction_" + label] == pytest.approx(
            np.median(source[selected] / old[selected])
        )
        assert summary["replications_" + label] == selected.sum()


def test_missing_or_duplicate_outer_fold_is_rejected():
    folds = analysis.decompose(pd.DataFrame([saved_row()]))
    with pytest.raises(ValueError, match="both outer folds"):
        analysis.replication_values(folds.iloc[:1])
    with pytest.raises(ValueError, match="duplicate.*fold"):
        analysis.replication_values(pd.concat([folds, folds.iloc[:1]]))


def test_valid_moments_do_not_authorize_an_incomplete_design():
    raw = pd.DataFrame([saved_row()])
    analysis._validate_values(raw)
    with pytest.raises(ValueError, match="all twelve conditions"):
        analysis.validate_results(raw)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("extra_numeric", np.inf, "finite"),
        ("entropy", 0, "entropy"),
        ("n", 999, "sample size"),
        ("rho", 0.123, "rho"),
        ("fold0_scale_source_v2", 0, "positive"),
        ("fold1_scale_target_v", -1, "positive"),
        ("fold0_coefficient", 2, "SQ coefficient"),
        ("fold0_da_source", 10, "linear approximation"),
        ("fold1_da_remainder", 10, "coefficient remainder"),
        ("fold1_coefficient_expansion_drift", 10, "remainder contribution"),
        ("arm", "UKL", "successful SQ"),
        ("status", "failed", "successful SQ"),
        ("rep", 0.5, "integers"),
    ],
)
def test_invalid_saved_values_are_rejected(column, value, message):
    raw = pd.DataFrame([saved_row()])
    raw[column] = value
    with pytest.raises(ValueError, match=message):
        analysis._validate_values(raw)


def test_duplicate_or_missing_saved_keys_are_rejected():
    raw = pd.DataFrame([saved_row()])
    with pytest.raises(ValueError, match="duplicate"):
        analysis.validate_results(pd.concat([raw, raw], ignore_index=True))
    with pytest.raises(ValueError, match="missing.*q"):
        analysis.validate_results(raw.drop(columns="q"))


@pytest.mark.parametrize("shape", [[-1, 1, 1, 3], [1, 1], [2, 2, 2, 2]])
def test_invalid_shape_coefficients_and_normalization_are_rejected(shape):
    raw = pd.DataFrame([saved_row()])
    raw["fold0_shape_coefficient"] = json.dumps(shape)
    with pytest.raises(ValueError, match="shape"):
        analysis._validate_values(raw)


def test_changed_conditional_shape_moment_is_rejected():
    raw = pd.DataFrame([saved_row()])
    raw["fold0_scale_denominator_center"] *= 1.01
    with pytest.raises(ValueError, match="conditional source second moment"):
        analysis._validate_values(raw)


def test_loader_reads_saved_parquet_and_rejects_an_omitted_programme(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    pd.DataFrame([saved_row()]).to_parquet(directory / "part-000001.parquet", index=False)
    with pytest.raises(ValueError, match="all twelve conditions"):
        analysis.load_results(tmp_path)


def test_partial_shard_is_not_silently_ignored(tmp_path):
    directory = tmp_path / "raw"
    directory.mkdir()
    (directory / ".part-000001.parquet").write_text("unfinished", encoding="utf-8")
    with pytest.raises(ValueError, match="complete Parquet"):
        analysis.load_results(tmp_path)


def test_existing_output_is_rejected_before_reading_or_changing_any_file(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "existing.txt"
    sentinel.write_text("preserve this scientific output", encoding="utf-8")
    with pytest.raises(FileExistsError):
        analysis.run(results=tmp_path / "missing", output=output)
    assert sentinel.read_text(encoding="utf-8") == "preserve this scientific output"
    assert list(output.iterdir()) == [sentinel]


def test_original_results_cannot_be_used_as_the_decomposition_destination(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    output = results / "decomposition"
    with pytest.raises(ValueError, match="outside the saved"):
        analysis.run(results=results, output=output)
    assert not output.exists()


def test_cli_exposes_saved_result_paths_only():
    with pytest.raises(SystemExit):
        analysis.main(["--results", "unused", "--output", "unused-output", "--R", "1"])


def test_table_has_short_numeric_columns_and_no_float_spacing_adjustments():
    summary = analysis.summarise(analysis.decompose(pd.DataFrame([saved_row()])))
    text = analysis.table(summary)
    assert r"\begin{tabular}{rrrrrrrr}" in text
    assert r"\vspace" not in text and "caption" not in text
    assert len(next(line for line in text.splitlines() if line.startswith("0.28")).split("&")) == 8


def test_decomposition_preserves_the_saved_combined_scale_statistic():
    records = [saved_row(rep) for rep in range(4)]
    raw = pd.DataFrame(records)
    analysis._validate_values(raw)
    values = analysis.replication_values(analysis.decompose(raw))
    np.testing.assert_allclose(
        values.coefficient_remainder,
        raw.combined_scale_coefficient_expansion_drift,
        rtol=2e-14,
        atol=2e-14,
    )
    raw.loc[0, "combined_scale_coefficient_expansion_drift"] *= 1.01
    with pytest.raises(ValueError, match="normalized coefficient remainder"):
        analysis._validate_values(raw)


def test_failed_atomic_write_leaves_no_partial_final_file(tmp_path):
    destination = tmp_path / "summary.csv"
    with pytest.raises(OSError, match="simulated storage failure"):
        with analysis._atomic_path(destination) as temporary:
            temporary.write_text("incomplete", encoding="utf-8")
            raise OSError("simulated storage failure")
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_summary_is_published_only_after_companions_succeed(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    raw = pd.DataFrame([saved_row(rep) for rep in range(4)])
    monkeypatch.setattr(analysis.env, "check_environment", lambda: {})
    monkeypatch.setattr(analysis, "load_results", lambda _: raw)
    monkeypatch.setattr(analysis, "figure", lambda _: Figure())

    def fail_figure(self, path, **kwargs):
        Path(path).write_bytes(b"partial PDF")
        raise OSError("simulated figure failure")

    monkeypatch.setattr(Figure, "savefig", fail_figure)
    output = tmp_path / "output"
    with pytest.raises(OSError, match="simulated figure failure"):
        analysis.run(results=tmp_path / "old", output=output)
    assert not (output / "summary.csv").exists()
    assert not (output / "fig_E27_remainders.pdf").exists()
    assert sorted(p.name for p in output.iterdir()) == [
        "remainders.parquet", "tab_E27_remainders.tex"
    ]
    assert len(pd.read_parquet(output / "remainders.parquet")) == 8
