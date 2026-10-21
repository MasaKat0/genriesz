"""Decompose SQ coefficient remainders from the complete saved E-27 results."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from . import env
from .e25 import quantile_interval
from .seeds import CONFIRMATORY_ENTROPY

env.use_submodule_genriesz()
e27 = importlib.import_module(".e27", __package__)

CONSTANTS = ("L", "M", "H", "lambda", "q", "rho", "theta0", "mu", "A0", "D_s", "stable_index", "V")
FOLD_FIELDS = (
    "scale_source_v2",
    "scale_target_v",
    "scale_true_v2_mean",
    "scale_denominator_center",
    "scale_numerator_center",
    "coefficient",
    "da_source",
    "da_remainder",
    "coefficient_expansion_drift",
    "shape_coefficient",
)
TERMS = (
    "coefficient_remainder",
    "source_denominator",
    "target_fluctuation",
    "pilot_centering",
    "pilot_source_fluctuation",
    "retained_denominator_remainder",
    "conditional_linear_remainder",
    "conditional_denominator",
    "conditional_rational_remainder",
)
IDENTITIES = ("coefficient_identity", "conditional_identity", "retained_identity")


def _close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=2e-12, atol=2e-12):
        raise ValueError(f"inconsistent saved E-27 values: {label}")


def _validate_values(raw):
    required = {
        "cell",
        "rep",
        "entropy",
        "b",
        "n",
        "arm",
        "status",
        "combined_scale_coefficient_expansion_drift",
        *CONSTANTS,
        *(f"fold{j}_{name}" for j in range(2) for name in FOLD_FIELDS),
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"missing E-27 columns: {sorted(missing)}")
    numeric_fields = required - {
        "arm",
        "status",
        "fold0_shape_coefficient",
        "fold1_shape_coefficient",
    }
    if any(not pd.api.types.is_numeric_dtype(raw[name]) for name in numeric_fields):
        raise ValueError("E-27 moment and design columns must be numerical")
    numeric = raw.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("E-27 SQ results must contain only finite numerical values")
    if not raw.arm.eq("SQ").all() or not raw.status.eq("ok").all():
        raise ValueError("every selected result must be a successful SQ result")
    for name in ("cell", "rep", "entropy", "n"):
        values = raw[name].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"E-27 {name} must contain integers")
    if not raw.entropy.eq(CONFIRMATORY_ENTROPY).all():
        raise ValueError("E-27 entropy differs from the fixed design")
    if raw.duplicated(["cell", "rep"]).any():
        raise ValueError("duplicate E-27 SQ condition and replication")
    if not raw.cell.isin(range(len(e27.CELLS))).all():
        raise ValueError("unknown E-27 condition")
    if not raw.rep.isin(range(e27.R)).all():
        raise ValueError("unknown E-27 replication")
    for cell, frame in raw.groupby("cell", sort=True):
        b, n = e27.CELLS[int(cell)]
        facts = e27.population(b, n)
        _close(frame.b, b, "b")
        if not frame.n.eq(n).all():
            raise ValueError("E-27 sample size differs from the fixed condition")
        for name in CONSTANTS:
            _close(frame[name], facts[name], name)
        for j in range(2):
            prefix = f"fold{j}_"
            for name in FOLD_FIELDS[:6]:
                if not frame[prefix + name].gt(0).all():
                    raise ValueError(f"E-27 {prefix + name} must be positive")
            shape = np.asarray(
                [json.loads(x) for x in frame[prefix + "shape_coefficient"]], dtype=float
            )
            if (
                shape.shape != (len(frame), facts["H"])
                or not np.isfinite(shape).all()
                or np.any(shape <= 0)
            ):
                raise ValueError("invalid saved E-27 shape coefficients")
            _close(shape.mean(axis=1), 1, "shape integral")
            _close(
                frame[prefix + "scale_denominator_center"],
                np.mean(shape**2, axis=1) / (1 - 2 * b),
                "conditional source second moment",
            )
            _close(
                frame[prefix + "scale_numerator_center"],
                (1 - b) * (shape @ facts["projected_shape"]) / (facts["H"] * (1 - 2 * b)),
                "conditional target first moment",
            )
            coefficient = frame[prefix + "coefficient"].to_numpy()
            _close(
                coefficient,
                frame[prefix + "scale_target_v"] / frame[prefix + "scale_source_v2"],
                "SQ coefficient",
            )
            source = -(1 - b) / facts["A0"] * (frame[prefix + "scale_true_v2_mean"] - facts["A0"])
            _close(frame[prefix + "da_source"], source, "source linear approximation")
            remainder = coefficient - (1 - b) - source
            _close(frame[prefix + "da_remainder"], remainder, "coefficient remainder")
            _close(
                frame[prefix + "coefficient_expansion_drift"],
                facts["q"] * facts["mu"] * remainder,
                "coefficient remainder contribution",
            )
        saved_average = sum(frame[f"fold{j}_coefficient_expansion_drift"] for j in range(2)) / 2
        _close(
            frame.combined_scale_coefficient_expansion_drift,
            saved_average / (n ** (-0.5) + facts["rho"] * facts["q"]),
            "normalized coefficient remainder contribution",
        )


def validate_results(raw):
    """Reject any omission, duplicate, failure or changed scientific condition."""
    _validate_values(raw)
    expected = pd.MultiIndex.from_product(
        [range(len(e27.CELLS)), range(e27.R)], names=["cell", "rep"]
    )
    actual = pd.MultiIndex.from_frame(raw[["cell", "rep"]]).sort_values()
    if not actual.equals(expected):
        raise ValueError("E-27 requires all twelve conditions and replications 0 through 1999")


def load_results(results):
    """Read every SQ record in the saved scientific Parquet shards."""
    directory = Path(results).expanduser().resolve() / "raw"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = sorted(directory.iterdir())
    if not paths or any(
        not path.is_file() or not path.name.startswith("part-") or path.suffix != ".parquet"
        for path in paths
    ):
        raise ValueError("E-27 raw directory must contain only complete Parquet shards")
    raw = ds.dataset(paths, format="parquet").to_table(filter=ds.field("arm") == "SQ").to_pandas()
    validate_results(raw)
    return raw.sort_values(["cell", "rep"]).reset_index(drop=True)


def decompose(raw):
    """Return exact fold contributions in coefficient units, without fitting."""
    rows = []
    for j in range(2):
        prefix = f"fold{j}_"
        frame = raw[["cell", "rep", "b", "n", "q", "rho", "mu", "A0", "V"]].copy()
        frame["fold"] = j
        for key, name in (
            ("S", "scale_source_v2"),
            ("T", "scale_target_v"),
            ("S0", "scale_true_v2_mean"),
            ("barS", "scale_denominator_center"),
            ("barT", "scale_numerator_center"),
            ("coefficient", "coefficient"),
        ):
            frame[key] = raw[prefix + name].to_numpy()
        k = 1 - frame.b
        x0, x, w = frame.S0 - frame.A0, frame.S - frame.A0, frame.S - frame.barS
        frame["normalization"] = frame.n ** (-0.5) + frame.rho * frame.q
        frame["denominator_ratio"] = frame.S / frame.barS
        frame["coefficient_remainder"] = frame.coefficient - k + k / frame.A0 * x0
        frame["source_denominator"] = k * (x0 / frame.A0) * (x / frame.S)
        frame["target_fluctuation"] = (frame["T"] - frame.barT) / frame.S
        frame["pilot_centering"] = (frame.barT - k * frame.barS) / frame.S
        frame["pilot_source_fluctuation"] = (
            -k * ((frame.S - frame.S0) - (frame.barS - frame.A0)) / frame.S
        )
        frame["retained_denominator_remainder"] = frame.coefficient - k + k * x0 / frame.S
        frame["conditional_linear_remainder"] = frame.coefficient - k + k * w / frame.barS
        frame["conditional_denominator"] = k * (w / frame.barS) * (w / frame.S)
        frame["conditional_rational_remainder"] = frame.coefficient - k + k * w / frame.S
        frame["coefficient_identity"] = frame.coefficient_remainder - frame[
            [
                "source_denominator",
                "target_fluctuation",
                "pilot_centering",
                "pilot_source_fluctuation",
            ]
        ].sum(axis=1)
        frame["conditional_identity"] = (
            frame.conditional_linear_remainder
            - frame.conditional_denominator
            - frame.target_fluctuation
            - frame.pilot_centering
        )
        frame["retained_identity"] = (
            frame.retained_denominator_remainder
            - frame.target_fluctuation
            - frame.pilot_centering
            - frame.pilot_source_fluctuation
        )
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ArithmeticError("nonfinite SQ remainder decomposition")
        tolerance = 2e-11 * (1 + frame.coefficient_remainder.abs())
        if frame[list(IDENTITIES)].abs().gt(tolerance, axis=0).any().any():
            raise ArithmeticError("SQ coefficient remainder identity failed")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["cell", "rep", "fold"])


def replication_values(folds):
    """Average the signed, scaled fold terms before taking absolute values."""
    if folds.duplicated(["cell", "rep", "fold"]).any():
        raise ValueError("duplicate coefficient remainder fold")
    grouped = folds.groupby(["cell", "rep"], sort=True)
    if not grouped.fold.agg(lambda x: set(x) == {0, 1}).all():
        raise ValueError("each replication requires both outer folds")
    scaled = folds[["cell", "rep"]].copy()
    multiplier = folds.q * folds.mu / folds.normalization
    for name in TERMS:
        scaled[name] = multiplier * folds[name]
    values = scaled.groupby(["cell", "rep"], sort=True)[list(TERMS)].mean().reset_index()
    return values


def _quantile(row, name, values, probability, digits=2):
    values = np.asarray(values, float)
    if not np.isfinite(values).all() or not len(values):
        raise ValueError(f"invalid numerical values in {name}")
    key = f"{name}_q{round(probability * 10**digits):0{digits}d}"
    row[key] = float(np.quantile(values, probability))
    row[key + "_mc_lo"], row[key + "_mc_hi"] = quantile_interval(values, probability)


def summarise(folds):
    """Summarize absolute fold averages and the observed source denominators."""
    values = replication_values(folds)
    rows = []
    for cell, frame in folds.groupby("cell", sort=True):
        observed = values[values.cell == cell]
        row = {"cell": int(cell), "b": float(frame.b.iloc[0]), "n": int(frame.n.iloc[0])}
        row["replications"], row["folds"] = len(observed), len(frame)
        for name in TERMS:
            for probability in (0.95, 0.99):
                _quantile(row, name + "_abs", observed[name].abs(), probability)
        for probability in (0.95, 0.99, 0.999):
            _quantile(row, "denominator_ratio", frame.denominator_ratio, probability, digits=3)
        absolute = observed.coefficient_remainder.abs()
        for probability, label in ((0.95, "top_05"), (0.99, "top_01")):
            selected = observed[absolute >= np.quantile(absolute, probability)]
            if selected.coefficient_remainder.eq(0).any():
                raise ValueError("source contribution fraction has a zero remainder denominator")
            row["source_fraction_" + label] = float(
                np.median(selected.source_denominator / selected.coefficient_remainder)
            )
            row["replications_" + label] = len(selected)
        for name in IDENTITIES:
            row[name + "_max_abs"] = float(frame[name].abs().max())
        rows.append(row)
    return pd.DataFrame(rows)


def table(summary):
    """Numeric absolute quantiles; source and target columns use probability .95."""
    lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$b$ & $N$ & $Q_{a,.95}$ & $Q_{a,.99}$ & Source & Target"
        r" & $Q_{\mathrm{obs},.95}$ & $Q_{\mathrm{obs},.99}$ \\",
        r"\midrule",
    ]
    fields = (
        "coefficient_remainder_abs_q95",
        "coefficient_remainder_abs_q99",
        "source_denominator_abs_q95",
        "target_fluctuation_abs_q95",
        "retained_denominator_remainder_abs_q95",
        "retained_denominator_remainder_abs_q99",
    )
    for row in summary.itertuples(index=False):
        numbers = " & ".join(f"{getattr(row, name):.4f}" for name in fields)
        lines.append(f"{row.b:.2f} & {row.n:d} & {numbers}" + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def figure(summary):
    """Plot the same absolute quantiles used in the saved numerical table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = (
        ("coefficient_remainder", "Coefficient remainder"),
        ("source_denominator", "Source denominator term"),
        ("target_fluctuation", "Target fluctuation"),
        ("pilot_centering", "Pilot centering"),
        ("pilot_source_fluctuation", "Source pilot fluctuation"),
        ("retained_denominator_remainder", "Observed denominator"),
    )
    fig, axes = plt.subplots(2, len(e27.B_VALUES), figsize=(11, 6), squeeze=False)
    for column, b in enumerate(e27.B_VALUES):
        frame = summary[summary.b == b].sort_values("n")
        for i, probability in enumerate((95, 99)):
            axis = axes[i, column]
            for name, label in labels:
                axis.plot(frame.n, frame[f"{name}_abs_q{probability}"], "o-", label=label)
            axis.set(xscale="log", yscale="log")
            axis.grid(alpha=0.25)
            if not column:
                axis.set_ylabel(f"Absolute {probability / 100:.2f} quantile")
            if i:
                axis.set_xlabel("N")
        axes[0, column].set_title(f"$b={b}$")
    handles, descriptions = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, descriptions, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


@contextmanager
def _atomic_path(path):
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(*, results, output):
    """Write scientific decompositions from saved results, without fitting."""
    results = Path(results).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if output == results or results in output.parents:
        raise ValueError("decomposition output must be outside the saved E-27 results")
    env.check_environment()
    raw = load_results(results)
    folds = decompose(raw)
    summary = summarise(folds)
    output.mkdir(parents=True, exist_ok=False)
    with _atomic_path(output / "remainders.parquet") as temporary:
        folds.to_parquet(temporary, compression="zstd", index=False)
    with _atomic_path(output / "tab_E27_remainders.tex") as temporary:
        temporary.write_text(table(summary), encoding="utf-8")
    plot = figure(summary)
    import matplotlib.pyplot as plt

    try:
        with _atomic_path(output / "fig_E27_remainders.pdf") as temporary:
            plot.savefig(temporary, format="pdf")
    finally:
        plt.close(plot)
    with _atomic_path(output / "summary.csv") as temporary:
        summary.to_csv(temporary, index=False)
    return output / "summary.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(results=args.results, output=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
