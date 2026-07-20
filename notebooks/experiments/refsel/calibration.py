"""Calibration of the hidden-direction scale.

Section 6.3 of ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``.

Experiment E4 needs to sweep the bias-to-standard-error ratio

    t = sqrt(n_eval) |B| / sqrt(V),

because that is the parameter indexing the bounded-normal-mean problem. The
scale ``b`` of the hidden direction is calibrated so that a *fixed benchmark
specification* attains a target ``t``. Fixing the benchmark keeps the definition
of ``b`` independent of the selection rule; the ``t`` actually realized by the
selected candidate then differs, and measuring that difference is the point of
the experiment.

The resulting table is committed as ``calibration.json`` so that publication
runs are deterministic and never re-calibrate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audit import audit_library, integration_sample, scenario_key
from .candidates import BENCHMARK_SPEC, FoldLibrary, fit_outcome
from .dgp import Design, generate_data, make_fold_roles, stable_seed

CALIBRATION_PATH = Path(__file__).with_name("calibration.json")

#: Hidden-direction scales evaluated when building the calibration curve, and the
#: cost settings, per design. The committed ``calibration.json`` is reproduced by
#: ``build_calibration(calibration_specifications())`` with these values; keeping
#: them here rather than in the caller is what makes the provenance of the table
#: recoverable from the code alone.
CALIBRATION_SETTINGS: dict[str, dict[str, object]] = {
    "low": {
        "b_grid": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0),
        "replications": 40,
        "integration_size": 100_000,
    },
    "high": {
        "b_grid": (0.0, 0.3, 0.6, 1.0, 1.5, 2.0),
        "replications": 15,
        "integration_size": 50_000,
    },
}

#: Retained for callers that want a single grid.
DEFAULT_B_GRID: tuple[float, ...] = CALIBRATION_SETTINGS["low"]["b_grid"]  # type: ignore[assignment]


def calibration_key(design: Design, sample_size: int, overlap_scale: float, target_t: float) -> str:
    return f"{design}|n={sample_size}|s={overlap_scale:g}|t={target_t:g}"


def benchmark_curve(
    *,
    design: Design,
    sample_size: int,
    overlap_scale: float,
    b_grid: tuple[float, ...] = DEFAULT_B_GRID,
    replications: int = 40,
    n_folds: int = 5,
    integration_size: int = 100_000,
    base_seed: int = 20260720,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
) -> dict[float, float]:
    """Return ``{b: t(b)}`` for the benchmark specification.

    Only the first fold rotation is used, since ``t`` is a property of a single
    training/evaluation split.
    """

    curve: dict[float, float] = {}
    attempts: dict[float, tuple[int, int]] = {}
    for b in b_grid:
        seed = stable_seed(base_seed, "calibration", design, sample_size, overlap_scale, b)
        shared = dict(
            design=design,
            overlap_scale=overlap_scale,
            hidden_scale=b,
            size=integration_size,
            seed=stable_seed(seed, "integration"),
        )
        integration = integration_sample(**shared)  # type: ignore[arg-type]
        key = scenario_key(**shared)  # type: ignore[arg-type]
        biases: list[float] = []
        variances: list[float] = []
        n_evaluation = 0
        for repetition in range(replications):
            data = generate_data(
                n=sample_size,
                design=design,
                overlap_scale=overlap_scale,
                hidden_scale=b,
                seed=stable_seed(seed, repetition, "sample"),
            )
            roles = make_fold_roles(
                sample_size, n_folds, stable_seed(seed, repetition, "folds")
            )[0]
            n_evaluation = len(roles.evaluation)
            X_train = data.X[roles.training]
            outcome = fit_outcome(
                X_train,
                data.outcomes()[roles.training],
                design=design,
                seed=stable_seed(seed, repetition, "outcome"),
            )
            library = FoldLibrary(
                X_train, (BENCHMARK_SPEC,), max_iter=max_iter, tolerance=tolerance
            )
            audit = audit_library(
                library, outcome, integration, key=key, n_evaluation=n_evaluation
            )
            if np.isfinite(audit.bias[0]) and np.isfinite(audit.variance[0]):
                biases.append(float(audit.bias[0]))
                variances.append(float(audit.variance[0]))
        attempts[b] = (len(biases), replications)
        if len(biases) < replications:
            # Conditioning t(b) on the replications that happened to fit would
            # calibrate a different data-generating process than the one the
            # publication grid then runs, and the failure rate grows with b.
            raise RuntimeError(
                f"The benchmark specification fit on only {len(biases)} of "
                f"{replications} replications at b={b}. The calibration curve "
                "would be conditional on success."
            )
        curve[b] = float(
            np.sqrt(n_evaluation) * abs(np.mean(biases)) / np.sqrt(np.mean(variances))
        )
    return curve


def invert_curve(curve: dict[float, float], target_t: float) -> float:
    """Return the ``b`` whose calibrated ``t`` matches ``target_t``.

    The curve is monotone in ``|b|`` by construction because the hidden
    direction enters the treatment index and the untreated regression with the
    same sign, so the product bias does not cancel.
    """

    if target_t <= 0.0:
        return 0.0
    b_values = np.asarray(sorted(curve), dtype=float)
    t_values = np.asarray([curve[b] for b in sorted(curve)], dtype=float)
    order = np.argsort(t_values)
    t_sorted = t_values[order]
    b_sorted = b_values[order]
    if target_t > float(t_sorted[-1]):
        raise ValueError(
            f"target t={target_t} exceeds the calibrated maximum {t_sorted[-1]:.3f}; "
            "widen DEFAULT_B_GRID."
        )
    return float(np.interp(target_t, t_sorted, b_sorted))


def build_calibration(
    specifications: tuple[tuple[Design, int, float, tuple[float, ...]], ...],
    *,
    settings: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build the full calibration table for a set of ``(design, n, s, targets)``.

    Cost settings are per design, because the high-dimensional benchmark is far
    more expensive per replication. The payload records the grid and the counts
    actually used, so the committed table can be regenerated and checked against
    the code that produced it.
    """

    settings = settings or CALIBRATION_SETTINGS
    entries: dict[str, float] = {}
    curves: dict[str, dict[str, float]] = {}
    for design, sample_size, overlap_scale, targets in specifications:
        chosen = settings[design]
        b_grid = tuple(float(b) for b in chosen["b_grid"])  # type: ignore[index]
        curve = benchmark_curve(
            design=design,
            sample_size=sample_size,
            overlap_scale=overlap_scale,
            b_grid=b_grid,
            replications=int(chosen["replications"]),  # type: ignore[index]
            integration_size=int(chosen["integration_size"]),  # type: ignore[index]
        )
        curves[f"{design}|n={sample_size}|s={overlap_scale:g}"] = {
            f"{b:g}": value for b, value in curve.items()
        }
        for target in targets:
            entries[calibration_key(design, sample_size, overlap_scale, target)] = (
                invert_curve(curve, target)
            )
    return {
        "benchmark": BENCHMARK_SPEC.label,
        "settings": {
            design: {
                "b_grid": [float(b) for b in chosen["b_grid"]],  # type: ignore[index]
                "replications": int(chosen["replications"]),  # type: ignore[index]
                "integration_size": int(chosen["integration_size"]),  # type: ignore[index]
            }
            for design, chosen in settings.items()
        },
        "curves": curves,
        "hidden_scale": entries,
    }


def load_calibration(path: Path | None = None) -> dict[str, float]:
    """Load the committed hidden-scale table."""

    target = path or CALIBRATION_PATH
    if not target.exists():
        return {}
    with target.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {key: float(value) for key, value in payload.get("hidden_scale", {}).items()}


def write_calibration(payload: dict[str, object], path: Path | None = None) -> Path:
    target = path or CALIBRATION_PATH
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target
