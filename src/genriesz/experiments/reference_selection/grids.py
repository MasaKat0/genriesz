"""Publication grids.

Section 13.2 of ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``.

Grid A sweeps overlap at zero calibrated bias, grid B sweeps the calibrated
bias-to-standard-error ratio at intermediate overlap, and grid C repeats the
exercise in the high-dimensional design. Overlap and bias are not crossed
because they move different quantities and the interaction carries no claim.

Every reference, every allowance scale, and every selection rule is evaluated
inside each job, so none of them multiplies the job count.
"""

from __future__ import annotations

from .calibration import calibration_key, load_calibration
from .candidates import CandidateSpec
from .runner import ExperimentConfig, Numerics, Scenario

#: Grid A: overlap sweep with a correctly approximable design.
GRID_A_SAMPLE_SIZES: tuple[int, ...] = (1000, 3000)
GRID_A_OVERLAP_SCALES: tuple[float, ...] = (0.5, 1.5, 2.5)

#: Grid B: calibrated bias sweep at intermediate overlap. The centrepiece of E4.
GRID_B_SAMPLE_SIZES: tuple[int, ...] = (1000, 3000)
GRID_B_OVERLAP_SCALE: float = 1.5
GRID_B_TARGETS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

#: Grid C: high-dimensional design.
GRID_C_SAMPLE_SIZE: int = 3000
GRID_C_OVERLAP_SCALES: tuple[float, ...] = (0.75, 2.0)
GRID_C_TARGETS: tuple[float, ...] = (0.0, 1.0)


def calibration_specifications() -> tuple[tuple[str, int, float, tuple[float, ...]], ...]:
    """The ``(design, n, s, targets)`` cells that need a calibrated hidden scale."""

    specs: list[tuple[str, int, float, tuple[float, ...]]] = [
        ("low", n, GRID_B_OVERLAP_SCALE, GRID_B_TARGETS) for n in GRID_B_SAMPLE_SIZES
    ]
    specs.extend(
        ("high", GRID_C_SAMPLE_SIZE, s, tuple(t for t in GRID_C_TARGETS if t > 0.0))
        for s in GRID_C_OVERLAP_SCALES
    )
    return tuple(specs)


def _hidden_scale(
    calibration: dict[str, float],
    design: str,
    sample_size: int,
    overlap_scale: float,
    target_t: float,
) -> float:
    if target_t == 0.0:
        return 0.0
    key = calibration_key(design, sample_size, overlap_scale, target_t)  # type: ignore[arg-type]
    if key not in calibration:
        raise KeyError(
            f"No calibrated hidden scale for {key}. Run refsel.calibration.build_calibration "
            "and commit the result to refsel/calibration.json."
        )
    return calibration[key]


def publication_grid(calibration: dict[str, float] | None = None) -> tuple[Scenario, ...]:
    """Return every scenario of the publication design."""

    table = load_calibration() if calibration is None else calibration
    scenarios: list[Scenario] = []
    for sample_size in GRID_A_SAMPLE_SIZES:
        for overlap_scale in GRID_A_OVERLAP_SCALES:
            scenarios.append(
                Scenario(
                    grid="A",
                    design="low",
                    sample_size=sample_size,
                    overlap_scale=overlap_scale,
                    target_t=0.0,
                    hidden_scale=0.0,
                )
            )
    for sample_size in GRID_B_SAMPLE_SIZES:
        for target in GRID_B_TARGETS:
            scenarios.append(
                Scenario(
                    grid="B",
                    design="low",
                    sample_size=sample_size,
                    overlap_scale=GRID_B_OVERLAP_SCALE,
                    target_t=target,
                    hidden_scale=_hidden_scale(
                        table, "low", sample_size, GRID_B_OVERLAP_SCALE, target
                    ),
                )
            )
    for overlap_scale in GRID_C_OVERLAP_SCALES:
        for target in GRID_C_TARGETS:
            scenarios.append(
                Scenario(
                    grid="C",
                    design="high",
                    sample_size=GRID_C_SAMPLE_SIZE,
                    overlap_scale=overlap_scale,
                    target_t=target,
                    hidden_scale=_hidden_scale(
                        table, "high", GRID_C_SAMPLE_SIZE, overlap_scale, target
                    ),
                )
            )
    return tuple(scenarios)


def experiment_config(
    *,
    name: str = "reference_selection",
    scenarios: tuple[Scenario, ...] | None = None,
    numerics: Numerics | None = None,
    batch_size: int = 20,
    max_workers: int | None = None,
    replications_by_grid: dict[str, int] | None = None,
    candidate_specs: tuple[CandidateSpec, ...] | None = None,
) -> ExperimentConfig:
    """Build the publication configuration or a test-specific configuration.

    The manuscript notebook calls this function without overriding the scenario
    grid or replication counts. Unit tests may supply a small explicit scenario
    set and matching replication counts; that does not define an alternative
    experiment mode.
    """

    selected_scenarios = publication_grid() if scenarios is None else scenarios
    kwargs: dict[str, object] = {}
    if replications_by_grid is not None:
        kwargs["replications_by_grid"] = replications_by_grid
    if candidate_specs is not None:
        kwargs["candidate_specs"] = candidate_specs
    return ExperimentConfig(
        name=name,
        scenarios=selected_scenarios,
        numerics=numerics or Numerics(),
        batch_size=batch_size,
        max_workers=max_workers,
        **kwargs,
    )
