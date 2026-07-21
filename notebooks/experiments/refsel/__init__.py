"""Reference-based loss--link selection experiment.

The design is specified in ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``.
Each experiment in that document maps to one entry point here:

===== ==================================================================
E1a   :func:`refsel.rescaling.rescaling_table`
E1b   selection rules in :mod:`refsel.selection`, reported by
      :func:`refsel.report.selection_rule_table`
E2    :func:`refsel.report.bias_bound_table`
E3    :func:`refsel.report.oracle_regret_table`
E4    :func:`refsel.report.uniform_coverage_table` and
      :func:`refsel.report.worst_case_coverage_table`
E5    :func:`refsel.report.reference_robustness_table` and
      :func:`refsel.report.reference_check_table`
E6    :func:`refsel.report.interval_length_table`
E7    the ``high`` scenarios of :func:`refsel.grids.publication_grid`
===== ==================================================================
"""

from __future__ import annotations

from .candidates import CandidateSpec, FoldLibrary, ScaledGenerator, candidate_grid
from .dgp import GeneratedData, generate_data, hidden_direction, make_fold_roles
from .grids import experiment_config, publication_grid
from .inference import bias_aware_critical_value
from .rescaling import rescaling_table
from .runner import (
    ExperimentConfig,
    Numerics,
    Scenario,
    load_experiment,
    run_experiment,
    run_repetition,
)
from .selection import RULES, DeltaBudget

__all__ = [
    "CandidateSpec",
    "DeltaBudget",
    "ExperimentConfig",
    "FoldLibrary",
    "GeneratedData",
    "Numerics",
    "RULES",
    "ScaledGenerator",
    "Scenario",
    "bias_aware_critical_value",
    "candidate_grid",
    "experiment_config",
    "generate_data",
    "hidden_direction",
    "load_experiment",
    "make_fold_roles",
    "publication_grid",
    "rescaling_table",
    "run_experiment",
    "run_repetition",
]
