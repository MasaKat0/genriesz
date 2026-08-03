"""Experiment E1a: raw Bregman objectives cannot rank candidates.

Section 4 of ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``.

For an unpenalized fit, replacing a generator ``g`` by ``kappa * g``
leaves the fitted representer unchanged and multiplies the objective value by
``kappa``. The held-out Bregman criterion obeys the same identity when the
exact link is defined on every held-out observation. Thus, the raw criterion
has an arbitrary numerical scale. This is a deterministic statement, so one
table settles it and no Monte Carlo is required.

The penalty column matters. The unpenalized rows show the exact scale
identity. The table also reports the numerical ``l1`` specification used by
``GRRGLM`` and an ``l2`` specification. Rescaling changes the effective penalty
unless the penalty and its numerical implementation transform homogeneously,
so penalized objective values do not provide a generator-independent ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from genriesz.functionals import ATEFunctional
from genriesz.glm import GRRGLM

from .candidates import (
    CandidateSpec,
    ExperimentBasis,
    ScaledGenerator,
    make_generator,
)
from .dgp import Design, FloatArray, generate_data

#: Rescaling factors applied to the generator.
KAPPA_GRID: tuple[float, ...] = (0.5, 1.0, 2.0)


def _heldout_bregman(
    generator, basis: ExperimentBasis, beta: FloatArray, X: FloatArray
) -> dict[str, object]:
    """Evaluate a held-out Bregman criterion without replacing invalid links.

    A candidate can be valid on the fitting sample and leave the exact dual
    domain on a new sample. The rescaling calculation records that event rather
    than clipping the dual index or substituting another generator.
    """

    Phi = np.asarray(basis(X), dtype=float)
    M = np.asarray(ATEFunctional(0).m_basis_matrix(X, basis), dtype=float)
    evaluation = generator.conjugate_status(X, Phi @ beta)
    valid_rows = int(np.sum(evaluation.valid))
    total_rows = int(X.shape[0])
    result: dict[str, object] = {
        "heldout_valid_rows": valid_rows,
        "heldout_total_rows": total_rows,
        "heldout_valid_fraction": valid_rows / total_rows,
    }
    if valid_rows != total_rows:
        result["heldout_status"] = "dual_domain_failure"
        result["heldout_bregman"] = float("nan")
        return result

    value = float(
        np.mean(np.asarray(evaluation.conjugate, dtype=float) - M @ beta)
    )
    if not np.isfinite(value):
        result["heldout_status"] = "nonfinite_criterion"
        result["heldout_bregman"] = float("nan")
        return result
    result["heldout_status"] = "available"
    result["heldout_bregman"] = value
    return result


#: ``(penalty, p_norm, multiplier)`` combinations tabulated.
PENALTY_GRID: tuple[tuple[str, float, float], ...] = (
    ("l1", 1.0, 0.0),
    ("l1", 1.0, 1.0),
    ("l2", 2.0, 1.0),
)


def rescaling_table(
    *,
    n: int = 2000,
    design: Design = "low",
    overlap_scale: float = 1.5,
    dictionary: str = "second_order",
    penalties: tuple[tuple[str, float, float], ...] = PENALTY_GRID,
    losses: tuple[str, ...] = ("SQ", "UKL", "BP"),
    omega: float = 0.5,
    seed: int = 20260720,
    tolerance: float = 1e-10,
    max_iter: int = 5000,
) -> pd.DataFrame:
    """Return the rescaling invariance table.

    One row per ``(loss, penalty, multiplier, kappa)``. ``alpha_max_deviation``
    is measured against ``kappa = 1``. The unpenalized rows give the exact
    scale identity; the penalized rows report the behavior of the numerical
    penalties used by ``GRRGLM``. ``heldout_ratio`` is reported only when the exact
    link is defined for every held-out observation. A restricted-domain loss
    that leaves its domain receives ``heldout_status = dual_domain_failure``;
    no finite value is substituted. Under ``l2`` the deviation is materially
    larger because rescaling changes the effective penalty.
    """

    train = generate_data(
        n=n, design=design, overlap_scale=overlap_scale, hidden_scale=0.0, seed=seed
    )
    holdout = generate_data(
        n=n, design=design, overlap_scale=overlap_scale, hidden_scale=0.0, seed=seed + 1
    )

    rows: list[dict[str, object]] = []
    for loss in losses:
        for penalty, p_norm, multiplier in penalties:
            spec = CandidateSpec(
                loss=loss,  # type: ignore[arg-type]
                dictionary=dictionary,  # type: ignore[arg-type]
                penalty_multiplier=multiplier,
                omega=omega if loss == "BP" else None,
            )
            inner = make_generator(spec)
            reference_basis = ExperimentBasis(spec.dictionary).fit(train.X)
            lam = multiplier * np.sqrt(np.log(max(reference_basis.n_features, 2)) / train.n)
            block: list[dict[str, object]] = []
            baseline: dict[str, object] = {}
            for kappa in KAPPA_GRID:
                generator = inner if kappa == 1.0 else ScaledGenerator(inner, kappa)
                model = GRRGLM(
                    basis=ExperimentBasis(spec.dictionary).fit(train.X),
                    generator=generator,
                    functional=ATEFunctional(treatment_index=0),
                    penalty=penalty,
                    lam=lam,
                    p_norm=p_norm,
                )
                fit = model.fit(
                    train.X, max_iter=max_iter, tol=tolerance, fit_basis=False
                )
                row: dict[str, object] = {
                    "loss": spec.label.split("|")[0],
                    "penalty": penalty,
                    "penalty_multiplier": multiplier,
                    "kappa": kappa,
                    "status": fit.status,
                }
                if fit.success:
                    alpha = np.asarray(model.predict_alpha(train.X), dtype=float)
                    heldout = _heldout_bregman(
                        generator, model.basis, fit.beta, holdout.X
                    )
                    row["objective"] = float(fit.objective_value)
                    row.update(heldout)
                    row["_alpha"] = alpha
                    if kappa == 1.0:
                        baseline = {
                            "alpha": alpha,
                            "objective": float(fit.objective_value),
                            "heldout": row["heldout_bregman"],
                        }
                else:
                    row.update(
                        {
                            "heldout_status": "fit_failure",
                            "heldout_bregman": float("nan"),
                            "heldout_valid_rows": 0,
                            "heldout_total_rows": int(holdout.n),
                            "heldout_valid_fraction": 0.0,
                        }
                    )
                block.append(row)
            for row in block:
                alpha = row.pop("_alpha", None)
                if alpha is None or not baseline:
                    continue
                row["alpha_max_deviation"] = float(
                    np.max(np.abs(alpha - baseline["alpha"]))
                )
                row["objective_ratio"] = float(row["objective"]) / float(
                    baseline["objective"]
                )
                heldout_value = float(row.get("heldout_bregman", np.nan))
                baseline_heldout = float(baseline.get("heldout", np.nan))
                if np.isfinite(heldout_value) and np.isfinite(baseline_heldout):
                    row["heldout_ratio"] = heldout_value / baseline_heldout
                else:
                    row["heldout_ratio"] = float("nan")
            rows.extend(block)

    return pd.DataFrame(rows)
