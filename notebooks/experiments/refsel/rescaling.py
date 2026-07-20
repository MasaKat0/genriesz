"""Experiment E1a: raw Bregman objectives cannot rank candidates.

Section 4 of ``notebooks/experiments/REFERENCE_SELECTION_PLAN.md``.

Replacing a generator ``g`` by ``kappa * g`` leaves the fitted representer
unchanged but multiplies the objective value, and the held-out Bregman
criterion, by ``kappa``. The estimator is the same; its cross-validation score is
arbitrary. This is a deterministic statement, so one table settles it and no
Monte Carlo is required.

The penalty column matters. Minimizing ``mean(kappa g*(Phi beta / kappa) - M
beta) + P(beta)`` at ``beta = kappa b`` gives ``kappa`` times the unpenalized
criterion plus ``P(kappa b)``, so the invariance survives exactly when the
penalty is positively homogeneous of degree one. The experiments use an ``l1``
penalty, which is: **the incomparability is not an artefact of dropping
regularization, it holds for the penalized estimator actually fitted.** Under
``l2`` the fitted representer does move, but only because rescaling the
generator silently changes the effective penalty, which is no more defensible a
basis for ranking. Both cases are tabulated so neither reading is available.
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


def _heldout_bregman(generator, basis: ExperimentBasis, beta: FloatArray, X: FloatArray) -> float:
    """Unpenalized Bregman criterion of a fitted coefficient on a new sample."""

    Phi = np.asarray(basis(X), dtype=float)
    M = np.asarray(ATEFunctional(0).m_basis_matrix(X, basis), dtype=float)
    g_star, _ = generator.conjugate(X, Phi @ beta)
    return float(np.mean(np.asarray(g_star, dtype=float) - M @ beta))


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
    is measured against ``kappa = 1``. For the degree-one homogeneous ``l1``
    penalty it should sit at the solver tolerance and ``objective_ratio`` and
    ``heldout_ratio`` should equal ``kappa``, whether or not the multiplier is
    zero. Under ``l2`` the deviation is materially larger and the ratios depart
    from ``kappa``.
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
                fit = model.fit(train.X, max_iter=max_iter, tol=tolerance)
                row: dict[str, object] = {
                    "loss": spec.label.split("|")[0],
                    "penalty": penalty,
                    "penalty_multiplier": multiplier,
                    "kappa": kappa,
                    "status": fit.status,
                }
                if fit.success:
                    alpha = np.asarray(model.predict_alpha(train.X), dtype=float)
                    heldout = _heldout_bregman(generator, model.basis, fit.beta, holdout.X)
                    row["objective"] = float(fit.objective_value)
                    row["heldout_bregman"] = heldout
                    row["_alpha"] = alpha
                    if kappa == 1.0:
                        baseline = {
                            "alpha": alpha,
                            "objective": float(fit.objective_value),
                            "heldout": heldout,
                        }
                block.append(row)
            for row in block:
                alpha = row.pop("_alpha", None)
                if alpha is None or not baseline:
                    continue
                row["alpha_max_deviation"] = float(np.max(np.abs(alpha - baseline["alpha"])))
                row["objective_ratio"] = float(row["objective"]) / float(baseline["objective"])
                row["heldout_ratio"] = float(row["heldout_bregman"]) / float(baseline["heldout"])
            rows.extend(block)

    return pd.DataFrame(rows)
