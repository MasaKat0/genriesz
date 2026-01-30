from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .rkhs_grr import RKHS_GRR

try:
    # Optional dependency (PyTorch)
    from .nn_grr import NN_GRR
except ImportError:  # pragma: no cover
    NN_GRR = None  # type: ignore[assignment]


class GRR_ATE:
    """Generalized Riesz Regression estimator for the Average Treatment Effect (ATE).

    The estimator implements three common scores computed with cross-fitting:

    - **DM** (difference in conditional means)
    - **IPW** (inverse-propensity weighting using the estimated Riesz representer)
    - **AIPW** (augmented IPW)

    Notes
    -----
    This class is designed to be a thin orchestration layer around model-specific
    learners:

    - :class:`grr.NN_GRR` (neural network, requires PyTorch)
    - :class:`grr.RKHS_GRR` (RKHS / kernel approximation, NumPy/SciPy only)

    The implementation follows the structure of the original research code and
    intentionally keeps the core logic close to that version.
    """

    def __init__(self) -> None:
        self.model = None

    def estimate(
        self,
        covariates: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        method: str,
        riesz_loss: str,
        riesz_with_D: bool,
        riesz_link_name: str,
        cross_fitting_folds: int = 2,
        is_separate: bool = False,
        riesz_hidden_dim: int = 100,
        riesz_max_iter: int = 3000,
        reg_hidden_dim: int = 100,
        reg_max_iter: int = 3000,
        tol: float = 1e-10,
        lbd: float = 0.01,
        lr: float = 0.01,
        batch_size: int = 1000,
        folds: int = 5,
        num_basis: int = 100,
        sigma_list: Optional[np.ndarray] = None,
        lda_list: Optional[np.ndarray] = None,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """Estimate the ATE using GRR.

        Parameters
        ----------
        covariates:
            Covariate matrix of shape (n, d).
        treatment:
            Binary treatment indicator of shape (n,) or (n, 1).
        outcome:
            Outcome vector of shape (n,) or (n, 1).
        method:
            Either ``"NN_GRR"`` or ``"RKHS_GRR"``.
        riesz_loss:
            Loss used to fit the Riesz representer. In the provided learners,
            valid options include ``"SQ"``, ``"UKL"``, and ``"BKL"``.
        riesz_with_D:
            If ``True``, the treatment indicator is concatenated to the covariates
            when fitting the Riesz model.
        riesz_link_name:
            Link function name (e.g., ``"Linear"`` or ``"Logit"``).
        cross_fitting_folds:
            Historical argument kept for backward compatibility.

            - If set to a value **other than 2**, it will be used as the number of
              cross-fitting folds.
            - Otherwise, the value of ``folds`` is used (matching the original code).
        is_separate:
            (RKHS only) If ``True``, fit separate parameters for treated/control parts.
        riesz_hidden_dim, riesz_max_iter:
            (NN only) Network width and maximum epochs for the Riesz model.
        reg_hidden_dim, reg_max_iter:
            (NN only) Network width and maximum epochs for the regression model.
        tol, lbd, lr, batch_size:
            Optimization hyperparameters forwarded to the NN learners.
        folds:
            Number of folds used by the original implementation. By default this is
            also used as the number of cross-fitting folds.
        num_basis:
            (RKHS only) Number of basis points used for kernel approximation.
        sigma_list, lda_list:
            (RKHS only) Optional candidate grids used in kernel cross-validation.
        random_state:
            Random seed used for the cross-fitting split.
        verbose:
            If ``True``, print a small amount of diagnostic information.

        Returns
        -------
        A tuple containing

        ``(DM_est, IPW_est, AIPW_est, DM_ci_low, DM_ci_high, IPW_ci_low, IPW_ci_high, AIPW_ci_low, AIPW_ci_high)``.
        """

        covariates = np.asarray(covariates)
        treatment = np.asarray(treatment).reshape(-1)
        outcome = np.asarray(outcome).reshape(-1)

        if covariates.ndim != 2:
            raise ValueError("covariates must be a 2D array of shape (n, d).")
        if len(treatment) != len(covariates) or len(outcome) != len(covariates):
            raise ValueError("covariates, treatment, and outcome must have the same length.")

        n = len(covariates)

        # Backward-compatibility: the original code used `folds` for cross-fitting.
        n_folds = folds if cross_fitting_folds == 2 else int(cross_fitting_folds)
        if n_folds < 2:
            raise ValueError("Number of folds must be at least 2.")

        # Cross-fitting split indices
        cv_fold = np.arange(n_folds)
        cv_split0 = np.floor(np.arange(n) * n_folds / n)
        if random_state is None:
            perm = np.random.permutation(n)
        else:
            perm = np.random.default_rng(random_state).permutation(n)
        cv_index = cv_split0[perm]

        covariates_cv = [covariates[cv_index == k, :] for k in cv_fold]
        treatment_cv = [treatment[cv_index == k] for k in cv_fold]
        outcome_cv = [outcome[cv_index == k] for k in cv_fold]

        dm_scores: list[np.ndarray] = []
        ipw_scores: list[np.ndarray] = []
        aipw_scores: list[np.ndarray] = []

        for k in range(n_folds):
            # Build training/test splits for this fold
            covariates_te = covariates_cv[k]
            treatment_te = treatment_cv[k]
            outcome_te = outcome_cv[k]

            covariates_tr = np.concatenate([covariates_cv[j] for j in range(n_folds) if j != k], axis=0)
            treatment_tr = np.concatenate([treatment_cv[j] for j in range(n_folds) if j != k], axis=0)
            outcome_tr = np.concatenate([outcome_cv[j] for j in range(n_folds) if j != k], axis=0)

            if method == "NN_GRR":
                if NN_GRR is None:  # pragma: no cover
                    raise ImportError(
                        "NN_GRR requires PyTorch, but torch could not be imported. "
                        "Install with `pip install grr[torch]` (or install torch manually)."
                    )

                self.model = NN_GRR()
                self.model.riesz_fit(
                    covariates_tr,
                    treatment_tr,
                    riesz_loss=riesz_loss,
                    riesz_with_D=riesz_with_D,
                    riesz_link_name=riesz_link_name,
                    riesz_hidden_dim=riesz_hidden_dim,
                    riesz_max_iter=riesz_max_iter,
                    tol=tol,
                    lbd=lbd,
                    lr=lr,
                    batch_size=batch_size,
                )
                self.model.reg_fit(
                    covariates_tr,
                    treatment_tr,
                    outcome_tr,
                    reg_hidden_dim=reg_hidden_dim,
                    reg_max_iter=reg_max_iter,
                    tol=tol,
                    lbd=lbd,
                    lr=lr,
                    batch_size=batch_size,
                )

                est_riesz = self.model.riesz_predict(covariates_te, treatment_te)
                est_reg, est_reg_one, est_reg_zero = self.model.reg_predict_diff(
                    covariates_te, treatment_te
                )

                if verbose:
                    max_abs = float(np.max(np.abs(est_riesz))) if len(est_riesz) else float("nan")
                    print(f"[Fold {k+1}/{n_folds}] max|riesz|={max_abs:.3g}")

            elif method == "RKHS_GRR":
                self.model = RKHS_GRR()
                self.model.riesz_fit(
                    covariate=covariates_tr,
                    treatment=treatment_tr,
                    covariate_test=covariates_te,
                    treatment_test=treatment_te,
                    riesz_loss=riesz_loss,
                    riesz_with_D=riesz_with_D,
                    riesz_link_name=riesz_link_name,
                    is_separate=is_separate,
                    folds=folds,
                    num_basis=num_basis,
                    sigma_list=sigma_list,
                    lda_list=lda_list,
                )
                self.model.reg_fit(outcome_tr)

                est_riesz = self.model.riesz_predict()
                est_reg, est_reg_one, est_reg_zero = self.model.reg_predict_diff()
            else:
                raise ValueError(f"Unknown method: {method}. Use 'NN_GRR' or 'RKHS_GRR'.")

            dm_scores.append(est_reg_one - est_reg_zero)
            ipw_scores.append(est_riesz * outcome_te)
            aipw_scores.append(est_riesz * (outcome_te - est_reg) + est_reg_one - est_reg_zero)

        self.DM_score = np.concatenate(dm_scores)
        self.IPW_score = np.concatenate(ipw_scores)
        self.AIPW_score = np.concatenate(aipw_scores)

        self.DM_est = float(np.mean(self.DM_score))
        self.IPW_est = float(np.mean(self.IPW_score))
        self.AIPW_est = float(np.mean(self.AIPW_score))

        # Influence-function variances
        self.DM_var = float(np.var(self.DM_score - self.DM_est))
        self.IPW_var = float(np.var(self.IPW_score - self.IPW_est))
        self.AIPW_var = float(np.var(self.AIPW_score - self.AIPW_est))

        # Normal-approximation confidence regions
        self.DM_confband = 1.96 * np.sqrt(self.DM_var / n)
        self.IPW_confband = 1.96 * np.sqrt(self.IPW_var / n)
        self.AIPW_confband = 1.96 * np.sqrt(self.AIPW_var / n)

        self.DM_confregion = [self.DM_est - self.DM_confband, self.DM_est + self.DM_confband]
        self.IPW_confregion = [
            self.IPW_est - self.IPW_confband,
            self.IPW_est + self.IPW_confband,
        ]
        self.AIPW_confregion = [
            self.AIPW_est - self.AIPW_confband,
            self.AIPW_est + self.AIPW_confband,
        ]

        return (
            self.DM_est,
            self.IPW_est,
            self.AIPW_est,
            float(self.DM_confregion[0]),
            float(self.DM_confregion[1]),
            float(self.IPW_confregion[0]),
            float(self.IPW_confregion[1]),
            float(self.AIPW_confregion[0]),
            float(self.AIPW_confregion[1]),
        )
