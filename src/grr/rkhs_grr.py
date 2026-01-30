from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from scipy import optimize


class RKHS_GRR:
    """RKHS-based learner for the Riesz representer.

    The implementation follows the original research code and uses a Gaussian
    kernel feature approximation with cross-validation to choose hyperparameters.

    Supported losses
    ----------------
    The public API uses the following loss names (matching the original code):

    - ``"SQ"``: squared-distance-style objective
    - ``"UKL"``: unnormalized-KL-style objective
    - ``"BKL"``: binary-KL / MLE-style objective
    """

    def __init__(self) -> None:
        # Attributes are created during fitting
        pass

    def riesz_fit(
        self,
        covariate: np.ndarray,
        treatment: np.ndarray,
        covariate_test: np.ndarray,
        treatment_test: np.ndarray,
        riesz_loss: str,
        riesz_with_D: bool,
        riesz_link_name: str,
        is_separate: bool = False,
        folds: int = 2,
        num_basis: int = 50,
        sigma_list: Optional[np.ndarray] = None,
        lda_list: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the RKHS Riesz representer.

        Parameters
        ----------
        covariate, treatment:
            Training data.
        covariate_test, treatment_test:
            Held-out (fold) data used for cross-fitting.
        riesz_loss:
            One of ``"SQ"``, ``"UKL"``, ``"BKL"``.
        riesz_with_D:
            If True, include treatment as a feature in the kernel representation.
        riesz_link_name:
            ``"Linear"`` or ``"Logit"``.
        is_separate:
            If True, use separate parameter vectors for treated and control parts.
        folds:
            Number of folds used to cross-validate kernel hyperparameters.
        num_basis:
            Number of basis points used for the kernel feature approximation.
        sigma_list, lda_list:
            Optional candidate grids for kernel width and ridge penalty.
        """

        self.riesz_loss = riesz_loss
        if self.riesz_loss == "SQ":
            self.riesz_loss_func = self.sq_loss
        elif self.riesz_loss == "UKL":
            self.riesz_loss_func = self.ukl_loss
        elif self.riesz_loss == "BKL":
            self.riesz_loss_func = self.bkl_loss
        else:
            raise ValueError(f"Invalid riesz_loss: {self.riesz_loss}")

        self.riesz_with_D = bool(riesz_with_D)
        self.riesz_link_name = riesz_link_name
        self.is_separate = bool(is_separate)

        # Store treatments for later prediction
        self.treatment = np.asarray(treatment).reshape(-1)
        self.treatment_test = np.asarray(treatment_test).reshape(-1)

        (
            self.X_train,
            self.X1_train,
            self.X0_train,
            self.X_test,
            self.X_test1,
            self.X_test0,
        ) = self.kernel_cv(
            covariate_train=np.asarray(covariate),
            treatment_train=self.treatment,
            covariate_test=np.asarray(covariate_test),
            treatment_test=self.treatment_test,
            folds=folds,
            num_basis=num_basis,
            sigma_list=sigma_list,
            lda_list=lda_list,
        )

        # Train using the chosen ridge penalty
        self.train(self.X1_train, self.X0_train, self.treatment, self.lda_chosen)

    def _model_construction(self, param: np.ndarray, X1: np.ndarray, X0: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """Construct alpha(X, D) from parameters and feature matrices."""

        if self.is_separate:
            half = int(len(param) / 2)
            param1 = param[:half]
            param0 = param[half:]
            fx1 = X1 @ param1
            fx0 = X0 @ param0
        else:
            fx1 = X1 @ param
            fx0 = X0 @ param

        if self.riesz_link_name == "Linear":
            alpha1 = fx1
            alpha0 = fx0
            alpha = alpha0.copy()
            alpha[treatment == 1] = alpha1[treatment == 1]
        elif self.riesz_link_name == "Logit":
            ex1 = 1.0 / (1.0 + np.exp(-fx1))
            ex0 = 1.0 / (1.0 + np.exp(-fx0))
            alpha = treatment / ex1 - (1.0 - treatment) / (1.0 - ex0)
        else:
            raise ValueError(f"Invalid riesz_link_name: {self.riesz_link_name}")

        return alpha

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    def sq_loss(
        self,
        param: np.ndarray,
        X1: np.ndarray,
        X0: np.ndarray,
        treatment: np.ndarray,
        regularizer: float,
        return_param: bool = False,
        estimation_param: bool = True,
    ):
        """Squared-distance-style objective.

        If ``estimation_param`` is True, the closed-form ridge solution is computed
        and used as the parameter. This is how the original code obtains the SQ
        solution efficiently.
        """

        if estimation_param:
            X1X1H = X1.T.dot(X1) / len(X1)
            X0X0H = X0.T.dot(X0) / len(X0)
            X1h = np.sum(X1[treatment == 1], axis=0) / len(X1)
            X0h = np.sum(X0[treatment == 0], axis=0) / len(X0)

            if self.is_separate:
                beta1 = np.linalg.pinv(X1X1H + regularizer * np.eye(X1.shape[1]))
                beta1 = beta1.dot(X1h)
                beta0 = np.linalg.pinv(X0X0H + regularizer * np.eye(X0.shape[1]))
                beta0 = beta0.dot(X0h)
                param = np.concatenate([beta1, beta0])
            else:
                beta = np.linalg.pinv(X1X1H + X0X0H + regularizer * np.eye(X1.shape[1]))
                param = beta.dot(X1h + X0h)

        treatment0 = np.zeros_like(treatment)
        treatment1 = np.ones_like(treatment)
        alpha1 = self._model_construction(param, X1, X0, treatment1)
        alpha0 = self._model_construction(param, X1, X0, treatment0)

        loss = -2 * (alpha1 - alpha0) + treatment * alpha1**2 + (1 - treatment) * alpha0**2
        loss = np.mean(loss) + regularizer * np.sum(param**2)

        if return_param:
            return loss, param
        return loss

    def ukl_loss(self, param: np.ndarray, X1: np.ndarray, X0: np.ndarray, treatment: np.ndarray, regularizer: float, estimation_param: bool = True) -> float:
        """Unnormalized-KL-style objective."""

        treatment0 = np.zeros_like(treatment)
        treatment1 = np.ones_like(treatment)
        alpha1 = self._model_construction(param, X1, X0, treatment1)
        alpha0 = self._model_construction(param, X1, X0, treatment0)

        loss = (
            -(1 - treatment) * np.log(alpha1 - 1)
            - treatment * np.log(-alpha0 - 1)
            + treatment * alpha1
            - (1 - treatment) * alpha0
        )
        return float(np.mean(loss) + regularizer * np.sum(param**2))

    def bkl_loss(self, param: np.ndarray, X1: np.ndarray, X0: np.ndarray, treatment: np.ndarray, regularizer: float, estimation_param: bool = True) -> float:
        """Binary-KL / MLE-style objective."""

        treatment0 = np.zeros_like(treatment)
        treatment1 = np.ones_like(treatment)
        alpha1 = self._model_construction(param, X1, X0, treatment1)
        alpha0 = self._model_construction(param, X1, X0, treatment0)

        loss = -treatment * np.log(1.0 / alpha1) - (1.0 - treatment) * np.log(-1.0 / alpha0)
        return float(np.mean(loss) + regularizer * np.sum(param**2))

    # ------------------------------------------------------------------
    # Optimization helpers
    # ------------------------------------------------------------------
    def obj_func_gen(
        self,
        X1: np.ndarray,
        X0: np.ndarray,
        treatment: np.ndarray,
        regularizer: float,
        *,
        estimation_param: bool = True,
    ):
        """Generate an objective function compatible with ``scipy.optimize.minimize``."""

        def obj_func(param: np.ndarray):
            return self.riesz_loss_func(
                param,
                X1,
                X0,
                treatment,
                regularizer,
                estimation_param=estimation_param,
            )

        return obj_func

    def train(self, X1: np.ndarray, X0: np.ndarray, treatment: np.ndarray, lda_chosen: float) -> None:
        """Train the parameter vector given feature matrices and regularization."""

        if self.is_separate:
            init_param = np.random.uniform(size=X1.shape[1] * 2)
        else:
            init_param = np.zeros(X1.shape[1])

        if self.riesz_loss == "SQ":
            _, self.params = self.riesz_loss_func(
                init_param,
                X1,
                X0,
                treatment,
                lda_chosen,
                return_param=True,
                estimation_param=True,
            )
        else:
            obj_func = self.obj_func_gen(X1, X0, treatment, lda_chosen)
            self.result = optimize.minimize(obj_func, init_param, method="BFGS")
            self.params = self.result.x

    def riesz_predict(self) -> np.ndarray:
        """Predict the Riesz representer on the test set created in :meth:`riesz_fit`."""

        if self.is_separate:
            half = int(len(self.params) / 2)
            param1 = self.params[:half]
            param0 = self.params[half:]
            fx1 = self.X_test @ param1
            fx0 = self.X_test @ param0
            fx = fx0.copy()
            fx[self.treatment_test == 1] = fx1[self.treatment_test == 1]
        else:
            fx = self.X_test @ self.params

        if self.riesz_link_name == "Linear":
            alpha = fx
        elif self.riesz_link_name == "Logit":
            ex = 1.0 / (1.0 + np.exp(-fx))
            alpha = self.treatment_test / ex - (1.0 - self.treatment_test) / (1.0 - ex)
        else:
            raise ValueError(f"Invalid riesz_link_name: {self.riesz_link_name}")

        self.riesz = alpha
        return self.riesz

    # ------------------------------------------------------------------
    # Kernel construction and cross validation
    # ------------------------------------------------------------------
    def dist(
        self,
        X: np.ndarray,
        X1: np.ndarray,
        X0: np.ndarray,
        X_test: np.ndarray,
        X_test1: np.ndarray,
        X_test0: np.ndarray,
        num_basis: Union[int, bool] = False,
    ):
        """Compute squared distance matrices between training features and bases."""

        d, n = X.shape

        if num_basis is False:
            num_basis = 1000

        idx = np.random.permutation(n)[: int(num_basis)]
        C = X[:, idx]

        # Squared distances
        XC_dist = CalcDistanceSquared(X, C)
        X1C_dist = CalcDistanceSquared(X1, C)
        X0C_dist = CalcDistanceSquared(X0, C)
        XCtest_dist = CalcDistanceSquared(X_test, C)
        X1Ctest_dist = CalcDistanceSquared(X_test1, C)
        X0Ctest_dist = CalcDistanceSquared(X_test0, C)
        CC_dist = CalcDistanceSquared(C, C)
        return XC_dist, X1C_dist, X0C_dist, XCtest_dist, X1Ctest_dist, X0Ctest_dist, CC_dist, n, int(num_basis)

    def kernel_cv(
        self,
        covariate_train: np.ndarray,
        treatment_train: np.ndarray,
        covariate_test: np.ndarray,
        treatment_test: np.ndarray,
        folds: int = 5,
        num_basis: Union[int, bool] = False,
        sigma_list: Optional[np.ndarray] = None,
        lda_list: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build kernel-based features and choose (sigma, lambda) by cross-validation."""

        # Add treatment as a feature if requested
        if self.riesz_with_D:
            treatment0 = treatment_train * 0
            treatment1 = treatment0 + 1
            X_train1 = np.concatenate([np.array([treatment1]).T, covariate_train], axis=1)
            X_train0 = np.concatenate([np.array([treatment0]).T, covariate_train], axis=1)
            X_train = np.concatenate([np.array([treatment_train]).T, covariate_train], axis=1)
        else:
            X_train1 = covariate_train
            X_train0 = covariate_train
            X_train = covariate_train

        if self.riesz_with_D:
            treatment_test0 = treatment_test * 0
            treatment_test1 = treatment_test0 + 1

            X_test = np.concatenate([np.array([treatment_test]).T, covariate_test], axis=1)
            X_test1 = np.concatenate([np.array([treatment_test1]).T, covariate_test], axis=1)
            X_test0 = np.concatenate([np.array([treatment_test0]).T, covariate_test], axis=1)
        else:
            X_test = covariate_test
            X_test1 = covariate_test
            X_test0 = covariate_test

        # Transpose to shape (d, n)
        X_train, X_train1, X_train0, X_test, X_test1, X_test0 = (
            X_train.T,
            X_train1.T,
            X_train0.T,
            X_test.T,
            X_test1.T,
            X_test0.T,
        )

        XC_dist, X1C_dist, X0C_dist, XCtest_dist, X1Ctest_dist, X0Ctest_dist, _CC_dist, n, num_basis = self.dist(
            X_train,
            X_train1,
            X_train0,
            X_test,
            X_test1,
            X_test0,
            num_basis,
        )

        # Cross-validation split indices
        cv_fold = np.arange(folds)
        cv_split0 = np.floor(np.arange(n) * folds / n)
        cv_index = cv_split0[np.random.permutation(n)]

        if sigma_list is None:
            sigma_list = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
        if lda_list is None:
            lda_list = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])

        score_cv = np.zeros((len(sigma_list), len(lda_list)))

        for sigma_idx, sigma in enumerate(sigma_list):
            # Pre-compute features per fold
            h1_cv = [np.exp(-X1C_dist[:, cv_index == k] / (2 * sigma**2)) for k in cv_fold]
            h0_cv = [np.exp(-X0C_dist[:, cv_index == k] / (2 * sigma**2)) for k in cv_fold]
            d_cv = [treatment_train[cv_index == k] for k in cv_fold]

            for k in range(folds):
                h1te = h1_cv[k].T
                h0te = h0_cv[k].T
                dte = d_cv[k]

                h1tr = np.concatenate([h1_cv[j].T for j in range(folds) if j != k], axis=0)
                h0tr = np.concatenate([h0_cv[j].T for j in range(folds) if j != k], axis=0)
                dtr = np.concatenate([d_cv[j] for j in range(folds) if j != k], axis=0)

                # Add bias term
                one_tr = np.ones((len(h1tr), 1))
                h1tr = np.concatenate([h1tr, one_tr], axis=1)
                h0tr = np.concatenate([h0tr, one_tr], axis=1)
                one_te = np.ones((len(h1te), 1))
                h1te = np.concatenate([h1te, one_te], axis=1)
                h0te = np.concatenate([h0te, one_te], axis=1)

                for lda_idx, lda in enumerate(lda_list):
                    self.train(h1tr, h0tr, dtr, lda)

                    # Evaluate score on the hold-out fold (do not re-fit params)
                    obj_func = self.obj_func_gen(
                        h1te,
                        h0te,
                        dte,
                        0.0,
                        estimation_param=False,
                    )
                    score_cv[sigma_idx, lda_idx] += float(obj_func(self.params))

        sigma_idx_chosen, lda_idx_chosen = np.unravel_index(np.argmin(score_cv), score_cv.shape)
        self.sigma_chosen = float(sigma_list[sigma_idx_chosen])
        self.lda_chosen = float(lda_list[lda_idx_chosen])

        # Construct final feature matrices with the chosen sigma
        sigma_chosen = self.sigma_chosen

        x_train = np.exp(-XC_dist / (2 * sigma_chosen**2)).T
        x1_train = np.exp(-X1C_dist / (2 * sigma_chosen**2)).T
        x0_train = np.exp(-X0C_dist / (2 * sigma_chosen**2)).T
        x_test = np.exp(-XCtest_dist / (2 * sigma_chosen**2)).T
        x_test1 = np.exp(-X1Ctest_dist / (2 * sigma_chosen**2)).T
        x_test0 = np.exp(-X0Ctest_dist / (2 * sigma_chosen**2)).T

        one_tr = np.ones((len(x1_train), 1))
        X_train = np.concatenate([x_train, one_tr], axis=1)
        X1_train = np.concatenate([x1_train, one_tr], axis=1)
        X0_train = np.concatenate([x0_train, one_tr], axis=1)

        one_te = np.ones((len(x_test), 1))
        X_test = np.concatenate([x_test, one_te], axis=1)
        X_test1 = np.concatenate([x_test1, one_te], axis=1)
        X_test0 = np.concatenate([x_test0, one_te], axis=1)

        return X_train, X1_train, X0_train, X_test, X_test1, X_test0

    def reg_fit(self, Y: np.ndarray) -> None:
        """Fit a ridge regression model for E[Y | X, T] using the same RKHS features."""

        Y = np.asarray(Y).reshape(-1)

        if self.riesz_with_D:
            XtX = self.X_train.T.dot(self.X_train)
            p = XtX.shape[0]
            self.reg_param = np.linalg.pinv(XtX + self.lda_chosen * np.eye(p)).dot(self.X_train.T.dot(Y))
        else:
            XtX1 = self.X_train[self.treatment == 1].T.dot(self.X_train[self.treatment == 1])
            p1 = XtX1.shape[0]
            self.reg_param1 = np.linalg.pinv(XtX1 + self.lda_chosen * np.eye(p1)).dot(
                self.X_train[self.treatment == 1].T.dot(Y[self.treatment == 1])
            )

            XtX0 = self.X_train[self.treatment == 0].T.dot(self.X_train[self.treatment == 0])
            p0 = XtX0.shape[0]
            self.reg_param0 = np.linalg.pinv(XtX0 + self.lda_chosen * np.eye(p0)).dot(
                self.X_train[self.treatment == 0].T.dot(Y[self.treatment == 0])
            )

    def reg_predict_diff(self):
        """Return m(X, T), m(X, 1) and m(X, 0) on the test set."""

        if self.riesz_with_D:
            est_reg = self.X_test @ self.reg_param
            est_reg_one = self.X_test1 @ self.reg_param
            est_reg_zero = self.X_test0 @ self.reg_param
        else:
            est_reg_one = self.X_test1 @ self.reg_param1
            est_reg_zero = self.X_test0 @ self.reg_param0
            est_reg = est_reg_zero.copy()
            est_reg[self.treatment_test == 1] = est_reg_one[self.treatment_test == 1]

        return est_reg, est_reg_one, est_reg_zero


def CalcDistanceSquared(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Return the matrix of squared distances between columns of X and C."""

    Xsum = np.sum(X**2, axis=0).T
    Csum = np.sum(C**2, axis=0)
    return Xsum[np.newaxis, :] + Csum[:, np.newaxis] - 2 * np.dot(C.T, X)
