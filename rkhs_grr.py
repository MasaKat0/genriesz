import numpy as np
from scipy import optimize


class RKHS_GRR:
    """
    Riesz representer learner based on an RKHS feature representation.

    The class supports several loss functions:
    - LS: least-squares-type loss
    - KL: KL-type loss
    - TL: tailored/logistic-type loss
    - MLE: maximum likelihood type loss
    """

    def __init__(self):
        # Attributes are set during fitting
        pass

    def riesz_fit(
        self,
        covariate,
        treatment,
        covariate_test,
        treatment_test,
        riesz_loss,
        riesz_with_D,
        riesz_link_name,
        is_separate=False,
        folds=2,
        num_basis=50,
    ):
        """
        Entry point for fitting the Riesz representer in RKHS.

        Parameters
        ----------
        covariate : ndarray
            Training covariates.
        treatment : ndarray
            Training treatment indicator (0/1).
        covariate_test : ndarray
            Test covariates.
        treatment_test : ndarray
            Test treatment indicator (0/1) for test points.
        riesz_loss : {"LS", "KL", "TL", "MLE"}
            Type of loss function.
        riesz_with_D : bool
            If True, the treatment is concatenated with covariates as features.
        riesz_link_name : {"Linear", "Logit"}
            Link function applied to the score to obtain the Riesz representer.
        is_separate : bool
            If True, use separate parameters for treated and control parts.
        folds : int
            Number of folds for cross-validation in kernel construction.
        num_basis : int
            Number of basis points (centers) for the kernel approximation.
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

        self.riesz_with_D = riesz_with_D
        self.riesz_link_name = riesz_link_name
        self.is_separate = is_separate

        # Store training treatment for later prediction
        self.treatment = treatment
        self.treatment_test = treatment_test

        # Construct kernel-based feature matrices with CV to choose sigma and lambda
        (
            self.X_train,
            self.X1_train,
            self.X0_train,
            self.X_test,
            self.X_test1,
            self.X_test0,
        ) = self.kernel_cv(
            covariate_train=covariate,
            treatment_train=treatment,
            covariate_test=covariate_test,
            treatment_test=treatment_test,
            folds=folds,
            num_basis=num_basis,
        )

        # Train model with chosen regularization parameter
        self.train(self.X1_train, self.X0_train, self.treatment, self.lda_chosen)

    def _model_construction(self, param, X1, X0, treatment):
        """
        Construct alpha(X, D) from parameters and feature matrices.

        If is_separate is True, parameter vector is split into (param1, param0);
        otherwise param is shared for both treated and control parts.
        """
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
            # Linear link
            alpha1 = fx1
            alpha0 = fx0

            # Construct alpha by selecting alpha1 where treatment==1, else alpha0
            alpha = alpha0.copy()
            alpha[treatment == 1] = alpha1[treatment == 1]
        elif self.riesz_link_name == "Logit":
            # Logistic link, interpreting fx1 and fx0 as logits of propensity
            ex1 = 1.0 / (1.0 + np.exp(-fx1))
            ex0 = 1.0 / (1.0 + np.exp(-fx0))
            alpha = treatment / ex1 - (1.0 - treatment) / (1.0 - ex0)
        else:
            raise ValueError(f"Invalid riesz_link_name: {self.riesz_link_name}")

        return alpha

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    def sq_loss(self, param, X1, X0, treatment, regularizer, return_param=False, estimation_param=True):
        """
        Least-squares-type closed-form solution and objective value.

        If return_param is True, returns (loss, param); otherwise returns loss only.
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
                beta = np.linalg.pinv(
                    X1X1H + X0X0H + regularizer * np.eye(X1.shape[1])
                )
                param = beta.dot(X1h + X0h)

        treatment0 = np.zeros_like(treatment)
        treatment1 = np.ones_like(treatment)
        alpha1 = self._model_construction(param, X1, X0, treatment1)
        alpha0 = self._model_construction(param, X1, X0, treatment0)

        loss = -2 * (alpha1 - alpha0) + treatment * alpha1**2 + (1 - treatment) * alpha0**2
        loss = np.mean(loss) + regularizer * np.sum(param**2)

        if return_param:
            return loss, param
        else:
            return loss

    def ukl_loss(self, param, X1, X0, treatment, regularizer, estimation_param=True):
        """
        Tailored/logistic-type loss:
        E[ -(1-T)log(alpha1 - 1) - T log(-alpha0 - 1) + T alpha1 - (1-T) alpha0 ] + regularization.
        """
        treatment0 = np.zeros_like(treatment)
        treatment1 = np.ones_like(treatment)
        alpha1 = self._model_construction(param, X1, X0, treatment1)
        alpha0 = self._model_construction(param, X1, X0, treatment0)

        loss = (
            - (1 - treatment) * np.log(alpha1 - 1)
            - treatment * np.log(-alpha0 - 1)
            + treatment * alpha1
            - (1 - treatment) * alpha0
        )
        loss = np.mean(loss) + regularizer * np.sum(param**2)
        return loss

    def bkl_loss(self, param, X1, X0, treatment, regularizer, estimation_param=True):
        """
        MLE-type loss:
        E[ -T log(1/alpha1) - (1-T) log(-1/alpha0) ] + regularization.
        """
        treatment0 = np.zeros_like(treatment)
        treatment1 = np.ones_like(treatment)
        alpha1 = self._model_construction(param, X1, X0, treatment1)
        alpha0 = self._model_construction(param, X1, X0, treatment0)

        loss = -treatment * np.log(1.0 / alpha1) - (1.0 - treatment) * np.log(-1.0 / alpha0)
        loss = np.mean(loss) + regularizer * np.sum(param**2)
        return loss

    # ------------------------------------------------------------------
    # Optimization helpers
    # ------------------------------------------------------------------
    def optimize(self, covariate, treatment, x_test):
        """
        Legacy stub method (not used in the current implementation).

        This method was previously calling an undefined 'self.minimize'.
        To avoid silent bugs, we raise an explicit error.
        """
        raise NotImplementedError(
            "optimize() is not implemented. Use `riesz_fit` instead."
        )

    def obj_func_gen(self, X1, X0, treatment, regularizer, estimation_param=True):
        """
        Generate an objective function for scipy.optimize.minimize.
        """
        def obj_func(param):
            return self.riesz_loss_func(param, X1, X0, treatment, regularizer, estimation_param=True)

        return obj_func

    def train(self, X1, X0, treatment, lda_chosen, folds=2, num_basis=50):
        """
        Train the parameter vector given feature matrices and regularization.

        For LS loss, a closed-form solution is used.
        For other losses, BFGS is used to minimize the objective.
        """
        if self.is_separate:
            init_param = np.random.uniform(size=X1.shape[1] * 2)
        else:
            init_param = np.zeros(X1.shape[1])

        if self.riesz_loss == "SQ":
            _, self.params = self.riesz_loss_func(
                init_param, X1, X0, treatment, lda_chosen, return_param=True
            )
        else:
            obj_func = self.obj_func_gen(X1, X0, treatment, lda_chosen)
            self.result = optimize.minimize(obj_func, init_param, method="BFGS")
            self.params = self.result.x

    def riesz_predict(self):
        """
        Predict the Riesz representer on the test set constructed in `riesz_fit`.

        Returns
        -------
        riesz : ndarray
            Estimated Riesz representer values on X_test.
        """
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
    def dist(self, X, X1, X0, X_test, X_test1, X_test0, num_basis=False):
        """
        Compute squared distance matrices between training features and bases.

        Parameters
        ----------
        X : ndarray, shape (d, n)
            Full training feature matrix.
        X1 : ndarray
            Feature matrix for treated points.
        X0 : ndarray
            Feature matrix for control points.
        X_for_DC : ndarray
            Feature matrix for test (or additional) points to compute DC distances.
        num_basis : int or bool
            Number of basis points (centers). If False, defaults to 1000.

        Returns
        -------
        X1C_dist, X0C_dist, DC_dist, CC_dist, n, num_basis
        """
        d, n = X.shape

        if num_basis is False:
            num_basis = 1000

        idx = np.random.permutation(n)[:num_basis]
        C = X[:, idx]

        # Squared distances
        XC_dist = CalcDistanceSquared(X, C)
        X1C_dist = CalcDistanceSquared(X1, C)
        X0C_dist = CalcDistanceSquared(X0, C)
        XCtest_dist = CalcDistanceSquared(X_test, C)
        X1Ctest_dist = CalcDistanceSquared(X_test1, C)
        X0Ctest_dist = CalcDistanceSquared(X_test0, C)
        CC_dist = CalcDistanceSquared(C, C)
        return XC_dist, X1C_dist, X0C_dist, XCtest_dist, X1Ctest_dist, X0Ctest_dist, CC_dist, n, num_basis

    def kernel_cv(
        self,
        covariate_train,
        treatment_train,
        covariate_test,
        treatment_test,
        folds=5,
        num_basis=False,
        sigma_list=None,
        lda_list=None,
    ):
        """
        Build kernel-based features and choose (sigma, lambda) by cross-validation.

        Returns
        -------
        X1_train, X0_train, X_test : ndarray
            Feature matrices for treated, control, and test sets.
        """
        # Add treatment as feature if requested
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
            
            X_test = np.concatenate(
                [np.array([treatment_test]).T, covariate_test],
                axis=1,
            )
            X_test1 = np.concatenate(
                [np.array([treatment_test1]).T, covariate_test],
                axis=1,
            )
            X_test0 = np.concatenate(
                [np.array([treatment_test0]).T, covariate_test],
                axis=1,
            )
        else:
            X_test = covariate_test
            X_test1 = covariate_test
            X_test0 = covariate_test

        # Transpose to shape (d, n) as used in distance computation
        X_train, X_train1, X_train0, X_test, X_test1, X_test0 = (
            X_train.T,
            X_train1.T,
            X_train0.T,
            X_test.T,
            X_test1.T,
            X_test0.T,
        )

        XC_dist, X1C_dist, X0C_dist, XCtest_dist, X1Ctest_dist, X0Ctest_dist, CC_dist, n, num_basis = self.dist(
            X_train, X_train1, X_train0, X_test, X_test1, X_test0, num_basis
        )

        # Cross-validation split indices
        cv_fold = np.arange(folds)
        cv_split0 = np.floor(np.arange(n) * folds / n)
        cv_index = cv_split0[np.random.permutation(n)]

        # Candidate sigma and lambda lists
        if sigma_list is None:
            sigma_list = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
        if lda_list is None:
            lda_list = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])

        score_cv = np.zeros((len(sigma_list), len(lda_list)))

        for sigma_idx, sigma in enumerate(sigma_list):
            # Pre-compute h-vectors per fold to speed up
            h1_cv = []
            h0_cv = []
            d_cv = []
            for k in cv_fold:
                h1_cv.append(np.exp(-X1C_dist[:, cv_index == k] / (2 * sigma**2)))
                h0_cv.append(np.exp(-X0C_dist[:, cv_index == k] / (2 * sigma**2)))
                d_cv.append(treatment_train[cv_index == k])

            for k in range(folds):
                # Build training and test h-vectors for this fold
                count = 0
                for j in range(folds):
                    if j == k:
                        h1te = h1_cv[j].T
                        h0te = h0_cv[j].T
                        dte = d_cv[j]
                    else:
                        if count == 0:
                            h1tr = h1_cv[j].T
                            h0tr = h0_cv[j].T
                            dtr = d_cv[j]
                            count += 1
                        else:
                            h1tr = np.append(h1tr, h1_cv[j].T, axis=0)
                            h0tr = np.append(h0tr, h0_cv[j].T, axis=0)
                            dtr = np.append(dtr, d_cv[j], axis=0)

                # Add bias (constant) term
                one_tr = np.ones((len(h1tr), 1))
                h1tr = np.concatenate([h1tr, one_tr], axis=1)
                h0tr = np.concatenate([h0tr, one_tr], axis=1)
                one_te = np.ones((len(h1te), 1))
                h1te = np.concatenate([h1te, one_te], axis=1)
                h0te = np.concatenate([h0te, one_te], axis=1)

                for lda_idx, lda in enumerate(lda_list):
                    # Train on current (sigma, lambda, fold)
                    self.train(h1tr, h0tr, dtr, lda)
                    # Evaluate CV score on hold-out fold
                    obj_func = self.obj_func_gen(h1te, h0te, dte, 0.0, estimation_param=False)
                    score = obj_func(self.params)
                    score_cv[sigma_idx, lda_idx] += score

        # Choose (sigma, lambda) with minimal CV score
        sigma_idx_chosen, lda_idx_chosen = np.unravel_index(
            np.argmin(score_cv), score_cv.shape
        )
        sigma_chosen = sigma_list[sigma_idx_chosen]
        lda_chosen = lda_list[lda_idx_chosen]
        self.sigma_chosen = sigma_chosen
        self.lda_chosen = lda_chosen

        # Construct final feature matrices with chosen sigma
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
    
    def reg_fit(self, Y):
        if self.riesz_with_D:
            self.reg_param = np.linalg.pinv(self.X_train.T.dot(self.X_train) + self.lda_chosen).dot(self.X_train.T.dot(Y))
        else:
            self.reg_param1 = np.linalg.pinv(self.X_train[self.treatment == 1].T.dot(self.X_train[self.treatment == 1]) + self.lda_chosen).dot(self.X_train[self.treatment == 1].T.dot(Y[self.treatment == 1]))
            self.reg_param0 = np.linalg.pinv(self.X_train[self.treatment == 0].T.dot(self.X_train[self.treatment == 0]) + self.lda_chosen).dot(self.X_train[self.treatment == 0].T.dot(Y[self.treatment == 0]))
        
    def reg_predict_diff(self):
        if self.riesz_with_D:
            est_reg = self.X_test @ self.reg_param
            est_reg_one = self.X_test1 @ self.reg_param
            est_reg_zero = self.X_test0 @ self.reg_param
        else:
            est_reg_one = self.X_test1 @ self.reg_param1
            est_reg_zero = self.X_test0 @ self.reg_param0
            est_reg = est_reg_zero
            est_reg[self.treatment_test == 1] = est_reg_one[self.treatment_test == 1]
            
        return est_reg, est_reg_one, est_reg_zero


def CalcDistanceSquared(X, C):
    """
    Calculate the squared distances between columns of X and columns of C.

    XC_dist2 = CalcDistanceSquared(X, C)
    [XC_dist2]_{ij} = ||X[:, j] - C[:, i]||^2

    Parameters
    ----------
    X : ndarray, shape (d, n)
        First set of vectors (columns).
    C : ndarray, shape (d, nc)
        Second set of vectors (columns).

    Returns
    -------
    XC_dist : ndarray, shape (nc, n)
        Squared distance matrix.
    """
    Xsum = np.sum(X**2, axis=0).T
    Csum = np.sum(C**2, axis=0)
    XC_dist = Xsum[np.newaxis, :] + Csum[:, np.newaxis] - 2 * np.dot(C.T, X)
    return XC_dist