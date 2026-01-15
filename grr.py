import numpy as np
from nn_grr import *
from rkhs_grr import *


class GRR_ATE:
    def __init__(self):
        pass
    
    def estimate(self, covariates, treatment, outcome, method, riesz_loss, riesz_with_D, riesz_link_name, cross_fitting_folds=2, is_separate=False, riesz_hidden_dim=100,
        riesz_max_iter=3000,
                 reg_hidden_dim = 100,
                 reg_max_iter = 3000,
        tol=1e-10,
        lbd=0.01,
        lr=0.01,
        batch_size=1000,
        folds=5, num_basis=100):
        
        n = len(covariates)
        # Cross-validation split indices
        cv_fold = np.arange(folds)
        cv_split0 = np.floor(np.arange(n) * folds / n)
        cv_index = cv_split0[np.random.permutation(n)]

        covariates_cv = []
        treatment_cv = []
        outcome_cv = []
        for k in cv_fold:
            covariates_cv.append(covariates[cv_index == k, :])
            treatment_cv.append(treatment[cv_index == k])
            outcome_cv.append(outcome[cv_index == k])

        for k in range(folds):
            # Build training and test h-vectors for this fold
            count = 0
            for j in range(folds):
                if j == k:
                    covariates_te = covariates_cv[j]
                    treatment_te = treatment_cv[j]
                    outcome_te = outcome_cv[j]
                else:
                    if count == 0:
                        covariates_tr = covariates_cv[j]
                        treatment_tr = treatment_cv[j]
                        outcome_tr = outcome_cv[j]
                        count += 1
                    else:
                        covariates_tr = np.append(covariates_tr, covariates_cv[j], axis=0)
                        treatment_tr = np.append(treatment_tr, treatment_cv[j], axis=0)
                        outcome_tr = np.append(outcome_tr, outcome_cv[j], axis=0)
            
            if method == "NN_GRR":
                self.model = NN_GRR()
                self.model.riesz_fit(covariates_tr, treatment_tr, riesz_loss=riesz_loss, riesz_with_D=riesz_with_D, riesz_link_name=riesz_link_name, riesz_hidden_dim=riesz_hidden_dim, riesz_max_iter=riesz_max_iter)
                self.model.reg_fit(covariates_tr, treatment_tr, outcome_tr, reg_hidden_dim=reg_hidden_dim, reg_max_iter=reg_max_iter)
                
                est_riesz = self.model.riesz_predict(covariates_te, treatment_te)
                est_reg, est_reg_one, est_reg_zero = self.model.reg_predict_diff(covariates_te, treatment_te)
                print("est", est_riesz)
                print("est_riesz", np.max(np.abs(est_riesz)))
                        
            elif method == "RKHS_GRR":
                self.model = RKHS_GRR()
                self.model.riesz_fit(covariates_tr, treatment_tr, covariates_te, treatment_te, riesz_loss=riesz_loss, riesz_with_D=riesz_with_D, riesz_link_name=riesz_link_name, is_separate=is_separate, folds=folds, num_basis=num_basis)
                self.model.reg_fit(outcome_tr)

                est_riesz = self.model.riesz_predict()
                est_reg, est_reg_one, est_reg_zero = self.model.reg_predict_diff()

            if k == 0:
                self.DM_score = est_reg_one - est_reg_zero
                self.IPW_score = est_riesz*outcome_te
                self.AIPW_score = est_riesz*(outcome_te - est_reg) + est_reg_one - est_reg_zero
            else:
                self.DM_score = np.append(self.DM_score, est_reg_one - est_reg_zero)
                self.IPW_score = np.append(self.IPW_score, est_riesz*outcome_te)
                self.AIPW_score = np.append(self.AIPW_score, est_riesz*(outcome_te - est_reg) + est_reg_one - est_reg_zero)
        
        self.DM_est = np.mean(self.DM_score)
        self.IPW_est = np.mean(self.IPW_score)
        self.AIPW_est = np.mean(self.AIPW_score)

        self.DM_var = np.var(self.DM_score - self.DM_est)
        self.IPW_var = np.var(self.IPW_score - self.IPW_est)
        self.AIPW_var = np.var(self.AIPW_score - self.AIPW_est)

        self.DM_confband = 1.96 * np.sqrt(self.DM_var / n)
        self.IPW_confband = 1.96 * np.sqrt(self.IPW_var / n)
        self.AIPW_confband = 1.96 * np.sqrt(self.AIPW_var / n)

        self.DM_confregion = [self.DM_est - self.DM_confband, self.DM_est + self.DM_confband]
        self.IPW_confregion = [self.IPW_est - self.IPW_confband, self.IPW_est + self.IPW_confband]
        self.AIPW_confregion = [self.AIPW_est - self.AIPW_confband, self.AIPW_est + self.AIPW_confband]

        return self.DM_est, self.IPW_est, self.AIPW_est, self.DM_confregion[0], self.DM_confregion[1], self.IPW_confregion[0], self.IPW_confregion[1], self.AIPW_confregion[0], self.AIPW_confregion[1]