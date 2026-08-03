# Reference-based selection of a loss--link specification

This document specifies the experiment implemented by `09_reference_based_loss_link_selection.ipynb` and `genriesz.experiments.reference_selection`. The experiment evaluates the selection results in the manuscript with a finite library of generalized Riesz specifications. Candidate fitting, candidate comparison, and final score evaluation use separate observations.

## 1. Statistical questions

The experiment addresses seven questions.

| ID | Question | Reported quantity |
|---|---|---|
| E1a | Can raw Bregman objectives rank different generators? | Generator-rescaling table |
| E1b | How does reference-based selection compare with other rules on the same fitted library? | Conditional risk, root mean squared error, coverage, and selection frequencies |
| E2 | Does the candidate-bias bound cover the conditional score drift, and how conservative is it? | Lower and upper bound checks |
| E3 | How close is the selected conditional risk to the best admissible risk? | Oracle regret and the stated remainder |
| E4 | Do the confidence intervals retain coverage over a calibrated family of omitted directions? | Coverage at each bias-to-standard-error ratio and minimum coverage over the family |
| E5 | What happens when a reference allowance is too small? | Reference-drift checks and coverage by reference |
| E6 | How much longer is a bias-aware interval when the estimated bias bound is small? | Bound-to-standard-error ratio and interval-length ratio |
| E7 | Does the procedure remain computable in the high-dimensional design? | Risk, coverage, selected specifications, and numerical failures |

E1a is deterministic. Its held-out criterion is reported only when the exact link is defined for every held-out observation. A restricted-domain failure is part of the table and is not replaced by a finite value. Rescaling a restricted-domain generator also rescales its observationwise dual constraints, as required by $\partial(\kappa g)=\kappa\partial g$. E1b--E7 use the same candidate fits within each fold, so another selection rule does not require another representer fit.

## 2. Candidate specifications

A candidate is determined by a Bregman generator, its compatible link, a treatment-specific dictionary, and an `l1` penalty multiplier. The library contains the following five losses:

- squared loss;
- unnormalized Kullback--Leibler loss;
- exact binary Kullback--Leibler loss;
- Basu power loss with `omega=0.25`;
- Basu power loss with `omega=0.5`.

The dictionaries are `linear`, `second_order`, and `rich`. The penalty is

```text
lambda = c * sqrt(log(p) / n_training),
```

where `c` is one of `0`, `0.25`, `0.5`, `1`, `2`, and `4`. The Cartesian product contains 90 candidates. `BP(1)` is omitted because its branchwise curvature equals the squared generator up to scale.

The exact binary Kullback--Leibler candidate remains in the ATE library even when it is unavailable. Its dual coordinate must stay in the exact domain on training, diagnostic, and evaluation observations. A domain failure is recorded; no bounded substitute is fitted.

## 3. Sample roles

Each replication uses five rotations. In rotation `k`, fold `k` is the evaluation sample, fold `k+1` is the diagnostic sample, and the other three folds form the training sample. The roles are disjoint, and each observation enters the evaluation sample exactly once.

The training sample is used to fit every candidate, the outcome estimator, and the reference estimators. The diagnostic sample is used to estimate relative score drift, simultaneous radii, variance bounds, and the selection criterion. The selected score is evaluated only on the evaluation sample. Simulation truth is used on a separate integration sample for audit quantities and does not enter candidate selection.

## 4. Data-generating processes

### 4.1 Low-dimensional design

Let `Z` have five independent standard normal coordinates. Define

```text
h_L(Z) = 0.6 Z1 - 0.4 Z2 + 0.5 (Z3^2 - 1) + 0.3 sin(Z4),
e_0(Z) = logistic(s h_L(Z)),
mu_0(Z) = 1 + Z1 + 0.5 (Z2^2 - 1) + 0.5 sin(Z3) + 0.25 Z4 Z5,
tau(Z) = 1 + 0.5 Z1 - 0.25 Z2,
Y = mu_0(Z) + D tau(Z) + epsilon,
epsilon ~ N(0,1).
```

The average treatment effect is one. Grid A uses sample sizes 1,000 and 3,000 and overlap scales `s=0.5`, `1.5`, and `2.5`.

### 4.2 Calibrated omitted direction

Grid B adds

```text
psi(Z) = cos(2 pi Z1)
```

to both the treatment index and the untreated outcome regression. The treatment effect is unchanged. The candidate dictionaries omit `psi`, whereas the correctly specified low-dimensional reference includes it. A committed calibration table chooses the coefficient of `psi` so that a fixed benchmark candidate has a target absolute bias-to-standard-error ratio of `0.5`, `1`, `2`, or `4`. Grid B uses sample sizes 1,000 and 3,000 and overlap scale `s=1.5`.

### 4.3 High-dimensional design

The high-dimensional design has 50 Gaussian covariates with covariance `0.5^|j-l|`. Its treatment index and outcome regression contain linear terms, interactions, a sine term, a centered square, and an absolute-value term. Grid C uses sample size 3,000, overlap scales `0.75` and `2`, and calibrated bias-to-standard-error ratios `0` and `1`.

## 5. Reference estimators and allowances

The truth reference is used only to audit the finite-sample calculations. The low-dimensional experiment also fits a correctly specified series reference and a deliberately misspecified series reference. Their allowances are estimated from the coefficient covariance calculations stated in the manuscript and are checked against the simulation truth.

The high-dimensional design uses a random Fourier feature reference. Its allowance has the stated sensitivity form `c_r / sqrt(n_evaluation)`, with `c_r` in `0.25`, `0.5`, `1`, and `2`.

For candidate `a` and reference `r`, the diagnostic sample estimates the mean score difference and a simultaneous Gaussian multiplier radius. The candidate-bias bound adds the reference allowance. The minimum bound over several references is reported only when every member of the set is available.

## 6. Selection rules

All rules use the same candidate library.

| Rule | Criterion |
|---|---|
| `proposed` | Squared candidate-bias bound plus an upper variance bound divided by evaluation sample size |
| `proposed_min` | The same criterion using the minimum valid bound over the stated references |
| `bregman_cv` | Candidate-specific held-out Bregman criterion |
| `lsif_cv` | Generator-independent held-out squared Riesz criterion |
| `abs_drift` | Absolute estimated score difference without a simultaneous radius or reference allowance |
| `score_var` | Upper variance bound |
| `fixed_sq`, `fixed_ukl`, `fixed_bkl`, `fixed_bp05` | Prespecified rich-dictionary candidate with multiplier one |
| `oracle` | Conditional risk computed from simulation truth |

A failed candidate remains in the failure denominator. A rule that cannot return an estimate is reported as unavailable, and its confidence interval does not cover.

## 7. Inference

The notebook reports two intervals supported by the manuscript. `bias_aware_split` is the single-split interval that uses the selected candidate's bias bound. `conservative_cf` combines fold-specific bounds for the cross-fitted estimate. The ordinary Wald interval is reported for comparison. The field `bias_aware_pooled` is retained as a diagnostic and is not presented as a theoretically supported interval.

## 8. Numerical requirements

The experiment uses exact inverse links. A candidate is unavailable when any of the following conditions occurs:

- the optimizer does not converge;
- a coefficient or score is nonfinite;
- the KKT residual exceeds the numerical tolerance;
- the exact inverse link is outside its mathematical domain or is not representable in float64;
- a prespecified effective-sample-size restriction removes the candidate.

No candidate is replaced after failure. The estimator does not clip fitted weights after optimization. All hyperparameters of the outcome and reference estimators are fixed before a replication begins and are fit using training observations only.

The simultaneous mean and variance radii use 10,000 Gaussian multiplier draws. Grid A has 1,000 replications per scenario, Grid B has 2,000, and Grid C has 500. The full design has 18 scenarios and 24,000 replication jobs.

## 9. Output and reproducibility

Results are written in batches below `notebooks/experiments/results/reference_selection/publication`. The manifest records the scenario list, candidate specifications, numerical settings, calibration values, resolved replication counts, and batch identities. Existing batches are reused only when the manifest digest agrees with the requested configuration. Report tables and figures are constructed only after every batch file named by the manifest is present.

The seed of every data sample, fold, reference fit, multiplier draw, and integration audit is a deterministic function of its scenario and replication identifiers. Changing the number or order of worker processes therefore does not change the result.

The committed repository contains the experiment code and calibration table, but it does not report publication-scale numerical results until the complete configuration has been executed.
