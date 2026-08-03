# Details of the manuscript experiments

The notebooks use the estimators and data-generating processes in `genriesz.experiments`. This document describes what each notebook computes and how the reported quantities are formed.

## Common estimators

For ATE and ATT, the representer and outcome regression are estimated on training observations. With cross fitting, predictions are formed only for observations outside the fitting fold. The notebooks report regression adjustment, Riesz weighting, augmented Riesz weighting, and Gaussian targeted maximum likelihood estimates when the requested specification is available. Regression adjustment, Riesz weighting, and the matching comparison are reported as point estimates only: their plug-in scores are not Neyman orthogonal, so no standard error, interval, or coverage is attached to them. Inference columns are populated for the augmented and targeted estimators, whose influence-function standard errors are valid under cross-fitting.

A generalized Riesz fit uses the generator and compatible link named in the notebook. The squared link is defined for every finite dual index. UKL has no mathematical restriction on that index, but the exact exponential value must remain representable in float64. The exact BKL and BP links have restricted dual domains. A candidate is unavailable when its exact inverse link is outside the mathematical domain or cannot be represented in float64. The software does not replace it with a bounded generator or change its fitted values.

For ATT, the treated proportion is estimated from the sample. The variance calculation subtracts the corresponding treatment-share term from the influence values. The point estimate is unchanged.

For the synthetic designs, the stored true effects are population values computed once from a fixed draw of 1,000,000 observations; the true ATT averages the treatment effect with propensity weights. The reported bias, root mean squared error, and coverage therefore target the population estimand that the influence-function standard errors estimate. The IHDP notebook keeps the per-replication true effects supplied with the semi-synthetic data.

The diagnostics include the 95th percentile and maximum of the absolute fitted representer, treated and control effective sample sizes, standardized mean differences, and held-out empirical Riesz imbalance. Numerical status is part of each result table.

## Notebook 01: main simulation study

The notebook uses three data-generating processes. The first has smooth heterogeneous treatment effects and moderate overlap. The second has a nonlinear treatment index and weaker overlap. The third has correlated high-dimensional covariates with sparse confounding. The true propensity is bounded between 0.05 and 0.95 before treatment is drawn, as in the original notebook design.

Compatible squared, unnormalized Kullback--Leibler, exact binary Kullback--Leibler, and Basu power specifications are compared under the same folds. The notebook also reports deliberately incompatible loss--link specifications and a propensity-score plug-in comparison. A candidate failure remains in the result data and is excluded only from summaries that require an estimate.

## Notebook 02: IHDP

The notebook combines each IHDP training and test replication, forms the observed outcome from the supplied semi-synthetic data, and estimates ATE and ATT. The true effect for each replication is computed from the supplied conditional mean outcomes. The notebook reports root mean squared error, coverage, representer tails, and effective sample sizes across the 100 replications.

## Notebook 03: Lalonde

The notebook reads the local Lalonde CSV, expands the race categories, and uses post-treatment earnings as the outcome. ATT is the primary target; ATE is a sensitivity analysis. Because the true causal effect is not observed, the notebook reports estimates, standard errors, confidence intervals, balance, and weight concentration rather than root mean squared error.

## Notebooks 04--08: supplementary comparisons

Notebook 04 compares cross-fitted and same-sample estimates. Notebook 05 changes the number of covariates while retaining the same fitting procedure. Notebook 06 compares polynomial, kernel, random-feature, forest-leaf, and matching specifications where the estimator is defined. Notebook 07 distinguishes treatment-specific regressor balancing from a treatment-invariant covariate specification. Notebook 08 studies Kang--Schafer balance paths and compares polynomial and sinusoidal outcome functions under kernel-based dictionaries.

## Notebook 09: reference-based selection

Each replication has five rotations. Three folds train the candidate and reference estimators, one fold estimates the score differences and selection criterion, and one fold evaluates the selected score. The evaluation fold never enters selection. A separate integration sample supplies simulation-only audit quantities.

The library has 90 candidates. Candidate selection uses a reference-based upper bound on conditional score drift together with an upper variance bound. The notebook compares this rule with raw Bregman validation, a generator-independent squared Riesz criterion, absolute score drift, score variance, fixed specifications, and an infeasible oracle.

The opening rescaling table uses exact status evaluation on the held-out sample. When a restricted-domain link is undefined for any row, the table reports the valid-row count and leaves the held-out criterion missing. It does not substitute a bounded link or omit the failed row. The scaled generator supplies the correspondingly scaled dual-domain interval to `GRRGLM`, so the fitting constraints remain on the same mathematical scale as the rescaled derivative.

Grid A changes overlap without the omitted direction. Grid B adds a cosine direction to the treatment index and untreated outcome regression and calibrates its coefficient to a target bias-to-standard-error ratio. Grid C repeats the calculation with 50 correlated covariates. The complete configuration contains 24,000 replication jobs. Reporting cells require all batches listed in the manifest, so numerical output is unavailable until that configuration has finished.

## Notebook 10: coverage decomposition

The treatment index and untreated outcome contain a centered quadratic term. The correct dictionary includes that term, whereas the misspecified dictionary omits it. Strong and weak overlap are considered separately. The notebook reports bias, Monte Carlo standard deviation, mean estimated standard error, coverage, root mean squared error, held-out imbalance, maximum absolute representer, and the number of successful fits. Coverage and the error summaries condition on the successful fits; the failure counts appear in the same table.

## Notebook 11: direct failure diagnostic

This notebook evaluates the earlier held-out-imbalance and variance diagnostic without a reference score. Each replication draws two independent samples: the diagnostic sample determines the selected specification, and the evaluation sample supplies the reported squared error. It records how often each loss--link specification is available and compares the selected root mean squared error with an infeasible oracle that picks the smallest realized error among the available candidates on the evaluation sample of each replication. The result does not evaluate the reference-based theorem of notebook 09.

## Notebook 12: truncated-model sensitivity

The notebook compares exact UKL and BKL with their truncated (bounded)
variants under the weak-overlap coverage-diagnostic design. The bound is part
of the fitted model: `e_min` values of 0.01, 0.02, and 0.05 correspond to
representer magnitude caps of 100, 50, and 20, stated through
`BoundedUKLGenerator.from_propensity_bounds` and `BoundedBKLGenerator`. No
fitted value is clipped afterwards. The table reports bias, Monte Carlo
standard deviation, mean estimated standard error, coverage of the population
effect, root mean squared error, the maximum absolute representer, per-side
binding rates, and the stable count; failures remain in the denominator.
Where a bound binds the estimator targets a modified (bounded) estimand, so
the bounded rows are a target-sensitivity analysis around the exact rows.

## Figures and tables

Each notebook constructs its figures and tables after computing the corresponding result data. Plot layout, labels, and output paths are stated directly in the notebook. The package does not contain a plotting function that can silently change the manuscript figures.
