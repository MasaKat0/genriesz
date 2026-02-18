# Diagnostics

## Love plot

For treatment-effect estimands (ATE / ATT / DID), `genriesz` computes simple
covariate-balance diagnostics based on **standardized mean differences (SMDs)**.

The balance diagnostics use the absolute value of the estimated Riesz representer
as nonnegative weights.

### Balance table

After you run `grr_functional` (or a wrapper such as `grr_ate`), you can extract
the balance table:

```python
res = grr_ate(...)

# Returns a pandas DataFrame when pandas is available, otherwise a list of dicts.
tbl = res.love_plot_data()
tbl
```

### Plot

To draw a Love plot, you need `matplotlib` installed:

```python
fig, ax = res.love_plot(threshold=0.1, max_covariates=30)
```

The vertical dashed line at `threshold=0.1` is a common rule-of-thumb cutoff.

### Notes

- Love-plot diagnostics are computed only for ATE / ATT / DID.
- If you use `cross_fit=True`, the plot reflects the final cross-fitted weights
  used by the estimators.
