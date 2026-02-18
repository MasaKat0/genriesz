# Lin et al. replication + Local-Polynomial NN–LSIF

This example reproduces the **kNN matching / bias-corrected matching** experiment setup from
Lin et al. and adds the **local-polynomial NN–LSIF** extension described in Kato (2026).

The notebook is located at:

- `notebooks/lin_et_al_local_polynomial_nn_lsif.ipynb`

## Data

Download the replication data from Zenodo:

- https://zenodo.org/records/8322609

After extracting the archive, set `LIN_REPL_DIR` in the notebook to point to the folder containing
`data/exp_generated.feather` and `data/shadish_generated.feather`.

## What is being replicated

Lin et al. implement a **bias-corrected matching** estimator, which can be written as an
**augmented Riesz weighting (ARW)** estimator:

$$
\hat\tau_{ARW}
= \mathbb{E}_n[\hat\mu_1(X) - \hat\mu_0(X)]
+ \mathbb{E}_n\bigl[D\,\hat w_1(X)\,(Y-\hat\mu_1(X))\bigr]
- \mathbb{E}_n\bigl[(1-D)\,\hat w_0(X)\,(Y-\hat\mu_0(X))\bigr],
$$

where $\hat w_d(X)$ are *inverse-propensity* weights estimated by NN matching.

## Local-polynomial extension

The notebook also runs a local-polynomial NN–LSIF estimator of the inverse-propensity weights.
This corresponds to fitting a local polynomial density-ratio model on the **M-NN ball** around each
point of interest and extracting the intercept.
