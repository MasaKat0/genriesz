"""Helpers for loading the IHDP semi-synthetic dataset.

The original research code was written for local experimentation and included
machine-specific absolute paths. Those have been removed; callers must provide
explicit file paths.
"""

from __future__ import annotations

import numpy as np


def convert_file(x):
    """Convert a pandas DataFrame to a float NumPy array.

    This helper is kept for backwards compatibility with older notebook code.
    """

    x = x.values
    return x.astype(float)


def load_and_format_covariates(file_path: str, delimiter: str = ",") -> np.ndarray:
    """Load and re-order IHDP covariates.

    Parameters
    ----------
    file_path:
        Path to an IHDP CSV file.
    delimiter:
        CSV delimiter.

    Returns
    -------
    x:
        Covariate matrix with binary features placed first.
    """

    data = np.loadtxt(file_path, delimiter=delimiter)

    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    contfeats = [i for i in range(25) if i not in binfeats]

    x = data[:, 5:]
    perm = binfeats + contfeats
    return x[:, perm]


def load_other_stuff(file_path: str, delimiter: str = ","):
    """Load treatment/outcome columns from an IHDP CSV file."""

    data = np.loadtxt(file_path, delimiter=delimiter)
    t = data[:, 0]
    y = data[:, 1][:, None]
    y_cf = data[:, 2][:, None]
    mu_0 = data[:, 3][:, None]
    mu_1 = data[:, 4][:, None]

    return t.reshape(-1, 1), y, y_cf, mu_0, mu_1
