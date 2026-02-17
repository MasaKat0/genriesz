Examples
========

This repository includes runnable example scripts in ``examples/`` and a set of
application-specific Jupyter notebooks (one per estimand).

Runnable scripts
----------------

From the repository root:

.. code-block:: bash

   python examples/ate_synthetic_glm.py
   python examples/ate_synthetic_glm_polynomial.py
   python examples/ate_synthetic_glm_rkhs_rff.py
   python examples/ate_synthetic_glm_rf_leaf_basis.py
   python examples/ate_synthetic_nn_matching.py

   python examples/ame_synthetic_glm.py
   python examples/att_synthetic_glm.py
   python examples/did_synthetic_glm.py


Notebooks
---------

The notebooks live in ``notebooks/`` and are also shipped into the docs tree
under ``docs/notebooks/`` for convenient download:

- :download:`ATE_end_to_end.ipynb <notebooks/ATE_end_to_end.ipynb>`
- :download:`AME_end_to_end.ipynb <notebooks/AME_end_to_end.ipynb>`
- :download:`ATT_simulation_true_value.ipynb <notebooks/ATT_simulation_true_value.ipynb>`
- :download:`DID_simulation_true_value.ipynb <notebooks/DID_simulation_true_value.ipynb>`
- :download:`LinEtAl_NN_matching_local_polynomial_replication.ipynb <notebooks/LinEtAl_NN_matching_local_polynomial_replication.ipynb>`

If you would like the notebooks rendered as HTML inside the docs, you can enable
``nbsphinx`` (or ``myst-nb``) in ``docs/conf.py``. The current configuration keeps
the documentation build lightweight.
