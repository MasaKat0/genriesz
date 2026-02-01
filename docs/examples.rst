Examples
========

This repository includes runnable example scripts in ``examples/`` and a
Jupyter notebook.

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
   python examples/ape_synthetic_glm.py


Notebook
--------

A ready-to-run end-to-end notebook is included here:

- :download:`GRR_end_to_end_examples.ipynb <notebooks/GRR_end_to_end_examples.ipynb>`

(If you would like the notebook rendered as HTML inside the docs, we can add
``nbsphinx`` and enable it in ``conf.py``. The current setup keeps the docs
build lightweight.)
