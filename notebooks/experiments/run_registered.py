"""Start a registered run of E-12..E-24 (see grrexp/runner.py).

python3 notebooks/experiments/run_registered.py E-16 stage0 14_E16_exact_examples.ipynb
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grrexp.runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
