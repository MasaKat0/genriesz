"""Run a paper experiment: python3 notebooks/experiments/run_registered.py E-25 --n-jobs 2."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grrexp.runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
