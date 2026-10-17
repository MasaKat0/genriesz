UV ?= uv
RUN = $(UV) run --locked --project experiments

.PHONY: lint test verify marimo-check notebooks-strip notebooks-check large-files-check install-hooks

lint:
	$(RUN) ruff check src tests tools experiments/grrexp experiments/apps experiments/run_registered.py

test:
	$(RUN) python -m pytest tests/ -q

marimo-check:
	$(RUN) marimo check experiments/apps/*.py

# Drop the NumPy RuntimeWarning spam that notebook re-runs write into stderr
# outputs. Run this after re-executing notebooks/experiments/.
notebooks-strip:
	$(RUN) python tools/strip_notebook_stderr.py

notebooks-check:
	$(RUN) python tools/strip_notebook_stderr.py --check

large-files-check:
	$(RUN) python tools/check_large_files.py

install-hooks:
	mkdir -p "$$(git rev-parse --git-dir)/hooks"
	install -m 0755 tools/hooks/pre-commit "$$(git rev-parse --git-dir)/hooks/pre-commit"
	@echo "pre-commit hook installed"

verify: lint test marimo-check large-files-check
