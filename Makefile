PYTHON ?= python3

.PHONY: lint test verify notebooks-strip notebooks-check large-files-check install-hooks

lint:
	ruff check src tests tools notebooks/experiments/refsel

test:
	$(PYTHON) -m pytest tests/ -q

# Drop the NumPy RuntimeWarning spam that notebook re-runs write into stderr
# outputs. Run this after re-executing notebooks/experiments/.
notebooks-strip:
	$(PYTHON) tools/strip_notebook_stderr.py

notebooks-check:
	$(PYTHON) tools/strip_notebook_stderr.py --check

large-files-check:
	$(PYTHON) tools/check_large_files.py

install-hooks:
	mkdir -p "$$(git rev-parse --git-dir)/hooks"
	install -m 0755 tools/hooks/pre-commit "$$(git rev-parse --git-dir)/hooks/pre-commit"
	@echo "pre-commit hook installed"

verify: lint test notebooks-check large-files-check
