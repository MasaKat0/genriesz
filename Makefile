PYTHON ?= python3

.PHONY: lint test verify

lint:
	ruff check src tests

test:
	$(PYTHON) -m pytest tests/ -q

verify: lint test
