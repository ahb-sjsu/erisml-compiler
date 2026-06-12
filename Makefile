# Convenience targets that wrap the underlying Python entry points.
# Cross-platform via `python` so this works on Windows (gnu make / msys),
# Linux, and macOS without adapting per OS.

PY ?= python

.PHONY: help test lint format check reproduce-nazi-attic clean

help:
	@echo "ErisML Compiler — convenience make targets"
	@echo ""
	@echo "  make test                  Run pytest"
	@echo "  make lint                  Run ruff check"
	@echo "  make format                Run black + ruff --fix"
	@echo "  make check                 Lint + format check (no writes)"
	@echo "  make reproduce-nazi-attic  Full pipeline on the bundled example"
	@echo "                             (IR + V3 tensor + DEME verdict + audit hash"
	@echo "                              + monitor trace + delta + RLEF + HTML report"
	@echo "                              + audit bundle + summary.txt)"
	@echo "  make clean                 Remove out/ + .pytest_cache/ + __pycache__/"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

format:
	$(PY) -m black src tests
	$(PY) -m ruff check --fix src tests

check:
	$(PY) -m ruff check src tests
	$(PY) -m black --check src tests

reproduce-nazi-attic:
	$(PY) scripts/reproduce_nazi_attic.py

clean:
	rm -rf out .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
