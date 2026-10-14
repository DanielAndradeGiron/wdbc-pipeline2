# Atajos (Linux/macOS). En Windows use las tareas de VS Code (Ctrl+Shift+P → "Run Task").
PY ?= .venv/bin/python
CONFIG ?= configs/default.yaml

.PHONY: venv install quick run replica test test-all lint format typecheck clean

venv:
	python3 -m venv .venv && $(PY) -m pip install -U pip

install: venv
	$(PY) -m pip install -e ".[all]"

quick:
	$(PY) -m wdbc_pipeline run --config configs/quick.yaml

run:
	$(PY) -m wdbc_pipeline run --config $(CONFIG)

replica:
	$(PY) -m wdbc_pipeline run --config configs/notebook_replica.yaml

test:
	$(PY) -m pytest -m "not slow"

test-all:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests && $(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff format src tests && $(PY) -m ruff check --fix src tests

typecheck:
	$(PY) -m mypy src

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ src/*.egg-info
