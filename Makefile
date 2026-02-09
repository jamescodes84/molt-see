# Use venv Python if available, otherwise system python3
VENV_PYTHON := $(wildcard venv/bin/python3)
PYTHON ?= $(if $(VENV_PYTHON),venv/bin/python3,python3)

.PHONY: help install run run-debug test test-unit lint clean demo viewer viewer-expanded stop status

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "%-15s %s\n", $$1, $$2}'

install:  ## Install dependencies and moltsee CLI
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

run:  ## Start vision pipeline (moltsee start)
	$(PYTHON) -m cli start

run-debug:  ## Start with debug logging
	$(PYTHON) -m cli start --log-level DEBUG

stop:  ## Stop all Molt-See processes
	$(PYTHON) -m cli stop

status:  ## Show pipeline status
	$(PYTHON) -m cli status

test:  ## Run tests with coverage
	$(PYTHON) -m pytest open_eyes/tests/ -v --cov=open_eyes/src --cov-report=term-missing

test-unit:  ## Run unit tests only
	$(PYTHON) -m pytest open_eyes/tests/unit/ -v

lint:  ## Run linter
	$(PYTHON) -m ruff check src/ open_eyes/src/

viewer:  ## Launch camera viewer
	$(PYTHON) -m cli viewer

viewer-expanded:  ## Launch camera viewer in expanded mode
	$(PYTHON) -m cli viewer --mode expanded

demo:  ## Run camera demo
	$(PYTHON) open_eyes/scripts/demo_camera.py

clean:  ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov .coverage .pytest_cache
