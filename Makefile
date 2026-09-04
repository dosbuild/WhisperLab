SHELL := /bin/sh
PYTHON ?= python3
VENV := .venv
RUN := $(VENV)/bin/python

.PHONY: setup help doctor test lint format check

setup:
	@test -x $(RUN) || $(PYTHON) -m venv $(VENV)
	@$(RUN) -m pip install --upgrade pip
	@$(RUN) -m pip install -e ".[dev]"

help:
	@$(RUN) -m whisperlab --help

doctor:
	@$(RUN) -m whisperlab doctor

test:
	@PYTHONDONTWRITEBYTECODE=1 $(RUN) -m unittest discover -s tests/unit -v

lint:
	@$(RUN) -m ruff check src tests
	@$(RUN) -m ruff format --check src tests

format:
	@$(RUN) -m ruff check --fix src tests
	@$(RUN) -m ruff format src tests

check: lint test
