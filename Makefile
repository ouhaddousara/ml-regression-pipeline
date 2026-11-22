# Atelier 1 - Regression. Run every target from the repository root.
# NOTE: recipe lines MUST start with a TAB character.

SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c
PY ?= .venv/bin/python

.DEFAULT_GOAL := all
.PHONY: setup data dirs part1 part2 part3 part4 all report clean

dirs:
	mkdir -p data/raw results/logs report/figures report/screenshots report/generated

setup: dirs
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

# Downloads the datasets (needs KAGGLE_USERNAME / KAGGLE_KEY in the environment
# for the two Kaggle datasets). Falls back to manual instructions (data/README.md).
data: dirs
	$(PY) src/utils.py fetch

part1: dirs
	$(PY) src/part1_eda.py 2>&1 | tee results/logs/part1.txt

part2: dirs
	$(PY) src/part2_simple_regression.py 2>&1 | tee results/logs/part2.txt

part3: dirs
	$(PY) src/part3_multiple_regression.py 2>&1 | tee results/logs/part3.txt

part4: dirs
	$(PY) src/part4_polynomial_regression.py 2>&1 | tee results/logs/part4.txt

all: part1 part2 part3 part4

report: dirs
	cd report && latexmk -pdf -interaction=nonstopmode -outdir=build main.tex

# Removes generated artefacts only (never data/raw, never screenshots).
clean:
	rm -rf report/build
	find report/generated report/figures results -maxdepth 2 -type f \( -name '*.tex' -o -name '*.png' -o -name '*.json' -o -name '*.csv' -o -name '*.txt' \) -delete 2>/dev/null || true
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
