.PHONY: install lint serve data

PYTHON ?= python3

install:
	$(PYTHON) -m pip install --no-deps -e .

lint:
	$(PYTHON) -m ruff check controller adapters bench training scripts engine_patches

serve:
	$(PYTHON) -m controller --config configs/study.yaml serve

data:
	$(PYTHON) scripts/fetch_mtrag.py
