.PHONY: install lint test serve data

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e '.[retrieval,evaluation,tokenizer]'

lint:
	$(PYTHON) -m ruff check controller adapters bench profiling training scripts engine_patches tests

test:
	$(PYTHON) -m unittest discover -s tests -v

serve:
	$(PYTHON) -m controller --config configs/qwen3_32b.yaml serve

data:
	$(PYTHON) scripts/fetch_mtrag.py
