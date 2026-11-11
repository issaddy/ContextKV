.PHONY: install test lint demo serve data

PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e '.[retrieval,evaluation,tokenizer]'

test:
	$(PYTHON) -m pytest -q tests/test_rag_dialogue.py tests/test_fact_reuse.py tests/test_study_metrics.py

lint:
	$(PYTHON) -m ruff check controller adapters bench profiling tests scripts engine_patches

demo:
	$(PYTHON) -m bench.run_sessions --config configs/demo.yaml demo

serve:
	$(PYTHON) -m controller --config configs/qwen3_32b.yaml serve

data:
	$(PYTHON) scripts/fetch_mtrag.py
