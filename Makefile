.PHONY: install test lint demo serve data

install:
	uv sync --extra retrieval --extra evaluation --locked

test:
	uv run --extra retrieval --extra evaluation pytest -q

lint:
	uv run ruff check controller adapters bench profiling tests scripts engine_patches

demo:
	uv run contextkv --config configs/demo.yaml demo

serve:
	uv run contextkv --config configs/qwen3_32b_cost.yaml serve

data:
	uv run python scripts/fetch_mtrag.py
