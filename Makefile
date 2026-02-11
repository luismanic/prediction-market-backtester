.PHONY: install lint typecheck test

install:
	uv sync --dev

lint:
	uv run ruff check .

typecheck:
	uv run basedpyright

test:
	uv run pytest
