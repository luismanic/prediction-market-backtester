.PHONY: install install-index lint typecheck test setup data-setup index data-index

SOURCE ?= all
MODE ?= all

install:
	uv sync --dev

install-index:
	uv sync --dev --group index

lint:
	uv run ruff check .

typecheck:
	uv run basedpyright

test:
	uv run pytest

setup: data-setup

data-setup:
	bash scripts/setup_data.sh

index: data-index

data-index: install-index
	uv run python scripts/index_data.py --source "$(SOURCE)" --mode "$(MODE)"
