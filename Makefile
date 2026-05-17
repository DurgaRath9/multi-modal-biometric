.PHONY: install lint format typecheck test all clean

install:
	uv sync --all-extras

lint:
	uv run ruff check src/ tests/

format:
	uv run black src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/ -v --tb=short

all: lint typecheck test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info .venv
