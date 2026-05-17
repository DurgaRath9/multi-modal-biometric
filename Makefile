.PHONY: install format test all clean

install:
	uv sync --all-extras

format:
	uv run black src/ tests/

test:
	uv run pytest tests/ -v --tb=short

all: format test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info .venv
