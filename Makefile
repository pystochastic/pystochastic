.PHONY: create_venv

create_venv:
	uv sync --locked --dev --all-extras

run_mypy:
	uv run mypy src tests

run_pyright:
	uv run pyright src tests

run_pylint:
	uv run pylint src tests

run_ruff:
	uv run ruff check src tests

run_tests:
	uv run pytest tests

run_coverage:
	uv run pytest \
		--disable-warnings \
		--maxfail=1 \
		--tb=short \
		--cov=src \
		--cov-report=term-missing:skip-covered \
		--cov-report=html \
		--cov-report=xml \
		-n auto \
		--dist=loadfile \
		-rsx \
		tests

build:
	uv build

publish:
	uv publish --index pypi

docs:
	poetry export --dev -f requirements.txt > docs/requirements.txt
	cd docs && \
	make html
	open docs/_build/html/index.html

clean:
	rm -rf dist

