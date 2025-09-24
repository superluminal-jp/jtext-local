# Makefile for jtext development

.PHONY: help install install-dev test test-unit test-integration test-bdd lint format type-check security clean build docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests"
	@echo "  test-integration Run integration tests"
	@echo "  test-bdd       Run BDD tests"
	@echo "  lint           Run linting checks"
	@echo "  format         Format code with black and isort"
	@echo "  type-check     Run type checking with mypy"
	@echo "  security       Run security checks"
	@echo "  quality        Run all quality checks"
	@echo "  clean          Clean build artifacts"
	@echo "  build          Build package"
	@echo "  docs           Generate documentation"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test: test-unit test-integration test-bdd

test-unit:
	pytest tests/test_domain_entities.py tests/test_value_objects.py tests/test_use_cases.py -v --cov=jtext --cov-report=term-missing --cov-report=html

test-integration:
	pytest tests/test_integration/ -v --cov=jtext --cov-report=term-missing

test-bdd:
	pytest tests/test_bdd_steps.py -v

test-coverage:
	pytest --cov=jtext --cov-report=html --cov-report=term-missing --cov-fail-under=80

# Code Quality
lint:
	flake8 jtext/ tests/
	pylint jtext/ --disable=C0114,C0116,R0903,R0913,W0613
	radon cc jtext/ --min=B
	radon mi jtext/ --min=B

format:
	black jtext/ tests/ --line-length=88
	isort jtext/ tests/ --profile=black --line-length=88
	autoflake jtext/ tests/ --remove-all-unused-imports --remove-unused-variables --in-place

type-check:
	mypy jtext/ --strict --ignore-missing-imports

security:
	bandit -r jtext/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

quality: lint type-check security
	@echo "All quality checks completed"

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/html

docs-clean:
	rm -rf docs/_build/

# Build and packaging
build: clean
	python -m build

build-wheel:
	python -m build --wheel

build-sdist:
	python -m build --sdist

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "bandit-report.json" -delete
	find . -type f -name "safety-report.json" -delete

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete"
	@echo "Run 'make test' to verify installation"

ci: test quality
	@echo "CI pipeline completed successfully"

# Performance testing
perf-test:
	pytest tests/test_performance/ -v --durations=10

# Memory profiling
mem-profile:
	python -m memory_profiler jtext/cli.py ocr test_image.png

# Benchmarking
benchmark:
	python -m pytest tests/test_benchmark.py -v --benchmark-only

# Docker support
docker-build:
	docker build -t jtext:latest .

docker-test:
	docker run --rm jtext:latest make test

# Release
release-check: test quality
	@echo "Release checks passed"

release: release-check build
	@echo "Ready for release"

# Database operations (if needed)
db-migrate:
	@echo "No database migrations needed for this project"

db-reset:
	@echo "No database to reset for this project"

# Monitoring and observability
monitor:
	@echo "Starting monitoring dashboard"
	@echo "Access at http://localhost:3000"

logs:
	tail -f logs/jtext.log

# Environment management
env-create:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

env-activate:
	@echo "Activate virtual environment with: source venv/bin/activate"

env-deactivate:
	@echo "Deactivate virtual environment with: deactivate"

# Git hooks
git-hooks:
	@echo "Installing git hooks..."
	@echo "#!/bin/sh" > .git/hooks/pre-commit
	@echo "make pre-commit" >> .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

# Help for specific targets
help-test:
	@echo "Test targets:"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-bdd       Run BDD tests only"
	@echo "  test-coverage  Run tests with coverage report"

help-quality:
	@echo "Quality targets:"
	@echo "  lint           Run linting (flake8, pylint, radon)"
	@echo "  format         Format code (black, isort, autoflake)"
	@echo "  type-check     Run type checking (mypy)"
	@echo "  security       Run security checks (bandit, safety)"
	@echo "  quality        Run all quality checks"

help-dev:
	@echo "Development targets:"
	@echo "  dev-setup      Setup development environment"
	@echo "  pre-commit     Run pre-commit hooks"
	@echo "  ci             Run CI pipeline"
	@echo "  clean          Clean build artifacts"