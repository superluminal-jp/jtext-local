# Makefile for jtext - Japanese Text Processing CLI System

.PHONY: help install install-dev test test-cov lint format clean build dist

# Default target
help:
	@echo "jtext - Japanese Text Processing CLI System"
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install jtext in production mode"
	@echo "  install-dev  Install jtext in development mode with dev dependencies"
	@echo "  test         Run the test suite"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  lint         Run all linting checks (flake8, mypy)"
	@echo "  format       Format code with black"
	@echo "  clean        Clean build artifacts and cache files"
	@echo "  build        Build the package"
	@echo "  dist         Create distribution packages"
	@echo "  setup-env    Setup development environment"
	@echo "  check-deps   Check system dependencies"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing targets
test:
	pytest

test-cov:
	pytest --cov=jtext --cov-report=term-missing --cov-report=html

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-fast:
	pytest -m "not slow"

# Code quality targets
lint:
	@echo "Running flake8..."
	flake8 jtext/ tests/
	@echo "Running mypy..."
	mypy jtext/
	@echo "All linting checks passed!"

format:
	black jtext/ tests/
	@echo "Code formatted with black"

format-check:
	black --check jtext/ tests/

# Build targets
build:
	python -m build

dist: clean build
	@echo "Distribution packages created in dist/"

# Cleanup targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/

# Development setup
setup-env:
	@echo "Setting up development environment..."
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"
	@echo "Then run: make install-dev"

# System dependency checks
check-deps:
	@echo "Checking system dependencies..."
	@command -v tesseract >/dev/null 2>&1 || { echo "❌ Tesseract not found. Install with: brew install tesseract tesseract-lang"; exit 1; }
	@command -v ffmpeg >/dev/null 2>&1 || { echo "❌ FFmpeg not found. Install with: brew install ffmpeg"; exit 1; }
	@echo "✅ All system dependencies found"

# Development workflow
dev-setup: check-deps install-dev
	@echo "Development environment ready!"

# CI/CD targets
ci-test: lint test-cov
	@echo "CI tests completed successfully"

# Documentation targets
docs:
	@echo "Generating documentation..."
	@echo "Documentation is in README.md"

# Release targets
release-check: lint test-cov
	@echo "Release checks passed"

# Quick development commands
quick-test:
	pytest -xvs tests/test_validation.py

quick-lint:
	flake8 jtext/cli.py

# Help for specific targets
help-install:
	@echo "Installation options:"
	@echo "  make install      - Production installation"
	@echo "  make install-dev  - Development installation with dev tools"
	@echo "  make setup-env    - Create virtual environment"

help-test:
	@echo "Testing options:"
	@echo "  make test         - Run all tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make test-unit    - Run unit tests only"
	@echo "  make test-fast    - Skip slow tests"
