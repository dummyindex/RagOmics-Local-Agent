# Makefile for ragomics_agent_local

.PHONY: help clean test test-all test-agents test-clustering test-cli test-unit test-integration test-verbose test-list

help:
	@echo "Available commands:"
	@echo "  make clean          - Clean all test outputs and cache files"
	@echo "  make test           - Run all tests (with cleanup)"
	@echo "  make test-quick     - Run tests without cleanup"
	@echo "  make test-agents    - Run agent tests only"
	@echo "  make test-clustering- Run clustering tests only"
	@echo "  make test-cli       - Run CLI tests only"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-verbose   - Run all tests with verbose output"
	@echo "  make test-list      - List all available tests"
	@echo "  make test-structure - Run tree structure test only"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-clean   - Remove Docker images"

# Clean test outputs
clean:
	@echo "Cleaning test outputs..."
	@python tests/run_tests.py --clean-only
	@echo "Done!"

# Run all tests
test: clean
	@python tests/run_tests.py

# Run tests without cleaning
test-quick:
	@python tests/run_tests.py --no-clean

# Run specific test categories
test-agents:
	@python tests/run_tests.py --category agents

test-clustering:
	@python tests/run_tests.py --category clustering

test-cli:
	@python tests/run_tests.py --category cli

test-unit:
	@python tests/run_tests.py --category unit

test-integration:
	@python tests/run_tests.py --category integration

# Run tests with verbose output
test-verbose:
	@python tests/run_tests.py --verbose

# List available tests
test-list:
	@python tests/run_tests.py --list

# Run specific important tests
test-structure:
	@echo "Running tree structure test..."
	@python tests/test_tree_structure.py

test-simple:
	@echo "Running simple clustering test..."
	@python tests/clustering/test_clustering_simple.py

# Docker commands
docker-build:
	@echo "Building Docker images..."
	cd docker && ./build.sh

docker-build-minimal:
	@echo "Building minimal Docker images..."
	cd docker && ./build_minimal.sh

docker-clean:
	@echo "Removing Docker images..."
	docker rmi ragomics/python-executor:latest ragomics/r-executor:latest 2>/dev/null || true
	docker rmi ragomics/python-executor:minimal ragomics/r-executor:minimal 2>/dev/null || true

# Development helpers
lint:
	@echo "Running linters..."
	@python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@python -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	@echo "Formatting code..."
	@python -m black . --line-length 100 --target-version py38

typecheck:
	@echo "Running type checks..."
	@python -m mypy . --ignore-missing-imports

# Installation
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install flake8 black mypy pytest pytest-cov

# Documentation
docs:
	@echo "Building documentation..."
	@python -c "print('Documentation building not yet configured')"

# Show project statistics
stats:
	@echo "Project Statistics:"
	@echo "==================="
	@echo "Python files: $$(find . -name '*.py' -not -path './__pycache__/*' | wc -l)"
	@echo "Test files: $$(find tests -name 'test_*.py' | wc -l)"
	@echo "Lines of code: $$(find . -name '*.py' -not -path './__pycache__/*' | xargs wc -l | tail -1)"
	@echo "==================="