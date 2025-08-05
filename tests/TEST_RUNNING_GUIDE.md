# Test Running Guide

## Test Structure

```
tests/
├── agents/                   # Agent-specific tests (mock-based)
├── clustering/               # Clustering functionality tests
├── cli/                      # CLI interface tests
├── helpers/                  # Test helper utilities
├── openai_integration/       # OpenAI API integration tests (requires API key)
└── (main test files)         # Core functionality tests
```

## Running All Tests

### 1. Run ALL tests (including subfolders)
```bash
# From project root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=ragomics_agent_local --cov-report=html
```

### 2. Run all tests EXCEPT OpenAI integration tests
```bash
# Exclude OpenAI tests to avoid API costs
pytest tests/ --ignore=tests/openai_integration/

# Short version
pytest tests/ --ignore=tests/openai_integration/ -v
```

### 3. Run tests in parallel (faster)
```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run tests in parallel with auto-detected workers
pytest tests/ -n auto

# Run with specific number of workers
pytest tests/ -n 4
```

## Running Specific Test Categories

### Core Tests Only (no subfolders)
```bash
# Run only tests in the main tests/ directory
pytest tests/*.py
```

### Agent Tests
```bash
pytest tests/agents/
```

### Clustering Tests
```bash
pytest tests/clustering/
```

### CLI Tests
```bash
pytest tests/cli/
```

### OpenAI Integration Tests (requires API key)
```bash
# Set up environment first
export OPENAI_API_KEY="your-api-key"

# Run OpenAI tests
pytest tests/openai_integration/
```

## Running Specific Test Files

```bash
# Run a single test file
pytest tests/test_file_passing.py

# Run with specific test class
pytest tests/test_file_passing.py::TestFilePassing

# Run specific test method
pytest tests/test_file_passing.py::TestFilePassing::test_parent_output_to_child_input
```

## Useful Pytest Options

```bash
# Stop on first failure
pytest tests/ -x

# Show print statements
pytest tests/ -s

# Show local variables on failure
pytest tests/ -l

# Run only tests matching pattern
pytest tests/ -k "file_passing"

# Generate HTML coverage report
pytest tests/ --cov=ragomics_agent_local --cov-report=html
# Then open htmlcov/index.html

# Show slowest tests
pytest tests/ --durations=10

# Run tests in random order
pytest tests/ --random-order

# Mark slow tests and skip them
pytest tests/ -m "not slow"
```

## Test Markers

You can mark tests and run only specific markers:

```python
# In test file
import pytest

@pytest.mark.slow
def test_slow_operation():
    pass

@pytest.mark.requires_docker
def test_docker_operation():
    pass
```

Then run:
```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only docker tests
pytest tests/ -m "requires_docker"

# Run tests that are not slow and don't require docker
pytest tests/ -m "not slow and not requires_docker"
```

## Continuous Testing During Development

```bash
# Install pytest-watch
pip install pytest-watch

# Auto-run tests when files change
ptw tests/ --ignore=tests/openai_integration/
```

## Test Discovery Rules

pytest discovers tests using these rules:
- Files named `test_*.py` or `*_test.py`
- Classes named `Test*` (without __init__ method)
- Functions/methods named `test_*`

## Disabled Tests

Tests with `.disabled` extension are skipped:
- `test_enhanced_framework.py.disabled` - Uses deleted enhanced_executor
- `scvelo_manual_tests_v2.py.disabled` - Uses deleted enhanced_executor

To re-enable, remove the `.disabled` extension (but fix imports first).

## Environment Variables for Tests

```bash
# Set test environment variables
export PYTEST_TIMEOUT=300  # Timeout tests after 5 minutes
export OPENAI_API_KEY="..."  # For OpenAI integration tests
export OPENAI_MODEL="gpt-4o-mini"  # Optional, defaults to gpt-4o-mini
```

## Clean Test Outputs

```bash
# Clean test output directories
bash tests/clean_test_outputs.sh

# Or manually
rm -rf tests/test_outputs/
rm -rf tests/clustering/test_outputs/
```

## Common Issues and Solutions

### Issue: Docker tests fail
**Solution**: Ensure Docker is running and images are built
```bash
docker ps  # Check Docker is running
docker images | grep ragomics  # Check images exist
```

### Issue: Import errors
**Solution**: Install in development mode
```bash
pip install -e .
```

### Issue: OpenAI tests fail
**Solution**: Check API key and network
```bash
python tests/openai_integration/test_openai_quick.py
```

### Issue: Tests are slow
**Solution**: Run in parallel or skip slow tests
```bash
pytest tests/ -n auto -m "not slow"
```