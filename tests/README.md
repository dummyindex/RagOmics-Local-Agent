# Test Suite for Ragomics Agent Local

This directory contains all tests for the ragomics_agent_local project.

## Important Note on Test Organization

Tests are organized into two main categories:

1. **Mock Tests** (in main tests/ directory) - Use mock LLM services, no API costs
2. **LLM-Required Tests** (in llm_required/ subdirectory) - Require OpenAI API key and incur costs

⚠️ **WARNING**: Tests in the `llm_required/` directory will consume OpenAI API tokens and incur charges!

## Directory Structure

```
tests/
├── agents/                      # Agent-specific tests (mock)
├── bug_fixer/                   # Bug fixer tests (mock)
├── cli/                         # CLI tests (mock)
├── clustering/                  # Clustering tests (mock)
├── helpers/                     # Test helper utilities
├── openai_integration/          # OpenAI-specific integration tests
├── llm_required/                # ⚠️ Tests requiring real OpenAI API
│   ├── agents/                  # Agent tests with LLM
│   ├── bug_fixer/              # Bug fixer tests with LLM
│   ├── cli/                    # CLI tests with LLM
│   └── ...                     # Other LLM-required tests
└── test_*.py                    # General mock tests
```

## Mock Tests (No API Costs)

### Core Mock Tests
- `test_agent_mock.py` - Full agent system with mock LLM
- `test_json_mock.py` - JSON operations with mock
- `test_clustering_mock.py` - Clustering with mock LLM
- `test_manual_tree.py` - Manual tree construction (no LLM)
- `test_job_history.py` - Job history tracking (no LLM)

### Agent Tests (Mock)
- `agents/test_bug_fixer_comprehensive.py` - Comprehensive bug fixer tests
- `agents/test_bug_fixer_scfates.py` - scFates-specific bug fixes
- `agents/test_main_agent_scvelo.py` - scVelo integration tests
- `agents/test_main_agent_simple.py` - Simple agent workflows

### Framework Tests
- `test_parallel_execution.py` - Parallel execution framework
- `test_file_passing.py` - File passing between nodes
- `test_tree_structure.py` - Analysis tree structure
- `test_output_verification.py` - Output structure validation

## LLM-Required Tests (API Costs)

See [llm_required/README.md](llm_required/README.md) for details on tests that require OpenAI API.

### Quick Overview
- Full system tests with real LLM
- Clustering benchmarks
- Bug fixing with LLM assistance
- Dynamic function generation
- Context passing and fixing

## Running Tests

### Running Mock Tests (Recommended for Development)
```bash
# Run all mock tests
python -m pytest . -v --ignore=llm_required --ignore=openai_integration

# Run specific mock test
python test_agent_mock.py

# Run agent mock tests
python -m pytest agents/ -v
```

### Running LLM Tests (Costs Apply)
```bash
# First set your API key
export OPENAI_API_KEY="your-key"

# Run a specific LLM test
python llm_required/test_agent_system.py

# Run all LLM tests (NOT RECOMMENDED - high cost)
python -m pytest llm_required/ -v
```

### Running OpenAI Integration Tests
```bash
# These are specifically for testing OpenAI integration
python -m pytest openai_integration/ -v
```

## Test Data

All tests use sample datasets from:
- `/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad`
- `test_data/` directory for smaller test files

## Output Structure

Tests create outputs in `test_outputs/` with the following structure:
```
test_outputs/
└── test_name_timestamp/
    ├── analysis_tree.json
    └── tree_id/
        ├── nodes/
        │   └── node_id/
        │       ├── function_block/
        │       ├── jobs/
        │       └── outputs/
        └── main_task/
```

## Best Practices

1. **Start with Mock Tests**: Use mock tests for development to avoid API costs
2. **Test Incrementally**: Run individual tests rather than full suites
3. **Monitor Costs**: If using LLM tests, monitor your OpenAI dashboard
4. **Use Cheaper Models**: Set `OPENAI_MODEL="gpt-3.5-turbo"` for testing
5. **Clean Up**: Run `clean_test_outputs.sh` periodically to remove old test outputs

## Troubleshooting

### Import Errors
Ensure you're running from the correct directory or have installed the package:
```bash
pip install -e .
```

### Docker Errors
Ensure Docker is running and you have the required images:
```bash
cd docker && ./build_minimal.sh
```

### API Key Errors
For LLM tests, ensure your API key is set:
```bash
export OPENAI_API_KEY="sk-..."
```

## Contributing

When adding new tests:
1. Place mock tests in the appropriate directory
2. Place LLM-required tests in `llm_required/`
3. Update this README
4. Add clear documentation about test requirements