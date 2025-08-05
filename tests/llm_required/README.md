# Tests Requiring OpenAI API

## Overview

This directory contains tests that require a valid OpenAI API key to run. These tests interact with the OpenAI API and will incur costs based on token usage.

## Prerequisites

Before running these tests, you must:

1. **Set up OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Understand Costs**: These tests will consume OpenAI API tokens and incur charges on your OpenAI account.

## Test Categories

### Core Agent Tests
- `test_agent_system.py` - Full end-to-end agent system test
- `test_agent_logging.py` - Agent logging with real LLM interactions
- `test_agent_logging_summary.py` - Summary logging tests
- `test_api_direct.py` - Direct API interaction tests

### Clustering Tests
- `test_clustering_benchmark_llm.py` - Benchmark clustering with LLM
- `test_clustering_quick.py` - Quick clustering tests
- `test_clustering_with_fixes.py` - Clustering with bug fixes
- `test_clustering_with_llm.py` - LLM-guided clustering
- `test_real_clustering_benchmark.py` - Real-world clustering benchmark

### Pipeline Tests
- `test_simple_pipeline.py` - Simple analysis pipeline
- `test_simple_qc.py` - Quality control pipeline
- `test_incremental_planning.py` - Incremental planning tests

### Context and Execution Tests
- `test_context_fix.py` - Context fixing tests
- `test_context_passing.py` - Context passing between nodes
- `test_docker_context.py` - Docker execution context

### Configuration Tests
- `test_config_debug.py` - Configuration debugging
- `test_node_agent_logging.py` - Node-level agent logging

### Subdirectories

#### agents/
- `test_bug_fixer_simple.py` - Bug fixer agent with LLM
- `test_llm_schema_debug.py` - LLM schema debugging

#### bug_fixer/
- `test_clustering_scanpy_scatter.py` - Scanpy scatter plot bug fixes

#### cli/
- `test_cli_structure.py` - CLI with LLM integration

## Running Tests

### Individual Test
```bash
python test_agent_system.py
```

### All Tests (Warning: High API Usage)
```bash
python -m pytest . -v
```

### With Cost Estimation
```bash
# Set a lower model for testing
export OPENAI_MODEL="gpt-3.5-turbo"
python test_clustering_quick.py
```

## Cost Management

1. **Use Cheaper Models**: Set `OPENAI_MODEL="gpt-3.5-turbo"` for testing
2. **Run Selectively**: Only run specific tests you need
3. **Use Mock Tests**: Consider using tests in the parent directory that use mock LLM services
4. **Monitor Usage**: Check your OpenAI dashboard regularly

## Alternative: Mock Tests

For development and testing without API costs, use the mock tests in the parent directory:
- `test_agent_mock.py` - Full system with mock LLM
- `test_json_mock.py` - JSON operations with mock
- `test_clustering_mock.py` - Clustering with mock LLM

## Troubleshooting

### API Key Not Found
```
Error: OPENAI_API_KEY environment variable not set
```
Solution: Export your API key as shown above

### Rate Limits
If you hit rate limits, consider:
- Adding delays between tests
- Using a lower-tier model
- Running tests sequentially rather than in parallel

### High Costs
- Check token usage in test outputs
- Use mock tests for development
- Set token limits in tests where possible