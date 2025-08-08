# Scripts Directory

This directory contains various scripts for benchmarks, tests, and analysis tools.

## Structure

### benchmarks/
Contains benchmark scripts and related logs:
- `run_clustering_benchmark.py` - Clustering method benchmarks
- `run_pseudotime_benchmark*.py` - Various pseudotime analysis benchmarks
- Associated log files from benchmark runs

### tests/
Contains test runner scripts:
- `run_agent_test.py` - Basic agent testing
- `test_cli_clustering.sh` - CLI clustering tests
- `run_unified_tests.py` - Unified test runner
- Associated test log files

### analysis/
Contains analysis and utility scripts:
- `analyze_debugging_reports.py` - Debug report analysis
- `analyze_expensive_requests.py` - Request cost analysis
- `detailed_token_analysis.py` - Token usage analysis
- `visualize_token_usage.py` - Token usage visualization
- `update_naming_convention.py` - Code refactoring utilities
- `update_test_signatures.py` - Test update utilities

## Usage

Most scripts can be run directly from the project root:

```bash
# Run benchmarks
python scripts/benchmarks/run_clustering_benchmark.py

# Run tests
python scripts/tests/run_agent_test.py

# Run analysis
python scripts/analysis/analyze_expensive_requests.py
```

Note: Some scripts may require specific environment variables or dependencies to be set up. Check individual script documentation for details.