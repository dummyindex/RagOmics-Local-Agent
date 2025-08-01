# Ragomics Agent Local Tests

This directory contains all tests for the Ragomics Agent Local system.

## Test Files

### test_job_history.py
Tests for job execution history tracking and output validation:
- Verifies job history is saved in `past_jobs` directory
- Checks stdout/stderr capture
- Validates job metrics CSV generation
- Confirms job_info.json creation
- Tests output data modifications

### test_manual_tree.py
Manual analysis tree construction and execution test:
- Creates a 3-node tree: QC → Normalization → RNA Velocity
- Tests tree structure persistence
- Verifies node execution order
- Checks result aggregation

### test_agent_mock.py
Mock agent test without OpenAI API:
- Demonstrates full workflow with predefined analysis tree
- Tests end-to-end system functionality
- Validates Docker container execution

### test_agent_system.py
Full agent system test (requires OpenAI API):
- Tests LLM-guided function block generation
- Validates dynamic tree expansion
- Tests debugging capabilities

## Running Tests

From the `ragomics_agent_local` directory:

```bash
# Run individual tests
python tests/test_job_history.py
python tests/test_manual_tree.py
python tests/test_agent_mock.py

# Run with OpenAI API
export OPENAI_API_KEY="your-key"
python tests/test_agent_system.py
```

## Test Data

All tests use the zebrafish.h5ad dataset from:
`/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad`

## Job History Structure

Each executed job creates the following structure:
```
node_id/
├── output_data.h5ad        # Modified data
├── figures/                # Generated plots
├── past_jobs/              # Execution history
│   └── TIMESTAMP_STATUS_JOBID/
│       ├── stdout.txt      # Standard output
│       ├── stderr.txt      # Standard error
│       ├── job_metrics.csv # Execution metrics
│       └── job_info.json   # Complete job info
└── current_job -> past_jobs/latest_success/  # Symlink
```