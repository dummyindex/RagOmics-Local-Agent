# Ragomics Agent Local

A command-line tool for hierarchical single-cell RNA-seq analysis using LLM-guided function blocks with Docker container isolation.

## Features

- **Hierarchical Analysis Trees**: Build and execute multi-step analysis pipelines
- **LLM-Guided Generation**: Automatically generate analysis code using OpenAI API
- **Docker Container Isolation**: Each analysis step runs in an isolated container
- **Job History Tracking**: Complete execution history with stdout/stderr capture
- **Support for Python and R**: Execute analysis in either language
- **Reproducible Workflows**: Save and reload analysis trees as JSON

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ragomics-agent-local.git
cd ragomics-agent-local

# Install the package
pip install -e .

# Build Docker images
cd docker
./build_minimal.sh  # For testing
# ./build.sh        # For full environment (larger images)
```

## Quick Start

```bash
# Set OpenAI API key (optional for manual trees)
export OPENAI_API_KEY="your-api-key"

# Run analysis
ragomics-agent analyze data.h5ad "Perform clustering and find marker genes"

# With options
ragomics-agent analyze data.h5ad "Analyze RNA velocity" \
  --max-nodes 20 \
  --output results/
```

## Project Structure

```
ragomics_agent_local/
├── agents/                    # Main orchestration logic
├── analysis_tree_management/  # Tree structure and execution
├── job_executors/            # Docker container execution
├── llm_service/              # OpenAI API integration
├── utils/                    # Utilities
├── docker/                   # Docker configurations
├── tests/                    # Test suite
└── models.py                 # Data models
```

## Testing

```bash
# Run tests from ragomics_agent_local directory
cd ragomics_agent_local

# Manual tree test (no API needed)
python tests/test_manual_tree.py

# Mock agent test (no API needed)
python tests/test_agent_mock.py

# Job history test
python tests/test_job_history.py

# Full agent test (requires OpenAI API)
python tests/test_agent_system.py
```

## Output Structure

Each analysis creates the following structure:

```
output_dir/
├── analysis_tree.json          # Complete tree structure
└── tree_id/                    # Unique tree execution
    └── node_id/                # Each analysis node
        ├── output_data.h5ad    # Modified data
        ├── figures/            # Generated plots
        ├── past_jobs/          # Execution history
        │   └── TIMESTAMP_STATUS_JOBID/
        │       ├── stdout.txt
        │       ├── stderr.txt
        │       ├── job_metrics.csv
        │       └── job_info.json
        └── current_job -> past_jobs/latest/
```

## Configuration

Configuration options in `config.py`:
- Docker image names
- Execution timeouts
- Resource limits
- Temporary directory paths

## Requirements

- Python 3.8+
- Docker
- OpenAI API key (for LLM features)

## License

[Your License]

## Contributing

[Contribution guidelines]