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
git clone https://github.com/dummyindex/RagOmics-Local-Agent.git
cd RagOmics-Local-Agent

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

## Testing

Run the test suite to verify the installation:

```bash
# Run scVelo RNA velocity tests
python tests/scvelo_manual_tests.py

# Run scFates trajectory inference test
python tests/test_scfates_successful_final.py

# Run agent tests
python tests/agents/test_main_agent_simple.py
```

Test outputs will be saved in the `test_outputs/` directory.

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
├── analysis_tree.json        # Tree definition
└── {tree_id}/
    ├── main_task/           # Orchestration logs
    └── nodes/               # Flat node structure
        └── node_{id}/
            ├── node_info.json
            ├── function_block/
            │   ├── code.py      # Generated code
            │   ├── config.json  # Block configuration
            │   └── requirements.txt
            ├── agent_tasks/     # Agent interaction logs
            ├── jobs/            # Execution history
            │   ├── job_{timestamp}_{id}/
            │   │   ├── execution_summary.json
            │   │   ├── input/
            │   │   ├── output/
            │   │   │   ├── _node_anndata.h5ad
            │   │   │   └── figures/
            │   │   └── logs/
            │   │       ├── stdout.txt
            │   │       └── stderr.txt
            │   └── latest -> job_{timestamp}_{id}
            └── outputs/         # Final node outputs
                ├── _node_anndata.h5ad
                └── figures/
```

## Configuration

Configuration options in `config.py`:
- Docker image names
- Execution timeouts
- Resource limits
- Temporary directory paths

## Requirements

- Python 3.11+
- Docker
- OpenAI API key (for LLM features)

## License

[Your License]

## Contributing

[Contribution guidelines]