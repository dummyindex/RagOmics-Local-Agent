"""Common utilities for tests."""

from pathlib import Path


def get_test_output_dir() -> Path:
    """Get the test output directory path.
    
    This ensures consistent output location regardless of where tests are run from.
    Returns the path to test_outputs/ directory in project root.
    """
    # Get the project root (ragomics_agent_local - parent of parent of this file)
    project_root = Path(__file__).parent.parent
    
    # Create test_outputs directory in project root
    test_outputs_dir = project_root / "test_outputs"
    
    # Ensure it exists
    test_outputs_dir.mkdir(exist_ok=True)
    
    return test_outputs_dir


def get_test_data_dir() -> Path:
    """Get the test data directory path.
    
    Returns the path to test_data/ directory at project root.
    """
    # Get project root (parent of parent of this file)
    project_root = Path(__file__).parent.parent
    
    # Return test_data directory
    return project_root / "test_data"