#!/usr/bin/env python3
"""Test job history tracking and output validation."""

import json
import csv
from pathlib import Path
from datetime import datetime
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragomics_agent_local.models import (
    NewFunctionBlock, FunctionBlockType, StaticConfig, Arg
)
from ragomics_agent_local.job_executors import ExecutorManager
from ragomics_agent_local.utils import setup_logger

logger = setup_logger("test_job_history")


def create_test_function_block():
    """Create a simple function block for testing."""
    config = StaticConfig(
        args=[],
        description="Test function block that modifies data",
        tag="test",
        source="test"
    )
    
    code = '''
def run(path_dict, params):
    """Test function that modifies AnnData."""
    import scanpy as sc
    import numpy as np
    import os
    from datetime import datetime
    
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

    print("Starting test function...")
    print(f"Input shape: {adata.shape}")
    
    # Add a new observation
    adata.obs['test_column'] = np.random.random(adata.n_obs)
    
    # Add metadata
    adata.uns['test_metadata'] = {
        'processed': True,
        'timestamp': str(datetime.now()),
        'random_value': np.random.random()
    }
    
    # Print to stderr for testing
    import sys
    print("This is a stderr message", file=sys.stderr)
    
    print(f"Modified shape: {adata.shape}")
    print("Test function completed!")
    
    # Save output
    output_path = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    adata.write(output_path)
    
    return adata
'''
    
    return NewFunctionBlock(
        name="test_modification",
        type=FunctionBlockType.PYTHON,
        description="Test data modification",
        code=code,
        requirements="scanpy>=1.9.0\nnumpy>=1.24.0",
        parameters={},
        static_config=config
    )


def test_job_history():
    """Test job history tracking."""
    print("\n=== Testing Job History Tracking ===")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "test_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test data
        import numpy as np
        try:
            import scanpy as sc
            adata = sc.AnnData(np.random.randn(100, 50))
            input_file = Path(temp_dir) / "test_data.h5ad"
            adata.write(input_file)
        except ImportError:
            # Fallback if scanpy not available
            import pandas as pd
            df = pd.DataFrame(np.random.randn(100, 50))
            input_file = Path(temp_dir) / "test_data.csv"
            df.to_csv(input_file, index=False)
        
        # Create executor
        executor_manager = ExecutorManager()
        
        # Create test function block
        function_block = create_test_function_block()
        
        # Execute the function block
        print("\nExecuting function block...")
        result = executor_manager.execute(
            function_block=function_block,
            input_data_path=input_file,
            output_dir=output_dir,
            parameters={}
        )
        
        # Check execution result
        assert result.success, f"Execution failed: {result.error}"
        assert result.stdout, "No stdout captured"
        assert result.stderr, "No stderr captured"
        assert result.start_time is not None, "No start time recorded"
        assert result.end_time is not None, "No end time recorded"
        assert result.duration > 0, "Invalid duration"
        assert result.exit_code == 0, f"Non-zero exit code: {result.exit_code}"
        
        print(f"✓ Execution completed successfully in {result.duration:.2f}s")
        
        # Check past_jobs directory
        past_jobs_dir = output_dir / "past_jobs"
        assert past_jobs_dir.exists(), "past_jobs directory not created"
        
        # Find the job directory
        job_dirs = list(past_jobs_dir.iterdir())
        assert len(job_dirs) == 1, f"Expected 1 job directory, found {len(job_dirs)}"
        
        job_dir = job_dirs[0]
        assert "success" in job_dir.name, "Job directory should indicate success"
        assert result.job_id[:8] in job_dir.name, "Job ID not in directory name"
        
        print(f"✓ Job directory created: {job_dir.name}")
        
        # Check stdout file
        stdout_file = job_dir / "stdout.txt"
        assert stdout_file.exists(), "stdout.txt not created"
        stdout_content = stdout_file.read_text()
        assert "Starting test function..." in stdout_content
        assert "Test function completed!" in stdout_content
        print("✓ stdout.txt saved correctly")
        
        # Check stderr file
        stderr_file = job_dir / "stderr.txt"
        assert stderr_file.exists(), "stderr.txt not created"
        stderr_content = stderr_file.read_text()
        assert "This is a stderr message" in stderr_content
        print("✓ stderr.txt saved correctly")
        
        # Check job metrics CSV
        metrics_file = job_dir / "job_metrics.csv"
        assert metrics_file.exists(), "job_metrics.csv not created"
        
        with open(metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            metrics = {row['metric']: row['value'] for row in reader}
        
        assert metrics['job_id'] == result.job_id
        assert metrics['function_block_name'] == function_block.name
        assert metrics['success'] == 'True'
        assert float(metrics['duration_seconds']) > 0
        assert metrics['exit_code'] == '0'
        print("✓ job_metrics.csv saved correctly")
        
        # Check job info JSON
        job_info_file = job_dir / "job_info.json"
        assert job_info_file.exists(), "job_info.json not created"
        
        with open(job_info_file, 'r') as f:
            job_info = json.load(f)
        
        assert job_info['job_id'] == result.job_id
        assert job_info['function_block']['name'] == function_block.name
        assert job_info['execution']['success'] is True
        assert job_info['execution']['exit_code'] == 0
        print("✓ job_info.json saved correctly")
        
        # Check current_job symlink
        current_link = output_dir / "current_job"
        assert current_link.exists() or current_link.is_symlink(), "current_job symlink not created"
        print("✓ current_job symlink created")
        
        print("\n✓ All job history tests passed!")


def test_output_data_modification():
    """Test that _node_anndata.h5ad is actually modified."""
    print("\n=== Testing Output Data Modification ===")
    
    import numpy as np
    
    # Create test data first
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            import scanpy as sc
            import anndata
            adata = sc.AnnData(np.random.randn(100, 50))
            input_file = Path(temp_dir) / "test_data.h5ad"
            adata.write(input_file)
            
            # Load original data
            original_adata = anndata.read_h5ad(input_file)
            original_shape = original_adata.shape
            original_obs_columns = set(original_adata.obs.columns)
            original_uns_keys = set(original_adata.uns.keys())
            
            print(f"Original data shape: {original_shape}")
            print(f"Original obs columns: {original_obs_columns}")
            print(f"Original uns keys: {original_uns_keys}")
            
            # Create output directory
            output_dir = Path(temp_dir) / "test_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create executor
            executor_manager = ExecutorManager()
            
            # Create test function block
            function_block = create_test_function_block()
            
            # Execute the function block
            print("\nExecuting function block...")
            result = executor_manager.execute(
                function_block=function_block,
                input_data_path=input_file,
                output_dir=output_dir,
                parameters={}
            )
            
            # Check execution result
            assert result.success, f"Execution failed: {result.error}"
            assert result.output_data_path is not None, "No output data path"
            
            # Load modified data
            output_path = Path(result.output_data_path)
            assert output_path.exists(), f"Output file not found: {output_path}"
            
            modified_adata = anndata.read_h5ad(output_path)
            modified_obs_columns = set(modified_adata.obs.columns)
            modified_uns_keys = set(modified_adata.uns.keys())
            
            print(f"\nModified data shape: {modified_adata.shape}")
            print(f"Modified obs columns: {modified_obs_columns}")
            print(f"Modified uns keys: {modified_uns_keys}")
            
            # Verify modifications
            assert modified_adata.shape == original_shape, "Data shape should not change"
            
            # Check new column was added
            new_columns = modified_obs_columns - original_obs_columns
            assert 'test_column' in new_columns, "test_column not added to obs"
            print("✓ New column 'test_column' added to obs")
            
            # Check column has valid data
            assert len(modified_adata.obs['test_column']) == modified_adata.n_obs
            assert np.all(modified_adata.obs['test_column'] >= 0)
            assert np.all(modified_adata.obs['test_column'] <= 1)
            print("✓ test_column contains valid random values")
            
            # Check new metadata was added
            new_uns_keys = modified_uns_keys - original_uns_keys
            assert 'test_metadata' in new_uns_keys, "test_metadata not added to uns"
            print("✓ New metadata 'test_metadata' added to uns")
            
            # Check metadata content
            test_metadata = modified_adata.uns['test_metadata']
            assert test_metadata['processed'] is True
            assert 'timestamp' in test_metadata
            assert 'random_value' in test_metadata
            assert 0 <= test_metadata['random_value'] <= 1
            print("✓ test_metadata contains expected fields")
            
            # Verify data is actually different
            assert output_path != input_file, "Output should be a different file"
            print("✓ Output is saved to a new file")
            
            print("\n✓ All output data modification tests passed!")
        except ImportError:
            print("Skipping test - scanpy not available")
            import pytest
            pytest.skip("scanpy not available")


def main():
    """Run all tests."""
    # Use zebrafish data for testing
    input_data = Path("/Users/random/Ragomics-workspace-all/data/zebrafish.h5ad")
    
    if not input_data.exists():
        print(f"Error: Test data not found at {input_data}")
        sys.exit(1)
    
    print(f"Using test data: {input_data}")
    
    # Run tests
    test_job_history(input_data)
    test_output_data_modification(input_data)
    
    print("\n=== All Tests Passed! ===")


if __name__ == "__main__":
    main()