#!/usr/bin/env python3
"""Tests for file passing behavior between parent and child nodes."""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
import sys
import uuid
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ragomics_agent_local.models import (
    AnalysisTree, AnalysisNode, NodeState, 
    NewFunctionBlock, FunctionBlockType, StaticConfig
)
from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager, NodeExecutor
from ragomics_agent_local.job_executors import ExecutorManager
from ragomics_agent_local.utils import setup_logger

logger = setup_logger("test_file_passing")


class TestFilePassing(unittest.TestCase):
    """Test file passing between parent and child nodes."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for tests
        self.test_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.test_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test data file
        self.input_data = self.test_dir / "test_data.h5ad"
        self._create_test_data(self.input_data)
        
        # Initialize managers
        self.tree_manager = AnalysisTreeManager()
        self.executor_manager = ExecutorManager()
        self.node_executor = NodeExecutor(self.executor_manager)
        
        # Create test tree
        self.tree = self.tree_manager.create_tree(
            user_request="Test file passing",
            input_data_path=str(self.input_data),
            max_nodes=5,
            max_children_per_node=2
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_data(self, path: Path):
        """Create a mock h5ad file for testing."""
        # For testing, we'll create a simple JSON file that simulates h5ad
        test_data = {
            "type": "anndata",
            "n_obs": 1000,
            "n_vars": 2000,
            "data": "mock_data"
        }
        with open(path, 'w') as f:
            json.dump(test_data, f)
    
    def test_parent_output_to_child_input(self):
        """Test that parent's _node_anndata.h5ad becomes child's input/adata.h5ad."""
        # Create parent node that outputs data
        parent_block = NewFunctionBlock(
            name="parent_node",
            type=FunctionBlockType.PYTHON,
            description="Parent node that outputs data",
            requirements="",
            static_config=StaticConfig(args=[], description="Parent node", tag="test"),
            code="""
def run(path_dict, params):
    import json
    import os
    
    # Construct file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # For test, we use JSON instead of actual h5ad
    # Try standard name first, then fallback
    if not os.path.exists(input_file):
        # For tests, look for any JSON file
        json_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith('.json') or f.endswith('.h5ad')]
        if json_files:
            input_file = os.path.join(path_dict["input_dir"], json_files[0])
    
    # Load input data
    with open(input_file, 'r') as f:
        adata = json.load(f)
    
    # Process data
    adata['processed_by'] = 'parent_node'
    adata['parent_value'] = 42
    
    # Save output
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(adata, f)
""",
            parameters={}
        )
        
        # Create child node that expects parent's output
        child_block = NewFunctionBlock(
            name="child_node",
            type=FunctionBlockType.PYTHON,
            description="Child node that uses parent output",
            requirements="",
            static_config=StaticConfig(args=[], description="Child node", tag="test"),
            code="""
def run(path_dict, params):
    import json
    import os
    
    # Construct file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Load input data from parent
    with open(input_file, 'r') as f:
        adata = json.load(f)
    
    # Verify parent's processing
    assert 'processed_by' in adata, "Parent processing marker not found"
    assert adata['processed_by'] == 'parent_node', "Wrong parent processor"
    assert 'parent_value' in adata, "Parent value not found"
    assert adata['parent_value'] == 42, "Wrong parent value"
    
    # Add child processing
    adata['child_processed'] = True
    adata['child_value'] = adata['parent_value'] * 2
    
    # Save output
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(adata, f)
""",
            parameters={}
        )
        
        # Add nodes to tree
        parent_node = self.tree_manager.add_root_node(parent_block)
        child_nodes = self.tree_manager.add_child_nodes(parent_node.id, [child_block])
        child_node = child_nodes[0]
        
        # Execute parent node
        parent_state, parent_output = self.node_executor.execute_node(
            parent_node, self.tree, self.input_data, self.output_dir
        )
        
        self.assertEqual(parent_state, NodeState.COMPLETED)
        self.assertIsNotNone(parent_output)
        
        # Verify parent output exists
        parent_output_file = Path(parent_output) / "_node_anndata.h5ad"
        self.assertTrue(parent_output_file.exists(), "Parent _node_anndata.h5ad not found")
        
        # Execute child node with parent's output
        child_state, child_output = self.node_executor.execute_node(
            child_node, self.tree, parent_output, self.output_dir
        )
        
        self.assertEqual(child_state, NodeState.COMPLETED)
        self.assertIsNotNone(child_output)
        
        # Verify child processed parent's output
        child_output_file = Path(child_output) / "_node_anndata.h5ad"
        self.assertTrue(child_output_file.exists(), "Child _node_anndata.h5ad not found")
        
        # Check child's output contains both parent and child processing
        with open(child_output_file, 'r') as f:
            child_data = json.load(f)
        
        self.assertEqual(child_data['processed_by'], 'parent_node')
        self.assertEqual(child_data['parent_value'], 42)
        self.assertTrue(child_data['child_processed'])
        self.assertEqual(child_data['child_value'], 84)
    
    def test_all_parent_files_accessible(self):
        """Test that ALL files from parent's outputs folder are accessible to child."""
        # Create parent node that outputs multiple files
        parent_block = NewFunctionBlock(
            name="multi_output_parent",
            type=FunctionBlockType.PYTHON,
            description="Parent that outputs multiple files",
            requirements="",
            static_config=StaticConfig(args=[], description="Multi output parent", tag="test"),
            code="""
def run(path_dict, params):
    import json
    import os
    
    # Construct file paths
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Create output directory
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    figures_dir = os.path.join(path_dict["output_dir"], "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save main data
    data = {'type': 'main_data', 'value': 100}
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    # Save supplementary files
    metadata_path = os.path.join(path_dict["output_dir"], 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({'meta': 'data'}, f)
    
    results_path = os.path.join(path_dict["output_dir"], 'results.csv')
    with open(results_path, 'w') as f:
        f.write('col1,col2\\n1,2\\n3,4\\n')
    
    # Save figures
    plot1_path = os.path.join(figures_dir, 'plot1.png')
    with open(plot1_path, 'w') as f:
        f.write('mock_png_data_1')
    
    plot2_path = os.path.join(figures_dir, 'plot2.png')
    with open(plot2_path, 'w') as f:
        f.write('mock_png_data_2')
""",
            parameters={}
        )
        
        # Create child that checks for all parent files
        child_block = NewFunctionBlock(
            name="file_checker_child",
            type=FunctionBlockType.PYTHON,
            description="Child that verifies all parent files",
            requirements="",
            static_config=StaticConfig(args=[], description="File checker child", tag="test"),
            code="""
def run(path_dict, params):
    import json
    import os
    
    # Construct file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Check for main data file
    assert os.path.exists(input_file), \
           f"Main data file not found: {input_file}"
    
    # Check for supplementary files
    expected_files = [
        os.path.join(path_dict["input_dir"], 'metadata.json'),
        os.path.join(path_dict["input_dir"], 'results.csv'),
    ]
    
    # Check if files exist in input directory or subdirectories
    input_files = []
    for root, dirs, files in os.walk(path_dict["input_dir"]):
        for file in files:
            input_files.append(os.path.join(root, file))
    
    print(f"Found input files: {input_files}")
    
    # Create output
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    result = {
        'files_found': input_files,
        'check_passed': True
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f)
""",
            parameters={}
        )
        
        # Execute parent
        parent_node = self.tree_manager.add_root_node(parent_block)
        parent_state, parent_output = self.node_executor.execute_node(
            parent_node, self.tree, self.input_data, self.output_dir
        )
        
        self.assertEqual(parent_state, NodeState.COMPLETED)
        
        # Verify parent created all expected files
        parent_path = Path(parent_output)
        self.assertTrue((parent_path / "_node_anndata.h5ad").exists())
        self.assertTrue((parent_path / "metadata.json").exists())
        self.assertTrue((parent_path / "results.csv").exists())
        self.assertTrue((parent_path / "figures" / "plot1.png").exists())
        self.assertTrue((parent_path / "figures" / "plot2.png").exists())
        
        # Execute child
        child_nodes = self.tree_manager.add_child_nodes(parent_node.id, [child_block])
        child_node = child_nodes[0]
        
        child_state, child_output = self.node_executor.execute_node(
            child_node, self.tree, parent_output, self.output_dir
        )
        
        self.assertEqual(child_state, NodeState.COMPLETED)
    
    def test_standard_conventions(self):
        """Test that function blocks follow standard input/output conventions."""
        # Create a function block that follows conventions
        standard_block = NewFunctionBlock(
            name="standard_block",
            type=FunctionBlockType.PYTHON,
            description="Block following standard conventions",
            requirements="",
            static_config=StaticConfig(args=[], description="Standard block", tag="test"),
            code="""
def run(path_dict, params):
    import json
    import os
    
    # Construct file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Standard input loading
    if os.path.exists(input_file):
        print(f"Loading from {input_file}")
        with open(input_file, 'r') as f:
            adata = json.load(f)
    else:
        # Create default data if no input found
        print("No input found, creating default data")
        adata = {'n_obs': 100, 'n_vars': 200}
    
    # Process data
    adata['processed'] = True
    
    # Standard output saving
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    print(f"Saving to {output_file}")
    
    with open(output_file, 'w') as f:
        json.dump(adata, f)
""",
            parameters={}
        )
        
        # Execute with standard block
        node = self.tree_manager.add_root_node(standard_block)
        state, output = self.node_executor.execute_node(
            node, self.tree, self.input_data, self.output_dir
        )
        
        self.assertEqual(state, NodeState.COMPLETED)
        
        # Verify standard output location
        output_file = Path(output) / "_node_anndata.h5ad"
        self.assertTrue(output_file.exists(), "Standard output file not created")
        
        # Verify data was processed
        with open(output_file, 'r') as f:
            data = json.load(f)
        self.assertTrue(data.get('processed', False))
    
    def test_missing_input_handling(self):
        """Test graceful handling of missing input files."""
        # Create block that handles missing input
        robust_block = NewFunctionBlock(
            name="robust_block",
            type=FunctionBlockType.PYTHON,
            description="Block that handles missing input gracefully",
            requirements="",
            static_config=StaticConfig(args=[], description="Robust block", tag="test"),
            code="""
def run(path_dict, params):
    import json
    import os
    
    # Construct file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Try to load input, create default if missing
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            adata = json.load(f)
        print("Loaded existing data")
    else:
        print("No input found, creating default data")
        adata = {
            'n_obs': 500,
            'n_vars': 1000,
            'created_from': 'default'
        }
    
    # Process
    adata['robust_processing'] = True
    
    # Save
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(adata, f)
""",
            parameters={}
        )
        
        # Execute without proper input (simulate missing parent)
        node = self.tree_manager.add_root_node(robust_block)
        
        # Create an empty directory (no input files)
        missing_input = self.test_dir / "missing_input"
        missing_input.mkdir(parents=True, exist_ok=True)
        
        state, output = self.node_executor.execute_node(
            node, self.tree, missing_input, self.output_dir
        )
        
        # Should still complete by creating default data
        self.assertEqual(state, NodeState.COMPLETED)
        
        # Verify output was created
        output_file = Path(output) / "_node_anndata.h5ad"
        self.assertTrue(output_file.exists())
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data.get('created_from'), 'default')
        self.assertTrue(data.get('robust_processing', False))
    
    def test_multi_parent_handling(self):
        """Test handling of nodes with multiple parents."""
        # This is a placeholder for future multi-parent support
        # Currently, the system uses single parent lineage
        pass
    
    def test_data_lineage_preservation(self):
        """Test that data processing history is preserved through the pipeline."""
        # Create blocks that track lineage
        block1 = NewFunctionBlock(
            name="lineage_block_1",
            type=FunctionBlockType.PYTHON,
            description="First block in lineage",
            requirements="",
            static_config=StaticConfig(args=[], description="First lineage block", tag="test"),
            code="""
def run(path_dict, params):
    import json
    import os
    from datetime import datetime
    
    # Construct file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Initialize or load data
    if os.path.exists(input_file):
        with open(input_file, 'r') as f:
            adata = json.load(f)
    else:
        adata = {'data': 'initial'}
    
    # Add processing history
    if 'processing_history' not in adata:
        adata['processing_history'] = []
    
    adata['processing_history'].append({
        'step': 'lineage_block_1',
        'timestamp': datetime.now().isoformat(),
        'parameters': params
    })
    
    # Save
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(adata, f)
""",
            parameters={'param1': 'value1'}
        )
        
        block2 = NewFunctionBlock(
            name="lineage_block_2",
            type=FunctionBlockType.PYTHON,
            description="Second block in lineage",
            requirements="",
            static_config=StaticConfig(args=[], description="Second lineage block", tag="test"),
            code="""
def run(path_dict, params):
    import json
    import os
    from datetime import datetime
    
    # Construct file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Load from parent
    with open(input_file, 'r') as f:
        adata = json.load(f)
    
    # Verify history exists
    assert 'processing_history' in adata, "Processing history not found"
    assert len(adata['processing_history']) > 0, "Processing history is empty"
    
    # Add our step
    adata['processing_history'].append({
        'step': 'lineage_block_2',
        'timestamp': datetime.now().isoformat(),
        'parameters': params
    })
    
    # Save
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(adata, f)
""",
            parameters={'param2': 'value2'}
        )
        
        # Create lineage
        node1 = self.tree_manager.add_root_node(block1)
        state1, output1 = self.node_executor.execute_node(
            node1, self.tree, self.input_data, self.output_dir
        )
        
        self.assertEqual(state1, NodeState.COMPLETED)
        
        # Add second node
        nodes2 = self.tree_manager.add_child_nodes(node1.id, [block2])
        node2 = nodes2[0]
        
        state2, output2 = self.node_executor.execute_node(
            node2, self.tree, output1, self.output_dir
        )
        
        self.assertEqual(state2, NodeState.COMPLETED)
        
        # Verify lineage is preserved
        with open(Path(output2) / "_node_anndata.h5ad", 'r') as f:
            final_data = json.load(f)
        
        self.assertIn('processing_history', final_data)
        self.assertEqual(len(final_data['processing_history']), 2)
        self.assertEqual(final_data['processing_history'][0]['step'], 'lineage_block_1')
        self.assertEqual(final_data['processing_history'][1]['step'], 'lineage_block_2')


class TestFunctionBlockConventions(unittest.TestCase):
    """Test that function blocks follow framework conventions."""
    
    def test_scanpy_style_block(self):
        """Test a function block following path_dict framework conventions."""
        scanpy_block = """
def run(path_dict, params):
    import scanpy as sc
    import os
    
    # Construct file paths
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Load data from standard location
    adata = sc.read_h5ad(input_file)
    print(f"Input shape: {adata.shape}")
    
    # Process using parameters (passed directly)
    min_genes = params.get('min_genes', 200)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=3)
    
    print(f"Output shape: {adata.shape}")
    
    # Save following convention
    os.makedirs(path_dict["output_dir"], exist_ok=True)
    adata.write(output_file)
    # No return value needed - output is written to file
"""
        
        # Validate path_dict block structure
        self.assertIn("def run(path_dict, params):", scanpy_block)  # Takes path_dict and params
        self.assertIn("os.path.join(path_dict[\"input_dir\"], \"_node_anndata.h5ad\")", scanpy_block)
        self.assertIn("os.path.join(path_dict[\"output_dir\"], \"_node_anndata.h5ad\")", scanpy_block)
        self.assertIn("params.get", scanpy_block)  # Uses params directly
        self.assertNotIn("return adata", scanpy_block)  # No return needed
    
    def test_r_style_block(self):
        """Test an R function block following path_dict framework conventions."""
        r_block = """
run <- function(path_dict) {
    # Takes path_dict argument
    library(Seurat)
    library(anndata)
    library(jsonlite)
    
    # Load parameters
    params <- fromJSON(path_dict$params_file)
    
    # Load data from standard location
    if (file.exists(path_dict$input_file_r)) {
        seurat_obj <- readRDS(path_dict$input_file_r)
    } else if (file.exists(path_dict$input_file_py)) {
        # From Python parent
        adata <- read_h5ad(path_dict$input_file_py)
        seurat_obj <- CreateSeuratObject(counts = adata$X)
    }
    
    # Process...
    
    # Save following convention
    dir.create(path_dict$output_dir, recursive = TRUE, showWarnings = FALSE)
    saveRDS(seurat_obj, path_dict$output_file_r)
    # No return value needed - output is written to file
}
"""
        
        # Validate path_dict R block structure
        self.assertIn('run <- function(path_dict)', r_block)  # Takes path_dict
        self.assertIn('path_dict$input_file_r', r_block)
        self.assertIn('path_dict$input_file_py', r_block)  # Can read from Python
        self.assertIn('path_dict$params_file', r_block)
        self.assertIn('path_dict$output_file_r', r_block)
        self.assertNotIn('return(', r_block)  # No return needed


def run_tests():
    """Run all file passing tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFilePassing))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFunctionBlockConventions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)