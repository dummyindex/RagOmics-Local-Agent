#!/usr/bin/env python3
"""Core validation tests for analysis tree and function block system."""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragomics_agent_local.models import (
    AnalysisTree, AnalysisNode, NodeState, GenerationMode,
    NewFunctionBlock, ExistingFunctionBlock, FunctionBlockType, 
    StaticConfig, Arg
)
from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager, NodeExecutor
from ragomics_agent_local.job_executors import ExecutorManager
from ragomics_agent_local.utils import setup_logger

logger = setup_logger(__name__)


class CoreValidator:
    """Core validation for analysis tree and function blocks."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize validator."""
        self.output_dir = output_dir or Path("test_outputs") / "core_validation"
        self.tree_manager = AnalysisTreeManager()
        self.executor_manager = ExecutorManager()
        self.node_executor = NodeExecutor(self.executor_manager)
        self.results = []
        
    def setup(self):
        """Setup test environment."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test data
        self.test_data = self._create_test_data()
        return self.test_data
    
    def _create_test_data(self) -> Path:
        """Create test data file."""
        import numpy as np
        try:
            import scanpy as sc
            data = np.random.randn(100, 50)
            adata = sc.AnnData(data)
            adata.obs_names = [f"cell_{i}" for i in range(100)]
            adata.var_names = [f"gene_{i}" for i in range(50)]
            
            # Add some metadata
            adata.obs['group'] = np.random.choice(['A', 'B', 'C'], 100)
            adata.obs['batch'] = np.random.choice([1, 2], 100)
            
            data_file = self.output_dir / "test_data.h5ad"
            adata.write(data_file)
            return data_file
        except ImportError:
            # Fallback to CSV
            import pandas as pd
            data = np.random.randn(100, 50)
            df = pd.DataFrame(data)
            data_file = self.output_dir / "test_data.csv"
            df.to_csv(data_file, index=False)
            return data_file
    
    def validate_tree_creation(self) -> bool:
        """Validate analysis tree creation."""
        print("\n1. Validating Tree Creation")
        print("-" * 40)
        
        try:
            # Test different generation modes
            modes = [GenerationMode.ONLY_NEW, GenerationMode.ONLY_EXISTING, GenerationMode.MIXED]
            
            for mode in modes:
                tree = self.tree_manager.create_tree(
                    user_request=f"Test tree with {mode.value} mode",
                    input_data_path=str(self.test_data),
                    max_nodes=10,
                    max_children_per_node=3,
                    generation_mode=mode
                )
                
                assert tree is not None, f"Failed to create tree with {mode.value}"
                assert tree.id, "Tree ID not generated"
                assert tree.user_request, "User request not stored"
                assert tree.generation_mode == mode, "Generation mode mismatch"
                
                print(f"   ✓ Created tree with {mode.value} mode")
            
            self.results.append(("Tree Creation", True))
            return True
            
        except Exception as e:
            print(f"   ✗ Tree creation failed: {e}")
            self.results.append(("Tree Creation", False))
            return False
    
    def validate_function_blocks(self) -> bool:
        """Validate function block creation and types."""
        print("\n2. Validating Function Blocks")
        print("-" * 40)
        
        try:
            # Test Python block
            python_block = NewFunctionBlock(
                name="python_test",
                type=FunctionBlockType.PYTHON,
                description="Test Python block",
                code='''
def run(adata, **parameters):
    """Test function."""
    import scanpy as sc
    print(f"Processing {adata.shape[0]} cells")
    sc.pp.filter_cells(adata, min_genes=parameters.get('min_genes', 200))
    return adata
''',
                requirements="scanpy",
                parameters={"min_genes": 100},
                static_config=StaticConfig(
                    args=[
                        Arg(name="min_genes", value_type="int", 
                            description="Minimum genes per cell", 
                            optional=True, default_value=200)
                    ],
                    description="Test Python function",
                    tag="test"
                )
            )
            
            assert python_block.name == "python_test"
            assert python_block.type == FunctionBlockType.PYTHON
            assert python_block.static_config is not None
            assert len(python_block.static_config.args) == 1
            print("   ✓ Python function block created")
            
            # Test R block
            r_block = NewFunctionBlock(
                name="r_test",
                type=FunctionBlockType.R,
                description="Test R block",
                code='''
run <- function(adata, ...) {
    print(paste("Processing", nrow(adata), "cells"))
    return(adata)
}
''',
                requirements="",
                parameters={},
                static_config=StaticConfig(
                    args=[],
                    description="Test R function",
                    tag="test"
                )
            )
            
            assert r_block.name == "r_test"
            assert r_block.type == FunctionBlockType.R
            print("   ✓ R function block created")
            
            # Test ExistingFunctionBlock
            existing_block = ExistingFunctionBlock(
                name="existing_test",
                type=FunctionBlockType.PYTHON,
                description="Existing block reference",
                function_block_id="test-block-id",
                version_id="v1.0",
                static_config=StaticConfig(
                    args=[],
                    description="Existing test function",
                    tag="test"
                )
            )
            
            assert existing_block.name == "existing_test"
            assert existing_block.function_block_id == "test-block-id"
            assert existing_block.new == False
            print("   ✓ Existing function block reference created")
            
            self.results.append(("Function Blocks", True))
            return True
            
        except Exception as e:
            print(f"   ✗ Function block validation failed: {e}")
            self.results.append(("Function Blocks", False))
            return False
    
    def validate_node_operations(self) -> bool:
        """Validate node addition and manipulation."""
        print("\n3. Validating Node Operations")
        print("-" * 40)
        
        try:
            # Create tree
            tree = self.tree_manager.create_tree(
                user_request="Test node operations",
                input_data_path=str(self.test_data),
                max_nodes=5
            )
            
            # Create function block
            block = self._create_simple_block("node_test")
            
            # Add root node
            root = self.tree_manager.add_root_node(block)
            assert root is not None, "Failed to add root node"
            assert root.id in tree.nodes, "Root node not in tree"
            assert root.parent_id is None, "Root node has parent"
            assert root.level == 0, "Root node level should be 0"
            print("   ✓ Root node added")
            
            # Add child nodes
            child_blocks = [
                self._create_simple_block(f"child_{i}") 
                for i in range(3)
            ]
            children = self.tree_manager.add_child_nodes(root.id, child_blocks)
            
            assert len(children) == 3, "Wrong number of children"
            for child in children:
                assert child.parent_id == root.id, "Child parent mismatch"
                assert child.level == 1, "Child level should be 1"
            print("   ✓ Child nodes added")
            
            # Test node state updates
            self.tree_manager.update_node_execution(
                root.id, 
                NodeState.RUNNING
            )
            assert tree.nodes[root.id].state == NodeState.RUNNING
            
            self.tree_manager.update_node_execution(
                root.id, 
                NodeState.COMPLETED,
                output_data_id="/test/output"
            )
            assert tree.nodes[root.id].state == NodeState.COMPLETED
            assert tree.nodes[root.id].output_data_id == "/test/output"
            print("   ✓ Node state updates work")
            
            # Test tree statistics
            assert tree.total_nodes == 4, "Wrong total nodes"
            assert tree.completed_nodes == 1, "Wrong completed count"
            print("   ✓ Tree statistics correct")
            
            self.results.append(("Node Operations", True))
            return True
            
        except Exception as e:
            print(f"   ✗ Node operations failed: {e}")
            self.results.append(("Node Operations", False))
            return False
    
    def validate_tree_structure(self) -> bool:
        """Validate new tree directory structure."""
        print("\n4. Validating Tree Directory Structure")
        print("-" * 40)
        
        try:
            # Create tree and node
            tree = self.tree_manager.create_tree(
                user_request="Test tree structure",
                input_data_path=str(self.test_data),
                max_nodes=2
            )
            
            block = self._create_simple_block("structure_test")
            node = self.tree_manager.add_root_node(block)
            
            # Create tree directory structure
            tree_dir = self.output_dir / f"tree_{tree.id}"
            tree_dir.mkdir(parents=True, exist_ok=True)
            
            # Create expected directories
            nodes_dir = tree_dir / "nodes"
            nodes_dir.mkdir(exist_ok=True)
            
            node_dir = nodes_dir / f"node_{node.id}"
            node_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (node_dir / "function_block").mkdir(exist_ok=True)
            (node_dir / "jobs").mkdir(exist_ok=True)
            (node_dir / "outputs").mkdir(exist_ok=True)
            (node_dir / "agent_tasks").mkdir(exist_ok=True)
            
            # Save tree metadata
            tree_metadata = {
                "id": tree.id,
                "user_request": tree.user_request,
                "created_at": datetime.now().isoformat(),
                "max_nodes": tree.max_nodes
            }
            with open(tree_dir / "tree_metadata.json", 'w') as f:
                json.dump(tree_metadata, f, indent=2)
            
            # Save analysis tree
            self.tree_manager.save_tree(tree_dir / "analysis_tree.json")
            
            # Validate structure
            assert tree_dir.exists(), "Tree directory not created"
            assert (tree_dir / "tree_metadata.json").exists(), "Tree metadata missing"
            assert (tree_dir / "analysis_tree.json").exists(), "Analysis tree missing"
            assert nodes_dir.exists(), "Nodes directory missing"
            assert node_dir.exists(), "Node directory missing"
            
            print(f"   ✓ Tree directory structure: tree_{tree.id[:8]}...")
            print(f"   ✓ Node directory structure: node_{node.id[:8]}...")
            
            # Check subdirectories
            for subdir in ["function_block", "jobs", "outputs", "agent_tasks"]:
                assert (node_dir / subdir).exists(), f"{subdir} directory missing"
                print(f"   ✓ {subdir}/ directory created")
            
            self.results.append(("Tree Structure", True))
            return True
            
        except Exception as e:
            print(f"   ✗ Tree structure validation failed: {e}")
            self.results.append(("Tree Structure", False))
            return False
    
    def validate_node_execution(self) -> bool:
        """Validate node execution with new structure."""
        print("\n5. Validating Node Execution")
        print("-" * 40)
        
        try:
            # Create tree and node
            tree = self.tree_manager.create_tree(
                user_request="Test execution",
                input_data_path=str(self.test_data),
                max_nodes=1
            )
            
            block = self._create_simple_block("execution_test")
            node = self.tree_manager.add_root_node(block)
            
            # Execute node
            state, output_path = self.node_executor.execute_node(
                node=node,
                tree=tree,
                input_path=self.test_data,
                output_base_dir=self.output_dir
            )
            
            # Validate execution
            assert state == NodeState.COMPLETED, f"Execution failed: {state}"
            assert output_path is not None, "No output path returned"
            
            # Check new structure
            tree_dir = self.output_dir / f"tree_{tree.id}"
            node_dir = tree_dir / "nodes" / f"node_{node.id}"
            
            assert tree_dir.exists(), "Tree directory not created"
            assert node_dir.exists(), "Node directory not created"
            
            # Check for job execution
            jobs_dir = node_dir / "jobs"
            job_dirs = list(jobs_dir.glob("job_*"))
            assert len(job_dirs) > 0, "No job directories created"
            
            # Check outputs
            outputs_dir = node_dir / "outputs"
            assert outputs_dir.exists(), "Outputs directory not created"
            
            # Check for output file
            output_files = list(outputs_dir.glob("*.h5ad")) + list(outputs_dir.glob("*.csv"))
            assert len(output_files) > 0, "No output files created"
            
            print(f"   ✓ Node executed successfully")
            print(f"   ✓ Job created: {job_dirs[0].name}")
            print(f"   ✓ Output saved: {output_files[0].name}")
            
            # Check symlink
            latest_link = jobs_dir / "latest"
            if latest_link.exists():
                print(f"   ✓ Latest symlink created")
            
            self.results.append(("Node Execution", True))
            return True
            
        except Exception as e:
            print(f"   ✗ Node execution failed: {e}")
            import traceback
            traceback.print_exc()
            self.results.append(("Node Execution", False))
            return False
    
    def validate_tree_persistence(self) -> bool:
        """Validate tree save and load operations."""
        print("\n6. Validating Tree Persistence")
        print("-" * 40)
        
        try:
            # Create tree with nodes
            tree = self.tree_manager.create_tree(
                user_request="Test persistence",
                input_data_path=str(self.test_data),
                max_nodes=3
            )
            
            # Add nodes
            root = self.tree_manager.add_root_node(
                self._create_simple_block("root")
            )
            children = self.tree_manager.add_child_nodes(
                root.id,
                [self._create_simple_block(f"child_{i}") for i in range(2)]
            )
            
            # Save tree
            save_path = self.output_dir / "test_tree.json"
            self.tree_manager.save_tree(save_path)
            assert save_path.exists(), "Tree file not saved"
            print("   ✓ Tree saved to JSON")
            
            # Load tree
            loaded_tree = self.tree_manager.load_tree(save_path)
            assert loaded_tree is not None, "Failed to load tree"
            assert loaded_tree.id == tree.id, "Tree ID mismatch"
            assert len(loaded_tree.nodes) == len(tree.nodes), "Node count mismatch"
            print("   ✓ Tree loaded from JSON")
            
            # Verify structure preserved
            for node_id, node in tree.nodes.items():
                loaded_node = loaded_tree.nodes.get(node_id)
                assert loaded_node is not None, f"Node {node_id} not loaded"
                assert loaded_node.function_block.name == node.function_block.name
                assert loaded_node.parent_id == node.parent_id
                assert loaded_node.level == node.level
            print("   ✓ Tree structure preserved")
            
            self.results.append(("Tree Persistence", True))
            return True
            
        except Exception as e:
            print(f"   ✗ Tree persistence failed: {e}")
            self.results.append(("Tree Persistence", False))
            return False
    
    def validate_environment(self) -> bool:
        """Validate execution environment."""
        print("\n7. Validating Environment")
        print("-" * 40)
        
        try:
            validation = self.executor_manager.validate_environment()
            
            # Check Docker
            docker_ok = validation.get('docker_available', False)
            print(f"   {'✓' if docker_ok else '✗'} Docker available")
            
            # Check Python image
            python_ok = validation.get('python_image', False)
            print(f"   {'✓' if python_ok else '✗'} Python Docker image")
            
            # Check R image (optional)
            r_ok = validation.get('r_image', False)
            print(f"   {'✓' if r_ok else '⚠'} R Docker image (optional)")
            
            # Environment is valid if Docker and at least Python image exist
            env_valid = docker_ok and python_ok
            
            self.results.append(("Environment", env_valid))
            return env_valid
            
        except Exception as e:
            print(f"   ✗ Environment validation failed: {e}")
            self.results.append(("Environment", False))
            return False
    
    def _create_simple_block(self, name: str) -> NewFunctionBlock:
        """Create a simple test function block."""
        return NewFunctionBlock(
            name=name,
            type=FunctionBlockType.PYTHON,
            description=f"Test block {name}",
            code=f'''
def run(adata, **parameters):
    """Test function {name}."""
    print(f"Executing {name}")
    print(f"Input shape: {{adata.shape}}")
    # Simple operation
    if hasattr(adata, 'obs'):
        adata.obs['{name}_processed'] = True
    return adata
''',
            requirements="scanpy",
            parameters={},
            static_config=StaticConfig(
                args=[],
                description=f"Test function {name}",
                tag="test"
            )
        )
    
    def run_all_validations(self) -> bool:
        """Run all validation tests."""
        print("="*60)
        print("CORE VALIDATION TEST SUITE")
        print(f"Timestamp: {datetime.now()}")
        print("="*60)
        
        # Setup
        self.setup()
        print(f"\nTest data: {self.test_data}")
        print(f"Output dir: {self.output_dir}")
        
        # Run validations
        validations = [
            self.validate_tree_creation,
            self.validate_function_blocks,
            self.validate_node_operations,
            self.validate_tree_structure,
            self.validate_node_execution,
            self.validate_tree_persistence,
            self.validate_environment
        ]
        
        for validation in validations:
            try:
                validation()
            except Exception as e:
                print(f"\nUnexpected error in {validation.__name__}: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        for test_name, passed in self.results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name:.<40} {status}")
        
        passed_count = sum(1 for _, p in self.results if p)
        total_count = len(self.results)
        
        print(f"\nTotal: {passed_count}/{total_count} passed")
        
        all_passed = passed_count == total_count
        print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        
        return all_passed


def main():
    """Run core validation tests."""
    validator = CoreValidator()
    success = validator.run_all_validations()
    
    # Provide guidance
    if not success:
        print("\n" + "="*60)
        print("TROUBLESHOOTING")
        print("="*60)
        print("If tests failed:")
        print("1. Ensure Docker is running")
        print("2. Build Docker images: cd docker && ./build.sh")
        print("3. Check test data exists")
        print("4. Review error messages above")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())