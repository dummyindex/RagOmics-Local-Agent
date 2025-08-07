#!/usr/bin/env python3
"""Manual test of R-Python conversion without LLM."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragomics_agent_local.models import (
    FunctionBlock, FunctionBlockType, NewFunctionBlock, 
    ExistingFunctionBlock, StaticConfig
)
from ragomics_agent_local.agents.main_agent import MainAgent
from ragomics_agent_local.analysis_tree_management import AnalysisTreeManager


def create_test_blocks():
    """Create test function blocks for Python->R->Python workflow."""
    
    # 1. Python QC block
    python_qc = NewFunctionBlock(
        name="quality_control_python",
        type=FunctionBlockType.PYTHON,
        description="Quality control using scanpy",
        code="""
def run(path_dict, params):
    import scanpy as sc
    import os
    
    input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    
    # Read data
    adata = sc.read_h5ad(input_file)
    print(f"Loaded data with shape: {adata.shape}")
    
    # Basic QC
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    print(f"After QC: {adata.shape}")
    
    # Save
    adata.write(output_file)
    return adata
""",
        requirements="scanpy\nanndata",
        parameters={"min_genes": 200, "min_cells": 3},
        static_config=StaticConfig(
            args=[],
            description="Quality control using scanpy",
            tag="quality_control"
        )
    )
    
    # 2. R analysis block (will trigger conversion)
    r_analysis = NewFunctionBlock(
        name="seurat_analysis",
        type=FunctionBlockType.R,
        description="Analysis using Seurat",
        code="""
run <- function(path_dict, params) {
    library(Seurat)
    
    # Check for SC matrix first
    sc_matrix_dir <- file.path(path_dict$input_dir, "_node_sc_matrix")
    if (dir.exists(sc_matrix_dir)) {
        cat("Reading from SC matrix format\\n")
        
        # Read expression matrix
        expr_file <- file.path(sc_matrix_dir, "X.mtx")
        if (file.exists(expr_file)) {
            expr_matrix <- Matrix::readMM(expr_file)
            
            # Read cell and gene names
            cell_names <- readLines(file.path(sc_matrix_dir, "obs_names.txt"))
            gene_names <- readLines(file.path(sc_matrix_dir, "var_names.txt"))
            
            # Set names
            colnames(expr_matrix) <- cell_names
            rownames(expr_matrix) <- gene_names
            
            # Create Seurat object
            seurat_obj <- CreateSeuratObject(counts = expr_matrix)
            
            cat("Created Seurat object with", ncol(seurat_obj), "cells and", nrow(seurat_obj), "features\\n")
        }
    } else {
        # Fallback to RDS file
        input_file <- file.path(path_dict$input_dir, "_node_seuratObject.rds")
        seurat_obj <- readRDS(input_file)
    }
    
    # Simple normalization
    seurat_obj <- NormalizeData(seurat_obj)
    
    # Save
    output_file <- file.path(path_dict$output_dir, "_node_seuratObject.rds")
    saveRDS(seurat_obj, output_file)
    
    cat("Saved Seurat object to:", output_file, "\\n")
    return(seurat_obj)
}
""",
        requirements="Seurat\nMatrix",
        parameters={},
        static_config=StaticConfig(
            args=[],
            description="Analysis using Seurat",
            tag="seurat_analysis"
        )
    )
    
    # 3. Python visualization (will trigger conversion back)
    python_viz = NewFunctionBlock(
        name="visualization_python",
        type=FunctionBlockType.PYTHON,
        description="Visualization using scanpy",
        code="""
def run(path_dict, params):
    import scanpy as sc
    import os
    import pandas as pd
    from pathlib import Path
    
    # Check for SC matrix first
    sc_matrix_dir = Path(path_dict["input_dir"]) / "_node_sc_matrix"
    
    if sc_matrix_dir.exists():
        print("Reading from SC matrix format")
        
        # Read the matrix
        import scipy.io
        expr_matrix = scipy.io.mmread(sc_matrix_dir / "X.mtx")
        
        # Read cell and gene names
        with open(sc_matrix_dir / "obs_names.txt") as f:
            cell_names = [line.strip() for line in f]
        with open(sc_matrix_dir / "var_names.txt") as f:
            gene_names = [line.strip() for line in f]
        
        # Create AnnData
        import anndata
        adata = anndata.AnnData(X=expr_matrix.T.tocsr())
        adata.obs_names = cell_names
        adata.var_names = gene_names
        
        print(f"Created AnnData with shape: {adata.shape}")
    else:
        # Fallback to h5ad
        input_file = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
        adata = sc.read_h5ad(input_file)
    
    # Simple analysis
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    
    # Save
    output_file = os.path.join(path_dict["output_dir"], "_node_anndata.h5ad")
    adata.write(output_file)
    
    print(f"Saved AnnData to: {output_file}")
    return adata
""",
        requirements="scanpy\nanndata\nscipy\npandas",
        parameters={},
        static_config=StaticConfig(
            args=[],
            description="Visualization using scanpy",
            tag="visualization"
        )
    )
    
    return [python_qc, r_analysis, python_viz]


def test_manual_conversion():
    """Test the conversion workflow manually."""
    print("Testing Manual R-Python Conversion")
    print("=" * 50)
    
    # Test data
    test_data = Path(__file__).parent.parent / "test_data" / "zebrafish.h5ad"
    
    if not test_data.exists():
        print(f"Test data not found: {test_data}")
        return False
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "manual_test"
        
        # Create blocks
        blocks = create_test_blocks()
        
        # Initialize agent without LLM
        agent = MainAgent(openai_api_key=None)
        
        # Create tree manually
        tree_manager = AnalysisTreeManager()
        tree = tree_manager.create_tree(
            user_request="Test R-Python conversion",
            input_data_path=str(test_data)
        )
        
        # Add root node (Python QC)
        root_node = tree_manager.add_root_node(blocks[0])
        
        if not root_node:
            print("Failed to create root node")
            return False
        
        # Execute root
        print(f"\n1. Executing: {root_node.function_block.name}")
        success = agent._execute_single_node(
            root_node, tree, test_data, output_dir, verbose=True
        )
        
        if not success:
            print("Root node failed")
            return False
        
        # Check if conversion is needed for R block
        print(f"\nChecking conversion: parent type={root_node.function_block.type}, child type={blocks[1].type}")
        print(f"Parent output_data_id: {root_node.output_data_id}")
        conversion_needed = agent._check_conversion_needed(
            root_node, blocks[1], output_dir
        )
        
        if conversion_needed:
            print(f"\n2. Adding conversion node: {conversion_needed.name}")
            conv_nodes = tree_manager.add_child_nodes(root_node.id, [conversion_needed])
            conv_node = conv_nodes[0]
            
            # Execute conversion
            success = agent._execute_single_node(
                conv_node, tree, test_data, output_dir, verbose=True
            )
            
            if success:
                parent_node = conv_node
            else:
                print("Conversion failed")
                return False
        else:
            print("\n2. No conversion needed")
            parent_node = root_node
        
        # Add R analysis node
        print(f"\n3. Executing: {blocks[1].name}")
        r_nodes = tree_manager.add_child_nodes(parent_node.id, [blocks[1]])
        r_node = r_nodes[0]
        
        success = agent._execute_single_node(
            r_node, tree, test_data, output_dir, verbose=True
        )
        
        if not success:
            print("R node failed")
            return False
        
        # Check if conversion is needed for Python block
        conversion_needed = agent._check_conversion_needed(
            r_node, blocks[2], output_dir
        )
        
        if conversion_needed:
            print(f"\n4. Adding conversion node: {conversion_needed.name}")
            conv_nodes = tree_manager.add_child_nodes(r_node.id, [conversion_needed])
            conv_node = conv_nodes[0]
            
            # Execute conversion
            success = agent._execute_single_node(
                conv_node, tree, test_data, output_dir, verbose=True
            )
            
            if success:
                parent_node = conv_node
            else:
                print("Conversion failed")
                return False
        else:
            parent_node = r_node
        
        # Add final Python node
        print(f"\n5. Executing: {blocks[2].name}")
        py_nodes = tree_manager.add_child_nodes(parent_node.id, [blocks[2]])
        py_node = py_nodes[0]
        
        success = agent._execute_single_node(
            py_node, tree, test_data, output_dir, verbose=True
        )
        
        if success:
            print("\n✓ All nodes executed successfully!")
            
            # Save tree
            tree_file = output_dir / "analysis_tree.json"
            tree_manager.save_tree(tree_file)
            
            print(f"\nTree saved to: {tree_file}")
            print(f"Total nodes: {tree.total_nodes}")
            print(f"Completed nodes: {tree.completed_nodes}")
            
            return True
        else:
            print("\n✗ Final node failed")
            return False


def main():
    """Run manual conversion test."""
    try:
        success = test_manual_conversion()
        return 0 if success else 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())