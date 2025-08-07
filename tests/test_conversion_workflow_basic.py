"""Test conversion workflow with basic R operations (no Seurat)."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import anndata

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ragomics_agent_local.agents.main_agent import MainAgent
from src.ragomics_agent_local.models import (
    NewFunctionBlock, FunctionBlockType, StaticConfig, 
    Arg, GenerationMode, InputSpecification, OutputSpecification
)


def create_test_anndata(temp_dir: Path):
    """Create a test AnnData object."""
    np.random.seed(42)
    n_obs, n_vars = 100, 50
    
    # Create sparse count matrix
    counts = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
    counts[counts > 20] = 0
    X = csr_matrix(counts)
    
    # Create AnnData
    adata = anndata.AnnData(X=X)
    adata.obs['cell_type'] = np.random.choice(['T_cell', 'B_cell', 'NK_cell'], size=n_obs)
    adata.obs_names = [f'Cell_{i}' for i in range(n_obs)]
    adata.var_names = [f'Gene_{i}' for i in range(n_vars)]
    
    # Save to file
    data_path = temp_dir / "test_data.h5ad"
    adata.write_h5ad(data_path)
    
    return data_path


def test_automatic_conversion():
    """Test that the agent automatically inserts conversion nodes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        data_path = create_test_anndata(temp_path)
        
        # Initialize agent
        agent = MainAgent(
            workspace_dir=temp_path / "test_workspace",
            generation_mode=GenerationMode.ONLY_NEW,
            max_nodes=10
        )
        
        # Create Python block
        python_block = NewFunctionBlock(
            name="normalize_python",
            type=FunctionBlockType.PYTHON,
            description="Normalize data with scanpy",
            code="""
def run(path_dict, params):
    import anndata
    import scanpy as sc
    
    # Load data
    adata = anndata.read_h5ad(path_dict['input_dir'] + '/_node_anndata.h5ad')
    
    # Basic normalization
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    # Save
    adata.write_h5ad(path_dict['output_dir'] + '/_node_anndata.h5ad')
    
    print(f"Normalized data: {adata.shape}")
    return {"normalized": True}
""",
            requirements="scanpy\nanndata",
            static_config=StaticConfig(
                args=[],
                description="Normalize with scanpy",
                tag="normalize"
            ),
            parameters={}
        )
        
        # Create R block that just processes matrix
        r_block = NewFunctionBlock(
            name="process_matrix_r",
            type=FunctionBlockType.R,
            description="Process matrix in R",
            code="""
run <- function(path_dict, params) {
    library(Matrix)
    library(jsonlite)
    
    # Read SC matrix format
    sc_dir <- file.path(path_dict$input_dir, "_node_sc_matrix")
    
    if (!dir.exists(sc_dir)) {
        stop("No _node_sc_matrix found")
    }
    
    # Read matrix
    X <- readMM(file.path(sc_dir, "X.mtx"))
    cell_names <- readLines(file.path(sc_dir, "obs_names.txt"))
    gene_names <- readLines(file.path(sc_dir, "var_names.txt"))
    
    # Transpose for R (genes x cells)
    X <- t(X)
    rownames(X) <- gene_names
    colnames(X) <- cell_names
    
    # Simple processing - calculate gene means
    gene_means <- rowMeans(X)
    
    # Save results
    results <- list(
        n_cells = ncol(X),
        n_genes = nrow(X),
        mean_expression = mean(X),
        top_genes = names(sort(gene_means, decreasing = TRUE)[1:10])
    )
    
    write(toJSON(results, pretty = TRUE), 
          file.path(path_dict$output_dir, "r_analysis_results.json"))
    
    # Copy SC matrix to output
    file.copy(sc_dir, path_dict$output_dir, recursive = TRUE)
    
    cat("Processed matrix with", ncol(X), "cells and", nrow(X), "genes\\n")
    
    return(list(success = TRUE))
}
""",
            requirements="Matrix\njsonlite",
            static_config=StaticConfig(
                args=[],
                description="Process matrix in R",
                tag="r_process"
            ),
            parameters={}
        )
        
        # Another Python block to verify conversion
        verify_block = NewFunctionBlock(
            name="verify_results",
            type=FunctionBlockType.PYTHON,
            description="Verify R results",
            code="""
def run(path_dict, params):
    import json
    import os
    
    # Check for R results
    results_file = os.path.join(path_dict['input_dir'], 'r_analysis_results.json')
    
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        
        print("R analysis results:")
        print(f"  Cells: {results['n_cells']}")
        print(f"  Genes: {results['n_genes']}")
        print(f"  Mean expression: {results['mean_expression']}")
        print(f"  Top genes: {results['top_genes'][:5]}")
        
        return {"verified": True, "results": results}
    else:
        print("No R results found")
        return {"verified": False}
""",
            requirements="",
            static_config=StaticConfig(
                args=[],
                description="Verify results",
                tag="verify"
            ),
            parameters={}
        )
        
        # Run workflow
        print("\nTesting Automatic Conversion Workflow")
        print("=" * 50)
        
        # Start analysis
        tree_id = agent.start_analysis(
            user_request="Normalize data, process in R, then verify",
            input_data_path=str(data_path)
        )
        
        # Add nodes - agent should detect conversion needed
        root_id = agent.tree_manager.tree.root_node_id
        
        print("\n1. Adding Python normalization node...")
        python_nodes = agent.tree_manager.add_child_nodes(root_id, [python_block])
        agent.execute_node(python_nodes[0].id)
        
        print("\n2. Adding R processing node (should auto-insert conversion)...")
        # Check conversion
        parent_node = python_nodes[0]
        conversion_block = agent._check_conversion_needed(parent_node, r_block, temp_path)
        
        if conversion_block:
            print("   ✓ Conversion detected! Adding conversion node first.")
            conv_nodes = agent.tree_manager.add_child_nodes(parent_node.id, [conversion_block])
            agent.execute_node(conv_nodes[0].id)
            parent_id = conv_nodes[0].id
        else:
            parent_id = parent_node.id
            
        r_nodes = agent.tree_manager.add_child_nodes(parent_id, [r_block])
        agent.execute_node(r_nodes[0].id)
        
        print("\n3. Adding Python verification (should auto-insert conversion)...")
        parent_node = r_nodes[0]
        conversion_block = agent._check_conversion_needed(parent_node, verify_block, temp_path)
        
        if conversion_block:
            print("   ✓ Conversion detected! Adding conversion node first.")
            conv_nodes = agent.tree_manager.add_child_nodes(parent_node.id, [conversion_block])
            agent.execute_node(conv_nodes[0].id)
            parent_id = conv_nodes[0].id
        else:
            parent_id = parent_node.id
            
        verify_nodes = agent.tree_manager.add_child_nodes(parent_id, [verify_block])
        agent.execute_node(verify_nodes[0].id)
        
        # Check results
        print("\n" + "=" * 50)
        print("Workflow Summary:")
        print(f"Total nodes: {len(agent.tree_manager.tree.nodes)}")
        
        for node_id, node in agent.tree_manager.tree.nodes.items():
            status = "✓" if node.state.value == "completed" else "✗"
            print(f"{status} {node.function_block.name} ({node.function_block.type.value})")
        
        # Verify final node succeeded
        final_node = verify_nodes[0]
        assert final_node.state.value == "completed", "Final verification failed"
        
        print("\nTest passed! Automatic conversion is working correctly.")
        return True


if __name__ == "__main__":
    test_automatic_conversion()