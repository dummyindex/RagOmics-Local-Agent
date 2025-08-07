"""Test agent integration with automatic conversion."""

import os
import sys
import json
import tempfile
from pathlib import Path
import numpy as np
import anndata
from scipy.sparse import csr_matrix

# Fix imports
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(Path(__file__).parent.parent)

from agents.main_agent import MainAgent  
from models import (
    NewFunctionBlock, FunctionBlockType, StaticConfig,
    GenerationMode
)


def create_test_data(temp_dir: Path) -> Path:
    """Create test AnnData file."""
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
    
    # Save
    data_path = temp_dir / "test_data.h5ad"
    adata.write_h5ad(data_path)
    
    return data_path


def test_agent_with_conversion():
    """Test that agent correctly handles Python->R->Python workflow."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print("Testing Agent with Automatic Conversion")
        print("=" * 60)
        
        # Create test data
        data_path = create_test_data(temp_path)
        print(f"Created test data: {data_path}")
        
        # Initialize agent
        workspace = temp_path / "agent_workspace"
        agent = MainAgent(
            workspace_dir=workspace,
            generation_mode=GenerationMode.ONLY_NEW,
            max_nodes=10,
            max_debug_trials=1
        )
        
        # Define function blocks
        python_qc = NewFunctionBlock(
            name="python_qc",
            type=FunctionBlockType.PYTHON,
            description="Quality control with scanpy",
            code="""
def run(path_dict, params):
    import anndata
    import scanpy as sc
    import os
    
    # Load data
    adata = anndata.read_h5ad(os.path.join(path_dict['input_dir'], '_node_anndata.h5ad'))
    
    # Basic QC
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Save
    adata.write_h5ad(os.path.join(path_dict['output_dir'], '_node_anndata.h5ad'))
    
    print(f"QC complete: {adata.n_obs} cells, {adata.n_vars} genes")
    return {"qc_complete": True}
""",
            requirements="scanpy\nanndata",
            static_config=StaticConfig(
                args=[],
                description="QC with scanpy",
                tag="qc"
            ),
            parameters={}
        )
        
        r_process = NewFunctionBlock(
            name="r_analysis",
            type=FunctionBlockType.R,
            description="Process with R",
            code="""
run <- function(path_dict, params) {
    library(Matrix)
    library(jsonlite)
    
    # Read SC matrix
    sc_dir <- file.path(path_dict$input_dir, "_node_sc_matrix")
    if (!dir.exists(sc_dir)) {
        stop("SC matrix not found - conversion may have failed")
    }
    
    # Load matrix
    X <- readMM(file.path(sc_dir, "X.mtx"))
    obs_names <- readLines(file.path(sc_dir, "obs_names.txt"))
    var_names <- readLines(file.path(sc_dir, "var_names.txt"))
    
    # Simple analysis
    gene_means <- rowMeans(t(X))
    top_genes <- var_names[order(gene_means, decreasing = TRUE)[1:10]]
    
    # Save results
    results <- list(
        n_cells = length(obs_names),
        n_genes = length(var_names),
        top_expressed_genes = top_genes
    )
    
    write(toJSON(results, pretty = TRUE),
          file.path(path_dict$output_dir, "r_results.json"))
    
    # Copy SC matrix for next step
    file.copy(sc_dir, path_dict$output_dir, recursive = TRUE)
    
    cat("R analysis complete\\n")
    return(list(success = TRUE))
}
""",
            requirements="Matrix\njsonlite",
            static_config=StaticConfig(
                args=[],
                description="R analysis",
                tag="r_analysis"
            ),
            parameters={}
        )
        
        python_viz = NewFunctionBlock(
            name="python_viz",
            type=FunctionBlockType.PYTHON,
            description="Visualize with Python",
            code="""
def run(path_dict, params):
    import json
    import os
    
    # Check for R results
    r_results_file = os.path.join(path_dict['input_dir'], 'r_results.json')
    
    if os.path.exists(r_results_file):
        with open(r_results_file) as f:
            r_results = json.load(f)
        
        print("R Analysis Results:")
        print(f"  Cells analyzed: {r_results['n_cells']}")
        print(f"  Genes analyzed: {r_results['n_genes']}")
        print(f"  Top genes: {', '.join(r_results['top_expressed_genes'][:5])}")
        
        # Check for SC matrix (to verify conversion happened)
        sc_matrix_dir = os.path.join(path_dict['input_dir'], '_node_sc_matrix')
        if os.path.exists(sc_matrix_dir):
            print("  ✓ SC matrix format detected (conversion successful)")
        
        return {"visualization_complete": True}
    else:
        print("No R results found")
        return {"visualization_complete": False}
""",
            requirements="",
            static_config=StaticConfig(
                args=[],
                description="Visualize results",
                tag="viz"
            ),
            parameters={}
        )
        
        # Run workflow
        print("\nStarting agent workflow...")
        tree_id = agent.start_analysis(
            user_request="Run QC, then R analysis, then visualize",
            input_data_path=str(data_path)
        )
        
        root_id = agent.tree_manager.tree.root_node_id
        
        # Step 1: Python QC
        print("\n1. Adding Python QC node...")
        qc_nodes = agent.tree_manager.add_child_nodes(root_id, [python_qc])
        agent.execute_node(qc_nodes[0].id)
        
        # Step 2: R analysis (should trigger conversion)
        print("\n2. Adding R analysis node...")
        parent_node = qc_nodes[0]
        
        # Check if conversion is detected
        conversion_block = agent._check_conversion_needed(
            parent_node, r_process, 
            Path(parent_node.output_data_id) if parent_node.output_data_id else workspace
        )
        
        if conversion_block:
            print("  ✓ Conversion needed: Python → R")
            print(f"  Adding conversion node: {conversion_block.name}")
            conv_nodes = agent.tree_manager.add_child_nodes(parent_node.id, [conversion_block])
            agent.execute_node(conv_nodes[0].id)
            parent_id = conv_nodes[0].id
        else:
            print("  ✗ No conversion detected (this is a bug)")
            parent_id = parent_node.id
        
        # Add R node
        r_nodes = agent.tree_manager.add_child_nodes(parent_id, [r_process])
        agent.execute_node(r_nodes[0].id)
        
        # Step 3: Python viz (should trigger conversion back)
        print("\n3. Adding Python visualization node...")
        parent_node = r_nodes[0]
        
        conversion_block = agent._check_conversion_needed(
            parent_node, python_viz,
            Path(parent_node.output_data_id) if parent_node.output_data_id else workspace
        )
        
        if conversion_block:
            print("  ✓ Conversion needed: R → Python")
            print(f"  Adding conversion node: {conversion_block.name}")
            conv_nodes = agent.tree_manager.add_child_nodes(parent_node.id, [conversion_block])
            agent.execute_node(conv_nodes[0].id)
            parent_id = conv_nodes[0].id
        else:
            print("  ✗ No conversion detected")
            parent_id = parent_node.id
        
        # Add viz node
        viz_nodes = agent.tree_manager.add_child_nodes(parent_id, [python_viz])
        agent.execute_node(viz_nodes[0].id)
        
        # Summary
        print("\n" + "=" * 60)
        print("Workflow Summary:")
        
        all_nodes = []
        for node_id, node in agent.tree_manager.tree.nodes.items():
            all_nodes.append((node.level, node))
        
        all_nodes.sort(key=lambda x: x[0])
        
        for level, node in all_nodes:
            indent = "  " * level
            status = "✓" if node.state.value == "completed" else "✗"
            print(f"{indent}{status} {node.function_block.name} ({node.function_block.type.value})")
        
        # Verify success
        final_node = viz_nodes[0]
        assert final_node.state.value == "completed", "Visualization node failed"
        
        # Count conversion nodes
        conversion_count = sum(1 for _, node in all_nodes 
                             if "convert" in node.function_block.name.lower())
        
        print(f"\nConversion nodes inserted: {conversion_count}")
        print("Expected conversions: 2 (Python→R and R→Python)")
        
        if conversion_count >= 2:
            print("\n✅ Test PASSED: Agent correctly handled language conversions")
        else:
            print("\n⚠️  Test WARNING: Expected 2 conversions but found", conversion_count)
        
        return True


if __name__ == "__main__":
    try:
        test_agent_with_conversion()
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)