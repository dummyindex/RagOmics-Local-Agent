#!/usr/bin/env python3
"""Debug pseudotime benchmark to understand why it stops early."""

import os
import sys
import json
import logging
from pathlib import Path
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from ragomics_agent_local.agents.function_creator_agent import FunctionCreatorAgent
from ragomics_agent_local.llm_service import OpenAIService
from ragomics_agent_local.models import AnalysisTree, AnalysisNode, GenerationMode
from ragomics_agent_local.config import config


def test_function_creator_logic():
    """Test the function creator's decision logic."""
    
    print("\n" + "="*80)
    print("TESTING FUNCTION CREATOR DECISION LOGIC")
    print("="*80 + "\n")
    
    # Create LLM service
    llm_service = OpenAIService()
    
    # Create function creator
    creator = FunctionCreatorAgent(llm_service)
    
    # Create minimal tree with one completed preprocessing node
    tree = AnalysisTree(
        user_request="""Run a comprehensive pseudotime analysis benchmark on the zebrafish dataset:
1. Preprocess the dataset:
   - Use Scanpy to normalize (normalize_total), log transform (log1p), 
   - Find highly variable genes (highly_variable_genes)
   - Compute PCA, build neighbor graph, and compute UMAP

2. Run DPT pseudotime analysis:
   - Use sc.tl.dpt() to compute diffusion pseudotime
   - Store result in adata.obs['dpt_pseudotime']

3. Run Palantir pseudotime analysis:
   - Use the PCA embedding from preprocessing
   - Compute Palantir pseudotime and store in adata.obs['palantir_pseudotime']

4. Create visualization:
   - Plot UMAP colored by each pseudotime method
   - Save as pseudotime_comparison.png""",
        input_data_path="test.h5ad",
        max_nodes=20,
        max_children_per_node=1,
        max_debug_trials=3,
        generation_mode=GenerationMode.MIXED
    )
    
    # Add a completed preprocessing node
    from ragomics_agent_local.models import NewFunctionBlock, FunctionBlockType, StaticConfig
    preprocessing_fb = NewFunctionBlock(
        name="basic_preprocessing",
        type=FunctionBlockType.PYTHON,
        description="Basic data preprocessing",
        code="# preprocessing code",
        requirements="scanpy",
        parameters={},
        static_config=StaticConfig(args=[], description="Basic preprocessing", tag="preprocessing")
    )
    
    preprocessing_node = AnalysisNode(
        parent_id=None,
        analysis_id=tree.id,
        function_block=preprocessing_fb,
        level=0
    )
    preprocessing_node.state = "completed"
    tree.nodes[preprocessing_node.id] = preprocessing_node
    tree.root_node_id = preprocessing_node.id
    tree.total_nodes = 1
    tree.completed_nodes = 1
    
    # Test the selection/creation logic
    print("1. Testing with completed preprocessing node...")
    
    context = {
        "user_request": tree.user_request,
        "tree": tree,
        "current_node": preprocessing_node,
        "parent_chain": [preprocessing_node],
        "generation_mode": tree.generation_mode,
        "max_children": 3,  # Allow multiple children
        "data_summary": {
            "n_obs": 26733,
            "n_vars": 2000,
            "obs_columns": ["n_genes", "n_counts"],
            "obsm_keys": ["X_pca", "X_umap"]
        }
    }
    
    # Make the request
    print("\nCalling process_selection_or_creation...")
    result = creator.process_selection_or_creation(context)
    
    # Display result
    print("\nResult:")
    print(f"  Satisfied: {result.get('satisfied', 'N/A')}")
    print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
    print(f"  Function blocks: {len(result.get('function_blocks', []))}")
    
    if result.get('function_blocks'):
        print("\n  Recommended function blocks:")
        for i, fb in enumerate(result['function_blocks']):
            print(f"    {i+1}. {fb.name}: {fb.description}")
            
    # Save the full result for inspection
    output_dir = Path("test_outputs/debug_creator")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "creator_result.json", 'w') as f:
        json.dump({
            'context': {k: str(v) if k in ['tree', 'current_node'] else v for k, v in context.items()},
            'result': {
                'satisfied': result.get('satisfied'),
                'reasoning': result.get('reasoning'),
                'function_blocks': [fb.name for fb in result.get('function_blocks', [])]
            }
        }, f, indent=2)
        
    print(f"\nFull result saved to: {output_dir / 'creator_result.json'}")
    
    # Test with empty tree
    print("\n\n2. Testing with empty tree (initial planning)...")
    
    empty_tree = AnalysisTree(
        user_request=tree.user_request,
        input_data_path="test.h5ad",
        max_nodes=20,
        max_children_per_node=1,
        max_debug_trials=3,
        generation_mode=GenerationMode.MIXED
    )
    
    context2 = {
        "user_request": empty_tree.user_request,
        "tree": empty_tree,
        "current_node": None,
        "parent_chain": [],
        "generation_mode": empty_tree.generation_mode,
        "max_children": 1,
        "data_summary": {}
    }
    
    result2 = creator.process_selection_or_creation(context2)
    
    print("\nResult for empty tree:")
    print(f"  Satisfied: {result2.get('satisfied', 'N/A')}")
    print(f"  Function blocks: {len(result2.get('function_blocks', []))}")
    
    if result2.get('function_blocks'):
        print("\n  Recommended function blocks:")
        for i, fb in enumerate(result2['function_blocks']):
            print(f"    {i+1}. {fb.name}: {fb.description}")


if __name__ == "__main__":
    test_function_creator_logic()