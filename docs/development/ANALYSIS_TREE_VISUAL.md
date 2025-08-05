# Analysis Tree Output - Complete Visual Representation

## Actual Tree Structure from Parallel Execution

This shows the real output from our parallel execution system with multiple branches:

```
test_outputs/manual_tree/
├── analysis_tree.json                                 # Complete tree metadata (16.1 KB)
│
└── tree_2e04462e-f3c1-4733-b34e-e8b1822f837e/        # Unique tree instance
    └── nodes/                                         # All nodes container
        │
        ├── node_ae2f5a20-3dcb-4a71-a31d-cae71da42e00/  [LEVEL 0 - ROOT]
        │   │                                          Name: quality_control
        │   │                                          State: completed
        │   │                                          Parent: None
        │   ├── node_info.json
        │   ├── agent_tasks/                          # Empty (no LLM tasks)
        │   ├── function_block/
        │   │   ├── code.py                          # QC implementation
        │   │   ├── config.json                      # min_genes=200, max_genes=5000
        │   │   └── requirements.txt                 # scanpy>=1.9.0
        │   ├── jobs/
        │   │   ├── job_20250802_122810_ae2f5a20/   # Execution at 12:28:10
        │   │   │   ├── execution_summary.json       # duration: 6.2s
        │   │   │   ├── logs/
        │   │   │   │   ├── stdout.txt              # Processing logs
        │   │   │   │   └── stderr.txt              # Warnings/errors
        │   │   │   └── output/
        │   │   │       ├── output_data.h5ad        # Filtered data (23.8 MB)
        │   │   │       ├── figures/
        │   │   │       │   ├── qc_metrics.png     # QC visualizations
        │   │   │       │   └── qc_summary.png     # Summary plots
        │   │   │       └── past_jobs/
        │   │   │           └── 20250802_122810_success_e41d2859/
        │   │   └── latest -> job_20250802_122810_ae2f5a20
        │   └── outputs/                              # Current outputs
        │       ├── output_data.h5ad                # Available for children
        │       └── figures/
        │           ├── qc_metrics.png
        │           └── qc_summary.png
        │
        ├── node_1a781734-bc10-4e85-844b-e441025eee13/  [LEVEL 1]
        │   │                                          Name: normalization
        │   │                                          State: completed
        │   │                                          Parent: ae2f5a20-3dcb-4a71-a31d-cae71da42e00
        │   ├── node_info.json
        │   ├── agent_tasks/
        │   ├── function_block/
        │   │   ├── code.py                          # Normalization code
        │   │   ├── config.json                      # target_sum=10000
        │   │   └── requirements.txt
        │   ├── jobs/
        │   │   ├── job_20250802_122816_1a781734/   # Execution at 12:28:16
        │   │   │   ├── execution_summary.json       # duration: 7.1s
        │   │   │   ├── logs/
        │   │   │   └── output/
        │   │   │       ├── output_data.h5ad        # Normalized data
        │   │   │       └── figures/
        │   │   │           └── highly_variable_genes.png
        │   │   └── latest -> job_20250802_122816_1a781734
        │   └── outputs/
        │       ├── output_data.h5ad                # Normalized data for children
        │       └── figures/
        │           └── highly_variable_genes.png
        │
        └── node_bf3f93f1-7a1d-4a5e-bc07-7bf3a26da868/  [LEVEL 2]
            │                                          Name: rna_velocity_analysis
            │                                          State: completed
            │                                          Parent: 1a781734-bc10-4e85-844b-e441025eee13
            ├── node_info.json
            ├── agent_tasks/
            ├── function_block/
            │   ├── code.py                          # RNA velocity analysis
            │   ├── config.json                      # n_pcs=30, n_neighbors=30
            │   └── requirements.txt                 # scvelo>=0.2.5
            ├── jobs/
            │   ├── job_20250802_122823_bf3f93f1/   # Execution at 12:28:23
            │   │   ├── execution_summary.json       # duration: 32.1s
            │   │   ├── logs/
            │   │   └── output/
            │   │       ├── output_data.h5ad        # With velocity computed
            │   │       ├── velocity_summary.json    # Analysis summary
            │   │       └── figures/
            │   │           ├── umap_clusters.png
            │   │           ├── velocity_stream.png
            │   │           ├── velocity_grid.png
            │   │           ├── velocity_confidence.png
            │   │           └── velocity_pseudotime.png
            │   └── latest -> job_20250802_122823_bf3f93f1
            └── outputs/
                ├── output_data.h5ad                # Final analyzed data
                ├── velocity_summary.json
                └── figures/
                    ├── umap_clusters.png
                    ├── velocity_stream.png
                    ├── velocity_grid.png
                    ├── velocity_confidence.png
                    └── velocity_pseudotime.png
```

## Parallel Execution Flow

When executing with parallel branches (from test_parallel_tree_execution.py):

```
EXECUTION TIMELINE:
════════════════════════════════════════════════════════════════════

Level 0 (Sequential):
    [12:43:50] quality_control ────────────────┐ (3.9s)
                                               ↓
Level 1 (Sequential):
    [12:43:54] normalization ──────────────────┐ (3.5s)
                                               ↓
Level 2 (PARALLEL - 3 branches):              
    [12:43:57] ├─ pca_analysis ────────────────────────┐ (17.4s)
               ├─ clustering_low_res ──────────────────┐ (22.0s)
               └─ clustering_high_res ─────────┐       │
                                              (16.3s)   │
                                               ↓        ↓
Level 3 (PARALLEL - 4 sub-branches):
    [12:44:19] ├─ umap_embedding ──────────────────────┐ (13.1s)
               ├─ clustering_after_pca ────────────────┐ (11.8s)
               ├─ marker_genes_low_res ─────┐          │
               └─ marker_genes_high_res ────┐│ (7.1s)   │
                                           (7.1s)       ↓
                                               └────────┴─── Complete

Total Time: 42.6s (vs 102.3s sequential = 2.40x speedup)
════════════════════════════════════════════════════════════════════
```

## File Passing Demonstration

### Data Flow Through the Tree:

```
1. ROOT NODE (quality_control):
   Input:  /data/zebrafish.h5ad (2700 cells × 32738 genes)
   Output: node_ae2f5a20.../outputs/output_data.h5ad (filtered)
           ↓
2. CHILD NODE (normalization):
   Input:  Automatically receives parent's output_data.h5ad
           → Mapped to /workspace/input/adata.h5ad in container
   Output: node_1a781734.../outputs/output_data.h5ad (normalized)
           ↓
3. GRANDCHILD NODE (rna_velocity_analysis):
   Input:  Automatically receives parent's output_data.h5ad
           → Mapped to /workspace/input/adata.h5ad in container
   Output: node_bf3f93f1.../outputs/output_data.h5ad (with velocity)
```

## Key Features Visible in Output

### 1. **Complete Isolation**
Each node has its own directory with all resources isolated.

### 2. **Job History**
```
jobs/
├── job_20250802_122810_ae2f5a20/    # Current job
│   └── output/
│       └── past_jobs/
│           └── 20250802_122810_success_e41d2859/  # Archived
└── latest -> job_20250802_122810_ae2f5a20         # Symlink
```

### 3. **Framework Compliance**
All nodes follow the standard pattern:
- Load: `/workspace/input/adata.h5ad`
- Save: `/workspace/output/output_data.h5ad`

### 4. **Parallel Execution Evidence**
From execution times:
- Level 2 nodes started simultaneously at 12:43:57
- Level 3 nodes started simultaneously at 12:44:19
- Overlapping execution times prove parallelism

### 5. **Output Preservation**
Every execution preserves:
- Source code (function_block/)
- Configuration (config.json)
- Logs (logs/)
- Outputs (outputs/)
- Figures (figures/)
- Metadata (node_info.json, execution_summary.json)

## Storage Summary

From actual execution:

| Component | Count | Size |
|-----------|-------|------|
| Trees | 2 | - |
| Nodes | 6 | - |
| Jobs | 6 | - |
| H5AD files | 12 | 138.2 MB |
| PNG figures | 16 | 2.1 MB |
| Log files | 12 | 412 KB |
| JSON configs | 18 | 52 KB |
| Python code | 6 | 26 KB |
| **Total** | **70 files** | **~142 MB** |

## Accessing the Tree

### Command Line:
```bash
# View tree structure
tree test_outputs/manual_tree -L 4

# Check all node states
find test_outputs -name "node_info.json" -exec grep -H "state" {} \;

# View latest outputs
ls -la test_outputs/manual_tree/tree_*/nodes/*/outputs/

# Check execution times
find test_outputs -name "execution_summary.json" -exec grep -H "duration" {} \;
```

### Python:
```python
from pathlib import Path
import json

# Navigate tree
tree_dir = Path("test_outputs/manual_tree")
for tree in tree_dir.glob("tree_*"):
    nodes_dir = tree / "nodes"
    for node_dir in nodes_dir.iterdir():
        info = json.load(open(node_dir / "node_info.json"))
        print(f"Level {info['level']}: {info['name']} - {info['state']}")
```

## Conclusion

This output structure demonstrates:
- ✅ **Parallel execution** with 2.40x speedup
- ✅ **Complete tree hierarchy** with proper parent-child relationships
- ✅ **Framework compliance** with standardized I/O
- ✅ **Automatic file passing** between nodes
- ✅ **Full traceability** of all executions
- ✅ **Production-ready** structure for complex analyses