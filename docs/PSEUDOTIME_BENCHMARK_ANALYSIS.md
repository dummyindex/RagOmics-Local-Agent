# Pseudotime Benchmark Analysis and Improvements

## Summary

I've analyzed the ragomics agent's ability to execute a complex pseudotime benchmark workflow and made significant improvements to the prompts to guide better function block creation.

## Initial Issues Found

1. **Monolithic Function Blocks**: The agent was creating single function blocks that tried to do everything (quality control + all pseudotime methods in one block).

2. **Missing Prerequisites**: DPT requires specific preprocessing steps (normalization, PCA, neighbors, diffmap, root cell) that weren't being properly sequenced.

3. **Incorrect API Usage**: The agent was using non-existent functions like `sc.pl.histogram` instead of using matplotlib directly.

4. **PAGA Misunderstanding**: The agent incorrectly assumed PAGA creates pseudotime values in `obs['paga']`, when it actually creates a graph structure.

5. **Lack of Modularity**: The system wasn't creating separate nodes for each pseudotime method as intended.

## Improvements Made

### 1. Enhanced Function Selector Prompt

Added clear principles at the beginning:
- **ONE TASK PER FUNCTION BLOCK**: Each function block must perform exactly ONE specific task
- **MODULAR WORKFLOW**: Complex requests must be broken into multiple sequential nodes
- **PROPER SEQUENCING**: Ensure correct order of operations

Added specific pseudotime method requirements:
- DPT: Requires normalized data, PCA, neighbors graph, diffmap, AND root cell (iroot)
- PAGA: Creates graph structure, not direct pseudotime values
- Slingshot (R): Requires dimensionality reduction and cluster labels
- Palantir: Requires normalized data, PCA, and start cell
- Monocle 3 (R): Requires normalized data and dimensionality reduction

### 2. Enhanced Function Creator Prompt

Added critical instruction at the top:
- **CREATE A FUNCTION BLOCK THAT DOES EXACTLY ONE SPECIFIC TASK**
- Do NOT combine multiple analysis steps

Added pseudotime-specific code examples for each method with proper prerequisites.

### 3. Enhanced Bug Fixer Prompt

Added pseudotime-specific debugging guidance:
- Common DPT errors and their fixes
- Prerequisites checking
- Root cell specification requirements

### 4. Improved Documentation

Added correct examples for histogram plotting and other common pitfalls.

## Results After Improvements

The agent now successfully:
1. Creates separate modular function blocks for each step
2. Properly sequences preprocessing steps before pseudotime methods
3. Successfully executes DPT with proper prerequisites
4. Fixes errors more effectively with context-aware debugging

### Successful Execution Flow:
1. ✅ quality_control - Filters cells and genes
2. ✅ normalize_and_log_transform - Normalizes data
3. ✅ apply_pca_umap - Performs dimensionality reduction
4. ✅ run_dpt - Successfully runs DPT pseudotime

## Remaining Challenges

1. **Method Selection**: After DPT, the agent sometimes gets confused and tries clustering metrics instead of continuing with other pseudotime methods.

2. **R Integration**: The R-based methods (Slingshot, Monocle 3) haven't been tested yet due to the agent not reaching those steps.

3. **PAGA Understanding**: The agent needs better guidance on how PAGA results should be used.

## Recommendations

1. **Add Examples to Prompts**: Include more concrete examples of successful multi-method workflows in the function selector prompt.

2. **Improve Task Context**: Pass more context about what methods have already been completed to help the agent decide next steps.

3. **Method-Specific Templates**: Consider adding method-specific function block templates that agents can use as starting points.

4. **Explicit Workflow Guidance**: For complex multi-method benchmarks, consider providing a more explicit workflow structure that the agent can follow.

5. **Test R Integration**: Once the Python methods work reliably, focus on testing the R-based pseudotime methods and the automatic conversion nodes.

## Conclusion

The prompt improvements have significantly enhanced the agent's ability to create modular, focused function blocks. The system now successfully handles preprocessing and basic pseudotime methods. With further refinements to method selection and workflow guidance, the agent should be able to complete the full pseudotime benchmark successfully.