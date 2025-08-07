# Architecture Refactoring Plan

**Status: COMPLETED (2025-08-05)**

## Current Issues (RESOLVED)
1. **Confusing separation** between FunctionSelectorAgent and FunctionCreatorAgent
2. **Hardcoded logic** in OrchestratorAgent.plan_next_steps() with 5-node satisfaction limit
3. **Disconnect** between orchestrator.process() (which uses function_selector) and main_agent using orchestrator.plan_next_steps()

## Proposed Solution

### 1. Combine Selector and Creator
Create a unified `FunctionCreatorAgent` that can:
- Decide whether to create new or use existing function blocks
- Generate new function block code when needed
- Select from existing blocks when appropriate
- Handle the full decision-making process

### 2. Fix Orchestrator Logic
- Remove the hardcoded `plan_next_steps()` method
- Make orchestrator use the enhanced function creator directly
- Respect user-provided limits (max_nodes, max_children) instead of hardcoded values

### 3. Simplify Main Agent
- Remove separate calls to selector and creator
- Use unified function creator for all function block decisions
- Cleaner flow: orchestrator → function_creator → execution

## Implementation Steps

1. **Enhance FunctionCreatorAgent**
   - Add methods from FunctionSelectorAgent
   - Add logic to decide between new/existing blocks
   - Keep all creation capabilities

2. **Update OrchestratorAgent**
   - Remove hardcoded plan_next_steps()
   - Use function_creator.process() directly
   - Remove 5-node limit and other hardcoded logic

3. **Update MainAgent**
   - Remove function_selector references
   - Update _process_recommendations to use unified creator
   - Simplify the flow

4. **Update Tests**
   - Fix tests that rely on separate selector/creator
   - Add tests for unified creator behavior

## Benefits
- Simpler architecture
- No confusion about which agent does what
- Easier to maintain and extend
- Respects user parameters properly
- No hardcoded analysis-specific logic

## Completion Summary

### What Was Done:
1. **✓ Combined FunctionSelectorAgent into FunctionCreatorAgent**
   - Added `process_selection_or_creation()` method to FunctionCreatorAgent
   - Removed FunctionSelectorAgent from codebase
   - Updated all imports and references

2. **✓ Fixed Hardcoded 5-Node Limit**
   - Removed hardcoded satisfaction logic from orchestrator_agent.py
   - Now respects user-provided max_nodes and max_children_per_node parameters
   - No analysis-specific hardcoded logic remains

3. **✓ Updated All Affected Components**
   - MainAgent now uses unified FunctionCreatorAgent
   - OrchestratorAgent updated to use function_creator instead of function_selector
   - All tests updated to work with new architecture
   - Import statements cleaned up

4. **✓ Removed Pseudotime-Specific Code**
   - All pseudotime-specific prompts and logic removed from agents
   - Agents are now completely generic and analysis-agnostic

### Key Changes:
- `FunctionCreatorAgent.process_selection_or_creation()` replaces separate selector logic
- Tree expansion now properly respects `max_nodes` and `max_children_per_node` limits
- No hardcoded satisfaction conditions based on node count or analysis type
- Cleaner separation of concerns between orchestration and function block management