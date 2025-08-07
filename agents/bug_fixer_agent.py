"""Agent responsible for debugging and fixing failed function blocks."""

from typing import Dict, Any, Optional, List
import re
from pathlib import Path
from datetime import datetime

from .base_agent import BaseAgent
from .agent_output_utils import AgentOutputLogger
from .task_manager import TaskType, TaskStatus
from ..models import NewFunctionBlock, FunctionBlockType
from ..llm_service import OpenAIService


class BugFixerAgent(BaseAgent):
    """Agent that debugs and fixes failed function blocks."""
    
    SYSTEM_PROMPT = """You are an expert debugger for bioinformatics function blocks.

Your task is to analyze errors and fix issues in function blocks based on the error messages and context provided.

Focus on:
1. PRIORITIZE actual Python errors (KeyError, AttributeError, ValueError) over warnings
2. When you see a KeyError for a missing key, investigate WHY the key is missing:
   - Is the computation creating a different key name?
   - Is there a prerequisite step missing?
   - Is the API being used incorrectly?
3. Don't just add existence checks - understand the root cause
4. Look at the code logic flow - are operations in the correct order?
5. Check if computations are being done before their prerequisites
6. Understanding library API conventions (what keys/attributes they create)
7. Fixing the actual problem, not just masking the error

Common issues to watch for:
- F-string syntax errors with nested quotes/brackets - use intermediate variables instead
- Missing embeddings like X_pca when calculating metrics - check existence first
- Case-sensitive column names in adata.obs - check actual column names
- Missing directories when saving files - create them first
- Scanpy plotting functions expect column names, not numpy arrays
- Use available columns from parent data structure context when provided
- AttributeError with scanpy functions - verify the function exists in the API

Important debugging principles:
- When you see KeyError: Understand WHY the key doesn't exist, don't just check for it
- The error location shows WHERE it failed, but you need to find WHAT caused it
- Look backwards from the error - what should have created that key?
- Common pattern: operations done in wrong order (e.g., accessing result before computation)
- Library APIs have specific conventions - understand what they expect and produce
- Fix the root cause, not the symptom

Common fixes:
- Replace 'sklearn' with 'scikit-learn' in requirements
- Use sc.tl.leiden instead of sc.tl.louvain (louvain is deprecated)
- For KMeans, use sklearn.cluster.KMeans instead of sc.tl.kmeans (doesn't exist)
- Scanpy plotting functions (sc.pl.*) expect column names, not arrays
- Add get_param helper function when params have nested dictionaries
- DPT requires preprocessing: neighbors and diffmap must be run first
- IMPORTANT: sc.tl.dpt() creates 'dpt_pseudotime' NOT 'dpt' in adata.obs
- Check if computation results exist before accessing (e.g., 'dpt_pseudotime' in adata.obs)
- PAGA requires neighbors to be computed first
- Many methods need PCA computed first (check 'X_pca' in adata.obsm)

When analyzing errors:
1. Read the full traceback to understand what function call failed
2. Identify what the function was supposed to do
3. Research which library provides that functionality
4. Consider alternative implementations from appropriate libraries

Understanding method requirements:
- Some methods require specific preprocessing steps
- Some methods need parameters set BEFORE calling them
- Check documentation or error messages for what's required
- Order matters: prerequisites must be done first
- Understand what keys/attributes each method creates

Scanpy API key mappings (CRITICAL):
- sc.tl.dpt() → creates 'dpt_pseudotime' in adata.obs (NOT 'dpt')
- sc.tl.paga() → creates 'paga' in adata.uns
- sc.tl.diffmap() → creates 'X_diffmap' in adata.obsm
- sc.pp.neighbors() → creates 'neighbors' in adata.uns
- sc.tl.umap() → creates 'X_umap' in adata.obsm

You will respond in JSON format with your analysis and fixed code."""
    
    def __init__(self, llm_service: Optional[OpenAIService] = None, task_manager=None):
        super().__init__("bug_fixer", task_manager)
        self.llm_service = llm_service
        self.max_error_lines = 1000  # Maximum lines of error output to include
        self.agent_logger = None  # Will be initialized per node
        
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fix a failed function block.
        
        Required context keys:
            - function_block: NewFunctionBlock
            - error_message: str
            - stdout: str
            - stderr: str
            - previous_attempts: List[str] (optional)
            - parent_task_id: str (optional)
            - analysis_id: str (optional)
            - node_id: str (optional)
            - job_id: str (optional)
            - node_dir: Path (optional) - Node directory for logging
            
        Returns:
            Dict with:
                - fixed_code: Optional[str]
                - fixed_requirements: Optional[str]
                - success: bool
                - reasoning: str
                - task_id: Optional[str]
        """
        self.validate_context(context, ['function_block', 'error_message'])
        
        function_block = context['function_block']
        error_message = context['error_message']
        stdout = context.get('stdout', '')
        stderr = context.get('stderr', '')
        previous_attempts = context.get('previous_attempts', [])
        
        # Initialize agent logger if node_dir is provided
        if 'node_dir' in context:
            self.agent_logger = AgentOutputLogger(context['node_dir'], 'bug_fixer')
        
        # Create task if we have a task manager
        task_id = None
        if self.task_manager:
            task_context = {
                'function_block_id': function_block.id,
                'function_block_name': function_block.name,
                'error_message': error_message,
                'analysis_id': context.get('analysis_id'),
                'node_id': context.get('node_id'),
                'job_id': context.get('job_id')
            }
            
            task_id = self.create_task(
                task_type=TaskType.BUG_FIXING,
                description=f"Fix error in {function_block.name}: {error_message[-300:]}",
                context=task_context,
                parent_task_id=context.get('parent_task_id')
            )
            
            # Save error details
            if task_id:
                self.task_manager.save_task_artifact(
                    task_id,
                    "error_details.json",
                    {
                        'error_message': error_message,
                        'stdout': stdout,
                        'stderr': stderr,
                        'original_code': function_block.code,
                        'original_requirements': function_block.requirements
                    }
                )
                
                self.update_task_status(TaskStatus.IN_PROGRESS)
        
        # Always use LLM for bug fixing - no hardcoded solutions
        if self.llm_service:
            fixed_code, fixed_requirements = self._debug_with_llm(
                function_block=function_block,
                error_message=error_message,
                stdout=stdout,
                stderr=stderr,
                previous_attempts=previous_attempts,
                task_id=task_id,
                parent_data_structure=context.get('parent_data_structure')
            )
            
            if fixed_code:
                result = {
                    'fixed_code': fixed_code,
                    'fixed_requirements': fixed_requirements,
                    'success': True,
                    'reasoning': 'Fixed using LLM debugging',
                    'task_id': task_id
                }
                
                if task_id:
                    self.update_task_status(TaskStatus.COMPLETED, results=result)
                    self.task_manager.save_task_artifact(
                        task_id,
                        "fixed_code.py",
                        fixed_code
                    )
                
                return result
        
        # If no fix available
        result = {
            'fixed_code': None,
            'fixed_requirements': None,
            'success': False,
            'reasoning': 'Unable to automatically fix the error',
            'task_id': task_id
        }
        
        if task_id:
            self.update_task_status(TaskStatus.FAILED, error=result['reasoning'])
            
        return result
    
    def _debug_with_llm(
        self,
        function_block: NewFunctionBlock,
        error_message: str,
        stdout: str,
        stderr: str,
        previous_attempts: List[str],
        task_id: Optional[str],
        parent_data_structure: Optional[Dict[str, Any]] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """Use LLM to debug the function block."""
        language = "python" if function_block.type == FunctionBlockType.PYTHON else "r"
        
        # Extract the actual error from stdout if it contains traceback
        actual_error_lines = []
        if stdout and "Traceback (most recent call last):" in stdout:
            # Find the traceback section in stdout
            lines = stdout.split('\n')
            traceback_start = -1
            for i, line in enumerate(lines):
                if "Traceback (most recent call last):" in line:
                    traceback_start = i
            
            if traceback_start >= 0:
                # Include the full traceback
                actual_error_lines = lines[traceback_start:]
                # Limit to reasonable size
                if len(actual_error_lines) > 50:
                    actual_error_lines = actual_error_lines[-50:]
        
        # Truncate long outputs but prioritize error information
        stdout_lines = stdout.split('\n')[-self.max_error_lines:] if stdout else []
        stderr_lines = stderr.split('\n')[-self.max_error_lines:] if stderr else []
        
        # Build debug prompt
        prompt = self._build_debug_prompt(
            function_block=function_block,
            error_message=error_message,
            stdout_lines=stdout_lines,
            stderr_lines=stderr_lines,
            previous_attempts=previous_attempts,
            parent_data_structure=parent_data_structure,
            actual_error_lines=actual_error_lines
        )
        
        try:
            # Define JSON schema for structured response
            schema = {
                "name": "bug_fix",
                "schema": {
                    "type": "object",
                    "properties": {
                        "analysis": {
                            "type": "object",
                            "properties": {
                                "error_type": {"type": "string", "description": "Type of error (e.g., ImportError, AttributeError, ValueError)"},
                                "root_cause": {"type": "string", "description": "Root cause of the error"},
                                "fix_strategy": {"type": "string", "description": "Strategy to fix the error"}
                            },
                            "required": ["error_type", "root_cause", "fix_strategy"]
                        },
                        "fixed_code": {"type": "string", "description": "Complete fixed Python/R code"},
                        "changes_made": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of specific changes made to fix the error"
                        },
                        "requirements_changes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of packages to add/remove from requirements. Format: '+package' to add, '-package' to remove (empty if no changes)"
                        }
                    },
                    "required": ["analysis", "fixed_code", "changes_made", "requirements_changes"]
                }
            }
            
            # Call LLM service
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            # Prepare LLM input for logging
            llm_input = {
                'messages': messages,
                'schema': schema,
                'temperature': 0.3,
                'max_tokens': 3000,
                'model': self.llm_service.model,
                'timestamp': datetime.now().isoformat()
            }
            
            response = self.llm_service.chat_completion_json(
                messages=messages,
                json_schema=schema,
                temperature=0.3,
                max_tokens=3000
            )
            
            # Extract code and requirements from JSON response
            fixed_code = response.get('fixed_code') if response else None
            
            # Handle requirement changes
            requirements = function_block.requirements or ""
            if response and response.get('requirements_changes'):
                req_changes = response['requirements_changes']
                for change in req_changes:
                    if change.startswith('+'):
                        # Add requirement
                        new_req = change[1:].strip()
                        if new_req not in requirements:
                            if requirements and not requirements.endswith('\n'):
                                requirements += '\n'
                            requirements += new_req + '\n'
                    elif change.startswith('-'):
                        # Remove requirement
                        old_req = change[1:].strip()
                        requirements = requirements.replace(old_req + '\n', '')
                        requirements = requirements.replace(old_req, '')
                    else:
                        # Handle case where LLM doesn't use +/- prefix
                        # Assume it's an addition if not already present
                        if change not in requirements:
                            if requirements and not requirements.endswith('\n'):
                                requirements += '\n'
                            requirements += change + '\n'
            
            # Log the bug fix attempt if agent_logger available
            if self.agent_logger:
                self.agent_logger.log_bug_fix_attempt(
                    attempt_number=len(previous_attempts) + 1,
                    error_info={
                        'error_message': error_message,
                        'function_block_name': function_block.name,
                        'language': language
                    },
                    fix_strategy='llm_debugging',
                    llm_input=llm_input,
                    llm_output=response,
                    fixed_code=fixed_code,
                    success=bool(fixed_code),
                    previous_attempts=previous_attempts
                )
            
            # Also save code versions if we have a logger
            if self.agent_logger and fixed_code:
                self.agent_logger.save_function_block_versions(
                    original_code=function_block.code,
                    fixed_code=fixed_code,
                    version=len(previous_attempts) + 1
                )
            
            return fixed_code, requirements
            
        except Exception as e:
            self.logger.error(f"Error debugging with LLM: {e}")
            
            # Log the error if agent_logger available
            if self.agent_logger:
                self.agent_logger.log_bug_fix_attempt(
                    attempt_number=len(previous_attempts) + 1,
                    error_info={'error_message': error_message},
                    fix_strategy='llm_debugging',
                    llm_input=llm_input if 'llm_input' in locals() else None,
                    llm_output=None,
                    fixed_code=None,
                    success=False,
                    error=str(e),
                    previous_attempts=previous_attempts
                )
            
            return None, None
    
    def _build_debug_prompt(
        self,
        function_block: NewFunctionBlock,
        error_message: str,
        stdout_lines: List[str],
        stderr_lines: List[str],
        previous_attempts: List[str],
        parent_data_structure: Optional[Dict[str, Any]] = None,
        actual_error_lines: Optional[List[str]] = None
    ) -> str:
        """Build a debug prompt for the LLM."""
        prompt_parts = []
        
        prompt_parts.append("## Function Block to Debug")
        prompt_parts.append(f"Name: {function_block.name}")
        prompt_parts.append(f"Description: {function_block.description}")
        prompt_parts.append("")
        prompt_parts.append("## Original Code")
        prompt_parts.append("```python")
        prompt_parts.append(function_block.code)
        prompt_parts.append("```")
        prompt_parts.append("")
        
        # Include actual error traceback FIRST if found
        if actual_error_lines:
            prompt_parts.append("## ACTUAL ERROR (This is the real problem to fix)")
            prompt_parts.append("```")
            prompt_parts.extend(actual_error_lines)
            prompt_parts.append("```")
            prompt_parts.append("")
        
        prompt_parts.append("## Container Exit Information")
        prompt_parts.append(f"Exit Message: {error_message}")
        prompt_parts.append("Note: Focus on the ACTUAL ERROR above, not warnings in stderr")
        prompt_parts.append("")
        
        if stderr_lines:
            prompt_parts.append("## Standard Error Output (last {} lines)".format(len(stderr_lines)))
            prompt_parts.append("```")
            prompt_parts.extend(stderr_lines)
            prompt_parts.append("```")
            prompt_parts.append("")
        
        if stdout_lines:
            prompt_parts.append("## Standard Output (last {} lines)".format(len(stdout_lines)))
            prompt_parts.append("```")
            prompt_parts.extend(stdout_lines)
            prompt_parts.append("```")
            prompt_parts.append("")
        
        if previous_attempts:
            prompt_parts.append("## Previous Fix Attempts (Learn from these failures)")
            prompt_parts.append("The following fixes were already tried and FAILED. Do NOT repeat these approaches:")
            prompt_parts.append("")
            
            for i, attempt in enumerate(previous_attempts, 1):
                if isinstance(attempt, dict):
                    # New structured format
                    prompt_parts.append(f"### Attempt {i} ({attempt.get('timestamp', 'Unknown time')})")
                    
                    # Show the analysis
                    analysis = attempt.get('llm_analysis', {})
                    if analysis:
                        prompt_parts.append(f"- **Error Type**: {analysis.get('error_type', 'Unknown')}")
                        prompt_parts.append(f"- **Root Cause Identified**: {analysis.get('root_cause', 'Unknown')}")
                        prompt_parts.append(f"- **Fix Strategy**: {analysis.get('fix_strategy', 'Unknown')}")
                    
                    # Show what changes were made
                    changes = attempt.get('changes_made', [])
                    if changes:
                        prompt_parts.append("- **Changes Made**:")
                        for change in changes:
                            prompt_parts.append(f"  - {change}")
                    
                    # Show dependency changes if any
                    req_changes = attempt.get('requirements_changes', [])
                    if req_changes:
                        prompt_parts.append("- **Dependency Changes**:")
                        for req_change in req_changes:
                            prompt_parts.append(f"  - {req_change}")
                    
                    # Show why it failed (if we know)
                    if attempt.get('error'):
                        prompt_parts.append(f"- **Why it failed**: {attempt['error']}")
                    elif not attempt.get('success'):
                        prompt_parts.append("- **Result**: This fix did not resolve the issue")
                    
                    prompt_parts.append("")
                else:
                    # Old string format (backward compatibility)
                    prompt_parts.append(f"{i}. {attempt}")
            
            prompt_parts.append("**IMPORTANT**: Learn from these failures. Try a DIFFERENT approach.")
            prompt_parts.append("")
        
        if parent_data_structure:
            prompt_parts.append("## Input Data Structure (from parent node)")
            prompt_parts.append("The input AnnData object has the following structure:")
            import json
            prompt_parts.append("```json")
            prompt_parts.append(json.dumps(parent_data_structure, indent=2))
            prompt_parts.append("```")
            prompt_parts.append("Use this information to understand available columns and data keys.")
            prompt_parts.append("")
        
        prompt_parts.append("## Task")
        prompt_parts.append("Analyze the ACTUAL ERROR and fix the root cause. Return the complete fixed code.")
        prompt_parts.append("")
        prompt_parts.append("Analysis approach:")
        prompt_parts.append("1. What is the actual error? (not warnings)")
        prompt_parts.append("2. Why is this error occurring? (root cause)")
        prompt_parts.append("3. What needs to be changed to fix it? (not just adding checks)")
        prompt_parts.append("")
        prompt_parts.append("Requirements:")
        prompt_parts.append("1. The function MUST be named 'run' with signature: def run(path_dict, params)")
        prompt_parts.append("2. Include all necessary imports inside the function")
        prompt_parts.append("3. Fix the ROOT CAUSE of the error, not just mask it")
        prompt_parts.append("4. NEVER add built-in Python modules (os, sys, pathlib, json, etc.) to requirements")
        prompt_parts.append("5. Maintain the original functionality")
        
        return "\n".join(prompt_parts)
    
    def fix_error(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process() to maintain compatibility with MainAgent.
        
        This method is expected by MainAgent when calling bug_fixer_agent.fix_error()
        """
        return self.process(context)