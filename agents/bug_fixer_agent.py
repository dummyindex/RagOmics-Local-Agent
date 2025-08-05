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
1. Understanding the actual error from the traceback - ALWAYS look for the real error at the end of stdout
2. Identifying missing dependencies or incorrect imports
3. Fixing logical errors in the code
4. Ensuring proper data handling
5. Fixing syntax errors, especially in f-strings and string formatting
6. Checking for missing data before using it (e.g., X_pca, ground truth columns)
7. Understanding scanpy API requirements (e.g., scatter plot needs column names from .obs)

Common issues to watch for:
- F-string syntax errors with nested quotes/brackets - use intermediate variables instead
- Missing embeddings like X_pca when calculating metrics - check existence first
- Case-sensitive column names in adata.obs - check actual column names
- Missing directories when saving files - create them first
- Scanpy plotting functions expect column names, not numpy arrays
- Use available columns from parent data structure context when provided
- AttributeError with scanpy functions - verify the function exists in the API

Important debugging context:
- When you see AttributeError, it means the function/attribute doesn't exist in that module
- Consider what the code is trying to achieve and which library would provide that functionality
- Check the traceback carefully - the actual error is often at the end of stdout, not in stderr
- Plotting functions often expect specific input types (column names vs arrays)
- Always verify the existence of data (like embeddings) before using it

When analyzing errors:
1. Read the full traceback to understand what function call failed
2. Identify what the function was supposed to do
3. Research which library provides that functionality
4. Consider alternative implementations from appropriate libraries

You will respond in JSON format with your analysis and fixed code."""
    
    def __init__(self, llm_service: Optional[OpenAIService] = None, task_manager=None):
        super().__init__("bug_fixer", task_manager)
        self.llm_service = llm_service
        self.common_fixes = self._load_common_fixes()
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
        
        # First try common fixes
        fixed_code, fixed_requirements = self._try_common_fixes(
            function_block, error_message, stdout, stderr
        )
        
        # Log the common fix attempt if agent_logger available
        if self.agent_logger:
            self.agent_logger.log_bug_fix_attempt(
                attempt_number=1,
                error_info={
                    'error_message': error_message,
                    'stdout_lines': len(stdout.split('\n')) if stdout else 0,
                    'stderr_lines': len(stderr.split('\n')) if stderr else 0
                },
                fix_strategy='common_fixes',
                llm_input={'strategy': 'pattern_matching', 'patterns_checked': len(self.common_fixes)},
                llm_output=None,
                fixed_code=fixed_code,
                success=bool(fixed_code),
                error=None if fixed_code else 'No matching pattern found'
            )
        
        if fixed_code:
            result = {
                'fixed_code': fixed_code,
                'fixed_requirements': fixed_requirements,
                'success': True,
                'reasoning': 'Applied common fix pattern',
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
        
        # If common fixes don't work and we have LLM service, use it
        if self.llm_service:
            fixed_code = self._debug_with_llm(
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
                    'fixed_requirements': function_block.requirements,
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
    ) -> Optional[str]:
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
                            "description": "List of packages to add/remove from requirements (empty if no changes)"
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
            
            # Extract code directly from JSON response
            fixed_code = response.get('fixed_code') if response else None
            
            # Log the bug fix attempt if agent_logger available
            if self.agent_logger:
                self.agent_logger.log_bug_fix_attempt(
                    attempt_number=len(previous_attempts) + 2,  # +2 because common fixes was attempt 1
                    error_info={
                        'error_message': error_message,
                        'function_block_name': function_block.name,
                        'language': language
                    },
                    fix_strategy='llm_debugging',
                    llm_input=llm_input,
                    llm_output=response,
                    fixed_code=fixed_code,
                    success=bool(fixed_code)
                )
            
            # Also save code versions if we have a logger
            if self.agent_logger and fixed_code:
                self.agent_logger.save_function_block_versions(
                    original_code=function_block.code,
                    fixed_code=fixed_code,
                    version=len(previous_attempts) + 2
                )
            
            return fixed_code
            
        except Exception as e:
            self.logger.error(f"Error debugging with LLM: {e}")
            
            # Log the error if agent_logger available
            if self.agent_logger:
                self.agent_logger.log_bug_fix_attempt(
                    attempt_number=len(previous_attempts) + 2,
                    error_info={'error_message': error_message},
                    fix_strategy='llm_debugging',
                    llm_input=llm_input if 'llm_input' in locals() else None,
                    llm_output=None,
                    fixed_code=None,
                    success=False,
                    error=str(e)
                )
            
            return None
    
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
        
        prompt_parts.append("## Error Information")
        prompt_parts.append(f"Error Message: {error_message}")
        prompt_parts.append("")
        
        # Include actual error traceback if found
        if actual_error_lines:
            prompt_parts.append("## Actual Error Traceback (extracted from output)")
            prompt_parts.append("```")
            prompt_parts.extend(actual_error_lines)
            prompt_parts.append("```")
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
            prompt_parts.append("## Previous Fix Attempts")
            for i, attempt in enumerate(previous_attempts, 1):
                prompt_parts.append(f"{i}. {attempt}")
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
        prompt_parts.append("Fix the code above to resolve the error. Return the complete fixed code.")
        prompt_parts.append("")
        prompt_parts.append("Requirements:")
        prompt_parts.append("1. The function MUST be named 'run' with signature: def run(path_dict, params)")
        prompt_parts.append("2. Include all necessary imports inside the function")
        prompt_parts.append("3. Handle the specific error shown above")
        prompt_parts.append("4. NEVER add built-in Python modules (os, sys, pathlib, json, etc.) to requirements")
        prompt_parts.append("4. Maintain the original functionality")
        
        return "\n".join(prompt_parts)
    
    def _load_common_fixes(self) -> List[Dict[str, Any]]:
        """Load common error patterns and their fixes."""
        return [
            {
                'pattern': r"AttributeError: kmeans",
                'fix': self._fix_kmeans_error
            },
            {
                'pattern': r"ModuleNotFoundError: No module named 'louvain'",
                'fix': self._fix_louvain_error  
            },
            {
                'pattern': r"ModuleNotFoundError: No module named '(\w+)'",
                'fix': self._fix_missing_module
            },
            {
                'pattern': r"ImportError: cannot import name '(\w+)'",
                'fix': self._fix_import_error
            },
            {
                'pattern': r"AttributeError: .* has no attribute '(\w+)'",
                'fix': self._fix_attribute_error
            },
            {
                'pattern': r"NameError: name '(\w+)' is not defined",
                'fix': self._fix_name_error
            },
            {
                'pattern': r"TypeError: .* missing \d+ required positional argument",
                'fix': self._fix_argument_error
            },
            {
                'pattern': r"TypeError: '>=' not supported between instances of '.*' and 'dict'",
                'fix': self._fix_parameter_dict_error
            },
            {
                'pattern': r"ValueError: .* must be 2D",
                'fix': self._fix_dimension_error
            },
            {
                'pattern': r"ValueError: `x`, `y`, and potential `color` inputs must all come from either `.obs` or `.var`",
                'fix': self._fix_scanpy_scatter_error
            },
            {
                'pattern': r"Could not find key .* in \.var_names or \.obs\.columns",
                'fix': self._fix_scanpy_color_key_error
            }
        ]
    
    def _try_common_fixes(
        self, 
        function_block: NewFunctionBlock,
        error_message: str,
        stdout: str,
        stderr: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Try to apply common fixes based on error patterns."""
        
        full_error = f"{error_message}\n{stderr}"
        
        for fix_info in self.common_fixes:
            pattern = fix_info['pattern']
            match = re.search(pattern, full_error)
            if match:
                fixed_code, fixed_requirements = fix_info['fix'](
                    function_block, match, full_error
                )
                if fixed_code:
                    return fixed_code, fixed_requirements
        
        return None, None
    
    def _fix_missing_module(
        self, 
        function_block: NewFunctionBlock,
        match: re.Match,
        error: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Fix missing module errors."""
        missing_module = match.group(1)
        
        # Common module mappings
        module_mappings = {
            'sklearn': 'scikit-learn>=1.0.0',
            'cv2': 'opencv-python>=4.5.0',
            'PIL': 'Pillow>=9.0.0',
            'igraph': 'python-igraph>=0.10.0',
            'leidenalg': 'leidenalg>=0.9.0',
            'fa2': 'fa2>=0.3.5',
            'louvain': 'python-louvain>=0.16',
            'umap': 'umap-learn>=0.5.0',
            'numba': 'numba>=0.56.0',
            'scvelo': 'scvelo>=0.2.5',
            'velocyto': 'velocyto>=0.17.0',
            'cellrank': 'cellrank>=1.5.0',
            'scFates': 'scFates>=0.9.0'
        }
        
        # Check if we have a known mapping
        package = module_mappings.get(missing_module, missing_module)
        
        # Add to requirements
        requirements = function_block.requirements or ""
        if package not in requirements:
            if requirements and not requirements.endswith('\n'):
                requirements += '\n'
            requirements += f"{package}\n"
            
        return function_block.code, requirements
    
    def _fix_import_error(
        self, 
        function_block: NewFunctionBlock,
        match: re.Match,
        error: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Fix import errors."""
        # Try alternative import patterns
        code = function_block.code
        
        # Common import fixes
        import_fixes = {
            'scatter': ('from scanpy.pl import scatter', 'import scanpy as sc\nsc.pl.scatter'),
            'read_h5ad': ('from anndata import read_h5ad', 'import anndata as ad\nad.read_h5ad'),
            'AnnData': ('from anndata import AnnData', 'import anndata as ad\nad.AnnData'),
        }
        
        failed_import = match.group(1)
        if failed_import in import_fixes:
            old_pattern, new_pattern = import_fixes[failed_import]
            if old_pattern in code:
                code = code.replace(old_pattern, new_pattern)
                return code, function_block.requirements
        
        return None, None
    
    def _fix_attribute_error(
        self, 
        function_block: NewFunctionBlock,
        match: re.Match,
        error: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Fix attribute errors."""
        code = function_block.code
        
        # Common attribute fixes for scanpy/anndata
        attribute_fixes = {
            # Plotting functions
            'pl.velocity_embedding': 'import scvelo as scv\nscv.pl.velocity_embedding',
            'pl.velocity': 'import scvelo as scv\nscv.pl.velocity',
            'tl.velocity': 'import scvelo as scv\nscv.tl.velocity',
            'pp.moments': 'import scvelo as scv\nscv.pp.moments',
            
            # Data access patterns
            '.raw.X': '.raw.to_adata().X if hasattr(adata, "raw") else adata.X',
            '.n_obs': '.shape[0]',
            '.n_vars': '.shape[1]',
        }
        
        for old, new in attribute_fixes.items():
            if old in code:
                code = code.replace(old, new)
                return code, function_block.requirements
                
        return None, None
    
    def _fix_name_error(
        self, 
        function_block: NewFunctionBlock,
        match: re.Match,
        error: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Fix name errors."""
        undefined_name = match.group(1)
        code = function_block.code
        requirements = function_block.requirements or ""
        
        # Special case: get_param helper function
        if undefined_name == 'get_param':
            # Add the get_param function definition
            get_param_code = """
    # Helper function for parameters
    def get_param(params, key, default):
        val = params.get(key, default)
        if isinstance(val, dict) and 'default_value' in val:
            return val.get('default_value', default)
        return val if val is not None else default
"""
            # Find where to insert it (after adata = sc.read_h5ad...)
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if 'sc.read_h5ad' in line:
                    # Insert after reading data
                    lines.insert(i + 1, get_param_code)
                    return '\n'.join(lines), requirements
            
            # If not found, add after imports
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('import') and not line.strip().startswith('from'):
                    lines.insert(i, get_param_code)
                    return '\n'.join(lines), requirements
        
        # Common undefined names and their fixes
        name_fixes = {
            'np': ('import numpy as np', 'numpy>=1.24.0'),
            'pd': ('import pandas as pd', 'pandas>=2.0.0'),
            'plt': ('import matplotlib.pyplot as plt', 'matplotlib>=3.6.0'),
            'sc': ('import scanpy as sc', 'scanpy>=1.9.0'),
            'ad': ('import anndata as ad', 'anndata>=0.8.0'),
            'scv': ('import scvelo as scv', 'scvelo>=0.2.5'),
            'scf': ('import scFates as scf', 'scFates>=1.0.0'),
            'cr': ('import cellrank as cr', 'cellrank>=2.0.0'),
            'squidpy': ('import squidpy', 'squidpy>=1.2.0'),
        }
        
        if undefined_name in name_fixes:
            import_stmt, package = name_fixes[undefined_name]
            
            # Add import if not present
            if import_stmt not in code:
                lines = code.split('\n')
                # Find where to insert import
                insert_idx = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith('def run('):
                        # Look for existing imports after function definition
                        j = i + 1
                        while j < len(lines) and (lines[j].strip().startswith('import') or 
                                                  lines[j].strip().startswith('from') or
                                                  lines[j].strip() == '' or
                                                  lines[j].strip().startswith('"""')):
                            j += 1
                        insert_idx = j
                        break
                
                if insert_idx > 0:
                    indent = '    '
                    lines.insert(insert_idx, f"{indent}{import_stmt}")
                    code = '\n'.join(lines)
                    
                    # Add package to requirements if needed
                    if package and package not in requirements:
                        if requirements and not requirements.endswith('\n'):
                            requirements += '\n'
                        requirements += f"{package}\n"
                    
                    return code, requirements
        
        return None, None
    
    def _fix_argument_error(
        self, 
        function_block: NewFunctionBlock,
        match: re.Match,
        error: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Fix function argument errors."""
        code = function_block.code
        
        # Common argument fixes
        if 'sc.pp.highly_variable_genes' in code and 'flavor=' not in code:
            code = code.replace(
                'sc.pp.highly_variable_genes(adata)',
                'sc.pp.highly_variable_genes(adata, flavor="seurat")'
            )
            return code, function_block.requirements
            
        if 'sc.pp.neighbors' in code and 'n_neighbors=' not in code:
            code = code.replace(
                'sc.pp.neighbors(adata)',
                'sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)'
            )
            return code, function_block.requirements
            
        return None, None
    
    def _fix_parameter_dict_error(
        self, 
        function_block: NewFunctionBlock,
        match: re.Match,
        error: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Fix parameter extraction errors when params contain nested dicts."""
        code = function_block.code
        
        # Look for incorrect parameter extraction patterns
        lines = code.split('\n')
        fixed_lines = []
        changed = False
        
        for line in lines:
            # Fix incorrect patterns for extracting parameters from nested dicts
            if "params.get(" in line and ("['value_type']" in line or '["value_type"]' in line):
                # This is wrong - trying to access wrong key
                line = line.replace("['value_type']", "['default_value']")
                line = line.replace('["value_type"]', '["default_value"]')
                changed = True
            
            # Look for simple parameter extraction that needs fixing
            elif re.search(r"(\w+)\s*=\s*params\.get\('(\w+)',\s*(\d+)\)", line):
                # Pattern like: min_genes = params.get('min_genes', 200)
                # Need to handle nested dict structure
                param_match = re.search(r"(\w+)\s*=\s*params\.get\('(\w+)',\s*(\w+|\d+)\)", line)
                if param_match:
                    var_name = param_match.group(1)
                    param_key = param_match.group(2)
                    default_val = param_match.group(3)
                    
                    # Replace with proper extraction
                    indent = len(line) - len(line.lstrip())
                    new_line = f"{' ' * indent}{var_name} = params.get('{param_key}', {default_val})"
                    new_line += f"\n{' ' * indent}if isinstance({var_name}, dict) and 'default_value' in {var_name}:"
                    new_line += f"\n{' ' * indent}    {var_name} = {var_name}.get('default_value', {default_val})"
                    fixed_lines.append(new_line)
                    changed = True
                    continue
            
            fixed_lines.append(line)
        
        if changed:
            return '\n'.join(fixed_lines), function_block.requirements
        
        # Alternative fix: Add get_param helper function if not present
        if 'get_param' not in code:
            get_param_code = """    # Helper function for parameters
    def get_param(params, key, default):
        val = params.get(key, default)
        if isinstance(val, dict) and 'default_value' in val:
            return val.get('default_value', default)
        return val if val is not None else default
"""
            # Find where to insert it
            for i, line in enumerate(lines):
                if 'def run(' in line:
                    # Insert after function definition
                    lines.insert(i + 1, get_param_code)
                    
                    # Now replace all params.get calls with get_param
                    for j in range(len(lines)):
                        lines[j] = re.sub(
                            r"params\.get\('(\w+)',\s*([\w\d\.]+)\)",
                            r"get_param(params, '\1', \2)",
                            lines[j]
                        )
                    
                    return '\n'.join(lines), function_block.requirements
        
        return None, None
    
    def _fix_dimension_error(
        self, 
        function_block: NewFunctionBlock,
        match: re.Match,
        error: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Fix dimension-related errors."""
        code = function_block.code
        
        # Add dimension checks
        if 'adata.X' in code and 'scipy.sparse' not in code:
            # Add sparse matrix handling
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if 'def run(' in line:
                    lines.insert(i + 2, '    import scipy.sparse as sp')
                    lines.insert(i + 3, '    # Ensure dense matrix for operations')
                    lines.insert(i + 4, '    if sp.issparse(adata.X):')
                    lines.insert(i + 5, '        adata.X = adata.X.toarray()')
                    code = '\n'.join(lines)
                    return code, function_block.requirements
                    
        return None, None
    
    def fix_error(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process() to maintain compatibility with MainAgent.
        
        This method is expected by MainAgent when calling bug_fixer_agent.fix_error()
        """
        return self.process(context)
    
    def _fix_scanpy_scatter_error(
        self, 
        function_block: NewFunctionBlock,
        match: re.Match,
        error: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Fix scanpy scatter plot API errors.
        
        The sc.pl.scatter function expects column names from .obs or .var, 
        not numpy arrays from .obsm.
        """
        code = function_block.code
        lines = code.split('\n')
        fixed_lines = []
        changed = False
        
        for line in lines:
            # Check for problematic scatter calls with numpy arrays
            if 'sc.pl.scatter' in line and 'adata.obsm' in line:
                # This is the problematic pattern
                changed = True
                
                # Extract the color parameter if present
                color_match = re.search(r"color='(\w+)'", line)
                color_param = color_match.group(1) if color_match else 'kmeans'
                
                # Check if this is for kmeans or agglo
                if 'kmeans' in line.lower() or color_param == 'kmeans':
                    # Replace with proper visualization
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(f"{' ' * indent}# Use PCA visualization for clustering results")
                    fixed_lines.append(f"{' ' * indent}if 'X_pca' in adata.obsm:")
                    fixed_lines.append(f"{' ' * indent}    sc.pl.pca(adata, color='kmeans', title='KMeans Clustering', show=False)")
                    fixed_lines.append(f"{' ' * indent}elif 'X_umap' in adata.obsm:")
                    fixed_lines.append(f"{' ' * indent}    sc.pl.umap(adata, color='kmeans', title='KMeans Clustering', show=False)")
                    fixed_lines.append(f"{' ' * indent}else:")
                    fixed_lines.append(f"{' ' * indent}    # Use matplotlib directly if no embeddings available for scanpy")
                    fixed_lines.append(f"{' ' * indent}    import matplotlib.pyplot as plt")
                    fixed_lines.append(f"{' ' * indent}    import numpy as np")
                    fixed_lines.append(f"{' ' * indent}    if 'X_pca' in adata.obsm:")
                    fixed_lines.append(f"{' ' * indent}        plt.scatter(adata.obsm['X_pca'][:, 0], adata.obsm['X_pca'][:, 1], c=adata.obs['kmeans'], cmap='tab10')")
                    fixed_lines.append(f"{' ' * indent}        plt.title('KMeans Clustering')")
                    fixed_lines.append(f"{' ' * indent}        plt.xlabel('PC1')")
                    fixed_lines.append(f"{' ' * indent}        plt.ylabel('PC2')")
                elif 'agglo' in line.lower() or color_param == 'agglo':
                    # Similar fix for agglomerative clustering
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(f"{' ' * indent}# Use PCA visualization for clustering results")
                    fixed_lines.append(f"{' ' * indent}if 'X_pca' in adata.obsm:")
                    fixed_lines.append(f"{' ' * indent}    sc.pl.pca(adata, color='agglo', title='Agglomerative Clustering', show=False)")
                    fixed_lines.append(f"{' ' * indent}elif 'X_umap' in adata.obsm:")
                    fixed_lines.append(f"{' ' * indent}    sc.pl.umap(adata, color='agglo', title='Agglomerative Clustering', show=False)")
                    fixed_lines.append(f"{' ' * indent}else:")
                    fixed_lines.append(f"{' ' * indent}    # Use matplotlib directly if no embeddings available for scanpy")
                    fixed_lines.append(f"{' ' * indent}    import matplotlib.pyplot as plt")
                    fixed_lines.append(f"{' ' * indent}    import numpy as np")
                    fixed_lines.append(f"{' ' * indent}    if 'X_pca' in adata.obsm:")
                    fixed_lines.append(f"{' ' * indent}        plt.scatter(adata.obsm['X_pca'][:, 0], adata.obsm['X_pca'][:, 1], c=adata.obs['agglo'], cmap='tab10')")
                    fixed_lines.append(f"{' ' * indent}        plt.title('Agglomerative Clustering')")
                    fixed_lines.append(f"{' ' * indent}        plt.xlabel('PC1')")
                    fixed_lines.append(f"{' ' * indent}        plt.ylabel('PC2')")
            # Also check for incorrect ground_truth column
            elif "'ground_truth'" in line or '"ground_truth"' in line:
                # Replace with Cell_type which is actually available
                line = line.replace("'ground_truth'", "'Cell_type'")
                line = line.replace('"ground_truth"', '"Cell_type"')
                changed = True
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        if changed:
            return '\n'.join(fixed_lines), function_block.requirements
        
        return None, None
    
    def _fix_scanpy_color_key_error(
        self,
        function_block: NewFunctionBlock,
        match: re.Match,
        error: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Fix scanpy plotting where color parameter gets Series instead of column name.
        
        Common pattern: sc.pl.umap(adata, color=labels, ...) where labels is a Series
        Should be: sc.pl.umap(adata, color='column_name', ...)
        """
        code = function_block.code
        lines = code.split('\n')
        fixed_lines = []
        changed = False
        
        for line in lines:
            # Check for scanpy plotting with variable as color
            if re.search(r'sc\.pl\.(umap|tsne|pca|draw_graph|scatter)', line) and 'color=' in line:
                # Extract the color variable
                color_match = re.search(r'color=([^,\)]+)', line)
                if color_match:
                    color_var = color_match.group(1).strip()
                    
                    # Check if it's not already a string literal
                    if not (color_var.startswith('"') or color_var.startswith("'")):
                        # This is likely a variable being passed
                        changed = True
                        
                        # Try to find what this variable represents
                        if 'kmeans' in color_var.lower() or 'kmeans' in line.lower():
                            fixed_line = line.replace(f'color={color_var}', "color='kmeans'")
                        elif 'leiden' in color_var.lower() or 'leiden' in line.lower():
                            fixed_line = line.replace(f'color={color_var}', "color='leiden'")
                        elif 'agglo' in color_var.lower() or 'agglomerative' in line.lower():
                            fixed_line = line.replace(f'color={color_var}', "color='agglomerative'")
                        elif 'louvain' in color_var.lower() or 'louvain' in line.lower():
                            fixed_line = line.replace(f'color={color_var}', "color='louvain'")
                        elif 'Cell_type' in code:
                            # Use Cell_type if available
                            fixed_line = line.replace(f'color={color_var}', "color='Cell_type'")
                        else:
                            # Generic fix - assume the variable name indicates the column
                            fixed_line = line.replace(f'color={color_var}', f"color='{color_var}'")
                        
                        fixed_lines.append(fixed_line)
                        continue
            
            fixed_lines.append(line)
        
        if changed:
            return '\n'.join(fixed_lines), function_block.requirements
        
        return None, None
    
    def _fix_kmeans_error(
        self,
        function_block: NewFunctionBlock,
        match: re.Match,
        error: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Fix sc.tl.kmeans error - scanpy doesn't have kmeans, use sklearn."""
        code = function_block.code
        
        # Replace sc.tl.kmeans with sklearn KMeans
        fixed_code = code.replace(
            "sc.tl.kmeans(adata, n_clusters=n_clusters, key_added='kmeans')",
            """from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['X_pca'])"""
        )
        
        # Also handle other variations
        fixed_code = re.sub(
            r"sc\.tl\.kmeans\(([^)]+)\)",
            r"# KMeans replaced with sklearn implementation above",
            fixed_code
        )
        
        if fixed_code != code:
            return fixed_code, function_block.requirements
        return None, None
    
    def _fix_louvain_error(
        self,
        function_block: NewFunctionBlock,
        match: re.Match,
        error: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Fix louvain import error - use leiden instead which is included in scanpy."""
        code = function_block.code
        
        # Replace louvain with leiden
        fixed_code = code.replace("sc.tl.louvain", "sc.tl.leiden")
        fixed_code = fixed_code.replace("'louvain'", "'leiden'")
        fixed_code = fixed_code.replace('"louvain"', '"leiden"')
        
        # Remove explicit louvain imports
        fixed_code = re.sub(r"^\s*import louvain.*$", "", fixed_code, flags=re.MULTILINE)
        fixed_code = re.sub(r"^\s*from louvain import.*$", "", fixed_code, flags=re.MULTILINE)
        
        if fixed_code != code:
            return fixed_code, function_block.requirements
        return None, None