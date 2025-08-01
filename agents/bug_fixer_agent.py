"""Agent responsible for debugging and fixing failed function blocks."""

from typing import Dict, Any, Optional, List
import re

from .base_agent import BaseAgent
from .task_manager import TaskType, TaskStatus
from ..models import NewFunctionBlock, FunctionBlockType
from ..llm_service import OpenAIService
from ..llm_service.prompt_builder import PromptBuilder


class BugFixerAgent(BaseAgent):
    """Agent that debugs and fixes failed function blocks."""
    
    def __init__(self, llm_service: Optional[OpenAIService] = None, task_manager=None):
        super().__init__("bug_fixer", task_manager)
        self.llm_service = llm_service
        self.prompt_builder = PromptBuilder() if llm_service else None
        self.common_fixes = self._load_common_fixes()
        
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
                description=f"Fix error in {function_block.name}: {error_message[:100]}",
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
        if self.llm_service and self.prompt_builder:
            fixed_code = self._debug_with_llm(
                function_block=function_block,
                error_message=error_message,
                stdout=stdout,
                stderr=stderr,
                previous_attempts=previous_attempts,
                task_id=task_id
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
        task_id: Optional[str]
    ) -> Optional[str]:
        """Use LLM to debug the function block."""
        language = "python" if function_block.type == FunctionBlockType.PYTHON else "r"
        
        # Prepare full error message
        full_error = f"{error_message}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        
        # Build debug prompt
        prompt = self.prompt_builder.build_debug_prompt(
            function_block_code=function_block.code,
            error_message=full_error,
            previous_attempts=previous_attempts
        )
        
        try:
            # Call LLM service
            messages = [
                {"role": "system", "content": self.prompt_builder.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_service.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=3000
            )
            
            # Extract code from response
            fixed_code = self.llm_service.extract_code_block(response, language)
            
            # Log the LLM interaction
            if task_id and fixed_code:
                self.log_llm_interaction(
                    prompt=prompt,
                    response=response,
                    model=self.llm_service.model,
                    metadata={
                        'attempt': len(previous_attempts) + 1,
                        'language': language
                    }
                )
            
            return fixed_code
            
        except Exception as e:
            self.logger.error(f"Error debugging with LLM: {e}")
            return None
    
    def _load_common_fixes(self) -> List[Dict[str, Any]]:
        """Load common error patterns and their fixes."""
        return [
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
                'pattern': r"ValueError: .* must be 2D",
                'fix': self._fix_dimension_error
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