"""Enhanced Function Creator Agent with centralized logging when node_dir is not available."""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .base_agent import BaseAgent
from .agent_output_utils import AgentOutputLogger
from ..models import (
    NewFunctionBlock, FunctionBlockType, StaticConfig, Arg,
    InputSpecification, OutputSpecification, FileInfo, FileType
)
from ..llm_service import OpenAIService
from ..utils import setup_logger

logger = setup_logger(__name__)


class EnhancedFunctionCreatorAgent(BaseAgent):
    """Enhanced version that logs to central location when node_dir is unavailable."""
    
    # Include the same FUNCTION_BLOCK_DOCUMENTATION and SYSTEM_PROMPT from original
    FUNCTION_BLOCK_DOCUMENTATION = """# Function Block Implementation Guide..."""  # Same as original
    SYSTEM_PROMPT = """You are an expert bioinformatics function block creator..."""  # Same as original
    
    def __init__(self, llm_service: Optional[OpenAIService] = None):
        super().__init__("function_creator")
        self.llm_service = llm_service
        self.logger = logger
        self.central_log_dir = None  # Will be set when needed
    
    def _get_or_create_central_log_dir(self, base_output_dir: Optional[Path] = None) -> Path:
        """Get or create a central logging directory for function creation."""
        if base_output_dir:
            # Use the tree directory for centralized logs
            central_dir = base_output_dir / "function_creation_logs"
        else:
            # Fallback to temp directory
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "ragomics_function_creation_logs"
            central_dir = temp_dir / datetime.now().strftime("%Y%m%d")
        
        central_dir.mkdir(parents=True, exist_ok=True)
        return central_dir
    
    def process(self, context: Dict[str, Any]) -> Optional[NewFunctionBlock]:
        """Create a new function block based on requirements.
        
        Enhanced to support both node-specific and centralized logging.
        """
        self.validate_context(context, ['task_description', 'user_request'])
        
        if not self.llm_service:
            self.logger.error("No LLM service available for function creation")
            return None
        
        # Determine logging location
        agent_logger = None
        
        if 'node_dir' in context and context['node_dir']:
            # Use node-specific logging (original behavior)
            agent_logger = AgentOutputLogger(context['node_dir'], 'function_creator')
            self.logger.info(f"Logging to node directory: {context['node_dir']}")
        elif 'output_dir' in context and context['output_dir']:
            # Use centralized logging when node doesn't exist yet
            central_dir = self._get_or_create_central_log_dir(Path(context['output_dir']))
            agent_logger = AgentOutputLogger(central_dir, 'function_creator')
            self.logger.info(f"Logging to central directory: {central_dir}")
        else:
            # Create a temporary central log directory
            central_dir = self._get_or_create_central_log_dir()
            agent_logger = AgentOutputLogger(central_dir, 'function_creator')
            self.logger.info(f"Logging to temporary directory: {central_dir}")
        
        try:
            # Build the prompt (same as original)
            prompt = self._build_creation_prompt(context)
            
            # Create messages and schema (same as original)
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": self.FUNCTION_BLOCK_DOCUMENTATION},
                {"role": "user", "content": prompt}
            ]
            
            schema = {
                # Same schema as original
            }
            
            # Prepare LLM input for logging
            llm_input = {
                'messages': messages,
                'schema': schema,
                'temperature': 0.3,
                'max_tokens': 4000,
                'model': self.llm_service.model,
                'timestamp': datetime.now().isoformat()
            }
            
            # Call LLM
            self.logger.info(f"Creating function block with {self.llm_service.model}")
            result = self.llm_service.chat_completion_json(
                messages=messages,
                json_schema=schema,
                temperature=0.3,
                max_tokens=4000
            )
            
            # Log LLM interaction
            if agent_logger:
                log_metadata = {
                    'task_description': context.get('task_description'),
                    'parent_output': context.get('parent_output'),
                    'logging_location': 'node' if 'node_dir' in context else 'central'
                }
                
                # Add function name to metadata if available
                if result and 'name' in result:
                    log_metadata['function_name'] = result['name']
                
                agent_logger.log_llm_interaction(
                    task_type='create_function',
                    llm_input=llm_input,
                    llm_output=result,
                    metadata=log_metadata
                )
            
            # Convert to NewFunctionBlock
            function_block = self._create_function_block(result, context)
            
            # Save function block code version
            if function_block and agent_logger:
                agent_logger.save_function_block_versions(
                    original_code=function_block.code,
                    fixed_code=None,
                    version=1
                )
            
            return function_block
            
        except Exception as e:
            self.logger.error(f"Error creating function block: {e}")
            
            # Log error
            if agent_logger:
                agent_logger.log_llm_interaction(
                    task_type='create_function',
                    llm_input=llm_input if 'llm_input' in locals() else None,
                    llm_output=None,
                    error=str(e),
                    metadata={'context': context, 'logging_location': 'error'}
                )
            
            return None
    
    def _build_creation_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for function block creation (same as original)."""
        # Same implementation as original
        pass
    
    def _create_function_block(self, result: Dict[str, Any], context: Dict[str, Any]) -> NewFunctionBlock:
        """Convert LLM result to NewFunctionBlock (same as original)."""
        # Same implementation as original
        pass
    
    def create_function_block(self, specification: Dict[str, Any], output_dir: Optional[Path] = None) -> Optional[NewFunctionBlock]:
        """Create a function block from specification.
        
        Enhanced to support output_dir for centralized logging.
        
        Args:
            specification: Dictionary with name, description, task, etc.
            output_dir: Optional output directory for centralized logging
            
        Returns:
            NewFunctionBlock if successful, None otherwise
        """
        # Convert specification to context format
        context = {
            "task_description": specification.get("task", specification.get("description", "")),
            "user_request": specification.get("task", specification.get("description", "")),
            "function_name": specification.get("name", "unknown_function"),
            "description": specification.get("description", ""),
            "requirements": specification.get("requirements", ""),
            "parameters": specification.get("parameters", {}),
            "input_type": "adata",
            "output_type": "adata",
            "output_dir": output_dir  # Add output_dir for centralized logging
        }
        
        return self.process(context)