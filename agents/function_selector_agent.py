"""Agent responsible for selecting or creating function blocks."""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .base_agent import BaseAgent
from .schemas import (
    FunctionBlockContent, ExistingFunctionBlockRef, 
    FunctionBlockRecommendation
)
from .task_manager import TaskType, TaskStatus
from ..models import (
    GenerationMode, FunctionBlock, NewFunctionBlock, 
    ExistingFunctionBlock, AnalysisNode, NodeState,
    FunctionBlockType, StaticConfig
)
from ..llm_service import OpenAIService
from ..llm_service.prompt_builder import PromptBuilder
from ..utils.data_handler import DataHandler


class FunctionSelectorAgent(BaseAgent):
    """Agent that selects existing or creates new function blocks based on analysis needs."""
    
    def __init__(self, llm_service: Optional[OpenAIService] = None, task_manager=None):
        super().__init__("function_selector", task_manager)
        self.llm_service = llm_service
        self.data_handler = DataHandler()
        self.prompt_builder = PromptBuilder()
        
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Select or create function blocks for the next analysis step.
        
        Required context keys:
            - user_request: str
            - tree: AnalysisTree
            - current_node: Optional[AnalysisNode]
            - parent_chain: List[AnalysisNode]
            - generation_mode: GenerationMode
            - max_children: int
            - data_path: Optional[Path]
            
        Returns:
            Dict with:
                - function_blocks: List[FunctionBlock]
                - satisfied: bool
                - reasoning: str
        """
        self.validate_context(context, [
            'user_request', 'tree', 'generation_mode', 'max_children'
        ])
        
        user_request = context['user_request']
        tree = context['tree']
        current_node = context.get('current_node')
        parent_chain = context.get('parent_chain', [])
        generation_mode = context['generation_mode']
        max_children = context['max_children']
        data_path = context.get('data_path')
        
        # Create task if we have a task manager
        task_id = None
        if self.task_manager:
            task_context = {
                'analysis_id': tree.id,
                'node_id': current_node.id if current_node else None,
                'user_request': user_request,
                'generation_mode': generation_mode.value
            }
            
            task_id = self.create_task(
                task_type=TaskType.FUNCTION_SELECTION,
                description=f"Select function blocks for {'node ' + current_node.id if current_node else 'root'}",
                context=task_context,
                parent_task_id=context.get('parent_task_id')
            )
            
            self.update_task_status(TaskStatus.IN_PROGRESS)
        
        # Get data summary if available
        data_summary = {}
        if data_path and Path(data_path).exists():
            try:
                adata = self.data_handler.load_data(Path(data_path))
                data_summary = self.data_handler.get_data_summary(adata)
            except Exception as e:
                self.logger.warning(f"Failed to load data for summary: {e}")
        
        # Determine rest task from current node
        rest_task = None
        if current_node and hasattr(current_node.function_block, 'rest_task'):
            rest_task = current_node.function_block.rest_task
        
        # Use LLM service if available
        if self.llm_service:
            recommendation = self._generate_function_blocks(
                user_request=user_request,
                current_node=current_node,
                parent_nodes=parent_chain,
                max_branches=max_children,
                generation_mode=generation_mode,
                data_summary=data_summary,
                rest_task=rest_task,
                task_id=task_id
            )
            
            # Convert recommendation to function blocks
            function_blocks = self._convert_to_function_blocks(recommendation)
            
            result = {
                'function_blocks': function_blocks,
                'satisfied': recommendation.satisfied,
                'reasoning': recommendation.reasoning
            }
            
            if task_id:
                self.update_task_status(TaskStatus.COMPLETED, results=result)
            
            return result
        else:
            # Mock implementation for testing
            return self._mock_select_function_blocks(context)
    
    def _generate_function_blocks(
        self,
        user_request: str,
        current_node: Optional[Any] = None,
        parent_nodes: List[Any] = None,
        max_branches: int = 3,
        generation_mode: GenerationMode = GenerationMode.MIXED,
        data_summary: Optional[Dict[str, Any]] = None,
        rest_task: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> FunctionBlockRecommendation:
        """Generate or select function blocks for next analysis steps."""
        
        # Build the prompt
        prompt = self.prompt_builder.build_generation_prompt(
            user_request=user_request,
            current_node=current_node,
            parent_nodes=parent_nodes or [],
            max_branches=max_branches,
            generation_mode=generation_mode,
            data_summary=data_summary or {},
            rest_task=rest_task
        )
        
        # Determine response schema based on generation mode
        if generation_mode == GenerationMode.ONLY_NEW:
            schema = {
                "name": "function_block_recommendation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "satisfied": {"type": "boolean"},
                        "next_level_function_blocks": {
                            "type": "array",
                            "items": FunctionBlockContent.model_json_schema()
                        },
                        "reasoning": {"type": "string"}
                    },
                    "required": ["satisfied", "next_level_function_blocks", "reasoning"]
                }
            }
        elif generation_mode == GenerationMode.ONLY_EXISTING:
            schema = {
                "name": "function_block_recommendation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "satisfied": {"type": "boolean"},
                        "next_level_function_blocks": {
                            "type": "array",
                            "items": ExistingFunctionBlockRef.model_json_schema()
                        },
                        "reasoning": {"type": "string"}
                    },
                    "required": ["satisfied", "next_level_function_blocks", "reasoning"]
                }
            }
        else:  # MIXED mode
            schema = {
                "name": "function_block_recommendation",
                "schema": FunctionBlockRecommendation.model_json_schema()
            }
        
        try:
            # Call OpenAI API
            self.logger.info(f"Generating function blocks with {self.llm_service.model}")
            
            messages = [
                {"role": "system", "content": self.prompt_builder.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            result = self.llm_service.chat_completion_json(
                messages=messages,
                json_schema=schema,
                temperature=0.7,
                max_tokens=4000
            )
            
            # Log LLM interaction if we have a task
            if task_id:
                self.log_llm_interaction(
                    prompt=prompt,
                    response=str(result),
                    model=self.llm_service.model,
                    metadata={'generation_mode': generation_mode.value}
                )
            
            # Convert to FunctionBlockRecommendation
            recommendation = FunctionBlockRecommendation(**result)
            
            self.logger.info(f"Generated {len(recommendation.next_level_function_blocks)} function blocks")
            self.logger.info(f"Satisfied: {recommendation.satisfied}")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error generating function blocks: {e}")
            # Return empty recommendation on error
            return FunctionBlockRecommendation(
                satisfied=False,
                next_level_function_blocks=[],
                reasoning=f"Error generating function blocks: {str(e)}"
            )
    
    def _convert_to_function_blocks(
        self, 
        recommendation: FunctionBlockRecommendation
    ) -> List[Union[NewFunctionBlock, ExistingFunctionBlock]]:
        """Convert recommendation to function block objects."""
        
        function_blocks = []
        
        for fb_data in recommendation.next_level_function_blocks:
            if isinstance(fb_data, FunctionBlockContent) or (isinstance(fb_data, dict) and fb_data.get('new', False)):
                # New function block
                # Determine language from code
                code = fb_data.function_block_code if hasattr(fb_data, 'function_block_code') else fb_data['function_block_code']
                fb_type = FunctionBlockType.R if 'library(' in code or '<-' in code else FunctionBlockType.PYTHON
                
                static_config_data = fb_data.static_config_file_content if hasattr(fb_data, 'static_config_file_content') else fb_data['static_config_file_content']
                if isinstance(static_config_data, dict):
                    static_config = StaticConfig(**static_config_data)
                else:
                    static_config = static_config_data
                
                fb = NewFunctionBlock(
                    name=fb_data.name if hasattr(fb_data, 'name') else fb_data['name'],
                    type=fb_type,
                    description=static_config.description,
                    code=code,
                    requirements=fb_data.requirements_file_content if hasattr(fb_data, 'requirements_file_content') else fb_data['requirements_file_content'],
                    parameters=fb_data.parameters if hasattr(fb_data, 'parameters') else fb_data.get('parameters', {}),
                    static_config=static_config,
                    rest_task=fb_data.rest_task if hasattr(fb_data, 'rest_task') else fb_data.get('rest_task')
                )
                function_blocks.append(fb)
                
            else:
                # Existing function block reference
                # For now, we'll create a placeholder since we don't have a function block library
                self.logger.warning(f"Existing function block requested but not implemented: {fb_data}")
                
        return function_blocks
    
    def _mock_select_function_blocks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock function block selection for testing without LLM."""
        # This will be overridden in tests
        return {
            'function_blocks': [],
            'satisfied': False,
            'reasoning': 'Mock selection - no LLM service available'
        }
    
    def determine_next_steps(
        self, 
        completed_nodes: List[AnalysisNode],
        user_request: str,
        data_summary: Dict[str, Any]
    ) -> List[str]:
        """Determine what analysis steps should come next.
        
        Args:
            completed_nodes: List of completed analysis nodes
            user_request: Original user request
            data_summary: Current data summary
            
        Returns:
            List of suggested next step descriptions
        """
        # Analyze what has been done
        completed_steps = [node.function_block.name for node in completed_nodes]
        
        # Basic heuristics for common workflows
        suggestions = []
        
        # Check if basic preprocessing is done
        has_qc = any('quality' in step.lower() or 'qc' in step.lower() for step in completed_steps)
        has_norm = any('normal' in step.lower() for step in completed_steps)
        has_hvg = any('variable' in step.lower() or 'hvg' in step.lower() for step in completed_steps)
        has_dim_reduction = any(
            term in step.lower() 
            for step in completed_steps 
            for term in ['pca', 'umap', 'tsne', 'dimensionality']
        )
        
        # Suggest next steps based on what's missing
        if not has_qc:
            suggestions.append("quality_control")
        elif not has_norm:
            suggestions.append("normalization")
        elif not has_hvg:
            suggestions.append("highly_variable_genes")
        elif not has_dim_reduction:
            suggestions.append("dimensionality_reduction")
        
        # Check for specific analysis requests
        request_lower = user_request.lower()
        if 'velocity' in request_lower and not any('velocity' in step.lower() for step in completed_steps):
            suggestions.append("rna_velocity_analysis")
        if 'cluster' in request_lower and not any('cluster' in step.lower() for step in completed_steps):
            suggestions.append("clustering")
        if 'trajectory' in request_lower and not any('trajectory' in step.lower() for step in completed_steps):
            suggestions.append("trajectory_inference")
        if 'differential' in request_lower and not any('differential' in step.lower() for step in completed_steps):
            suggestions.append("differential_expression")
            
        return suggestions