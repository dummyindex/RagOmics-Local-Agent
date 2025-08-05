"""Agent responsible for selecting or creating function blocks."""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from .base_agent import BaseAgent
from .agent_output_utils import AgentOutputLogger
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
from ..utils.data_handler import DataHandler


class FunctionSelectorAgent(BaseAgent):
    """Agent that selects existing or creates new function blocks based on analysis needs."""
    
    SYSTEM_PROMPT = """You are an expert bioinformatics analyst specializing in single-cell RNA sequencing data analysis.
Your task is to recommend function blocks that process AnnData objects to fulfill user requests.

You should:
1. Analyze what has been done so far
2. Determine what needs to be done next
3. Recommend appropriate function blocks (either existing or new)
4. Ensure logical progression of the analysis workflow

Common single-cell analysis workflows include:
- Quality control and filtering
- Normalization and scaling  
- Dimensionality reduction (PCA, UMAP, t-SNE)
- Clustering (Leiden, Louvain)
- Differential expression analysis
- Trajectory inference
- Cell type annotation"""
    
    def __init__(self, llm_service: Optional[OpenAIService] = None, task_manager=None, function_creator=None):
        super().__init__("function_selector", task_manager)
        self.llm_service = llm_service
        self.data_handler = DataHandler()
        self.function_creator = function_creator
        self.agent_logger = None  # Will be initialized per node
        
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
            - node_dir: Optional[Path] - Node directory for logging
            
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
        
        # Initialize agent logger if node_dir is provided
        if 'node_dir' in context:
            self.agent_logger = AgentOutputLogger(context['node_dir'], 'function_selector')
        
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
            function_blocks = self._convert_to_function_blocks(recommendation, user_request)
            
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
    
    def _build_selection_prompt(
        self,
        user_request: str,
        current_node: Optional[Any] = None,
        parent_nodes: List[Any] = None,
        max_branches: int = 3,
        generation_mode: GenerationMode = GenerationMode.MIXED,
        data_summary: Optional[Dict[str, Any]] = None,
        rest_task: Optional[str] = None
    ) -> str:
        """Build prompt for function block selection/generation."""
        
        prompt_parts = []
        
        # Add context about the analysis
        prompt_parts.append("## Analysis Context\n")
        prompt_parts.append(f"User Request: {user_request}\n")
        
        if rest_task:
            prompt_parts.append(f"Remaining Tasks: {rest_task}\n")
        
        # Add data summary
        if data_summary:
            prompt_parts.append("\n## Current Data State\n")
            prompt_parts.append(f"- Shape: {data_summary.get('n_obs', 'unknown')} observations × {data_summary.get('n_vars', 'unknown')} variables")
            prompt_parts.append(f"- Observations columns: {', '.join(data_summary.get('obs_columns', []))}")
            prompt_parts.append(f"- Layers: {', '.join(data_summary.get('layers', []))}")
            prompt_parts.append(f"- Embeddings: {', '.join(data_summary.get('obsm_keys', []))}")
            prompt_parts.append(f"- Annotations: {', '.join(data_summary.get('uns_keys', []))}")
        
        # Add parent node context
        if parent_nodes:
            prompt_parts.append("\n## Previous Analysis Steps\n")
            for i, node in enumerate(parent_nodes):
                prompt_parts.append(f"{i+1}. {node.function_block.name}: {node.function_block.description}")
        
        # Add current node if exists
        if current_node:
            prompt_parts.append(f"\nCurrent Step: {current_node.function_block.name}")
            prompt_parts.append(f"Description: {current_node.function_block.description}")
        
        # Add generation instructions
        prompt_parts.append(f"\n## Task\n")
        prompt_parts.append(f"Recommend up to {max_branches} function blocks for the next analysis steps.")
        
        if generation_mode == GenerationMode.ONLY_NEW:
            prompt_parts.append("Generate NEW function blocks with descriptions.")
            prompt_parts.append("For each new block, provide:")
            prompt_parts.append("- name: descriptive snake_case name")
            prompt_parts.append("- task description: what it should do")
            prompt_parts.append("- required parameters and their types")
        elif generation_mode == GenerationMode.ONLY_EXISTING:
            prompt_parts.append("Select from EXISTING function blocks in the library.")
        else:
            prompt_parts.append("You may either recommend NEW function blocks or select EXISTING ones.")
        
        prompt_parts.append("\nConsider:")
        prompt_parts.append("- What analysis steps are needed to fulfill the user request?")
        prompt_parts.append("- What has already been done in previous steps?")
        prompt_parts.append("- What logical next steps would progress toward the goal?")
        prompt_parts.append("- Are we satisfied that the request has been fulfilled?")
        
        return "\n".join(prompt_parts)
    
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
        prompt = self._build_selection_prompt(
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
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "task_description": {"type": "string"},
                                    "parameters": {"type": "object"},
                                    "new": {"type": "boolean"},
                                    "rest_task": {"type": ["string", "null"]}
                                },
                                "required": ["name", "task_description"]
                            }
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
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            # Prepare LLM input for logging
            llm_input = {
                'messages': messages,
                'schema': schema,
                'temperature': 0.7,
                'max_tokens': 4000,
                'model': self.llm_service.model,
                'timestamp': datetime.now().isoformat()
            }
            
            result = self.llm_service.chat_completion_json(
                messages=messages,
                json_schema=schema,
                temperature=0.7,
                max_tokens=4000
            )
            
            # Log LLM interaction if agent_logger is available
            if self.agent_logger:
                self.agent_logger.log_selection_process(
                    available_functions=[],  # In future, pass actual available functions
                    requirements={
                        'user_request': user_request,
                        'generation_mode': generation_mode.value,
                        'max_branches': max_branches,
                        'data_summary': data_summary
                    },
                    llm_input=llm_input,
                    llm_output=result,
                    selected_function=None,  # Will be multiple functions
                    selection_reason=result.get('reasoning', 'Generated function blocks'),
                    error=None
                )
            
            # Ensure reasoning field exists
            if 'reasoning' not in result:
                result['reasoning'] = 'Generated function blocks for analysis'
            
            # Convert to FunctionBlockRecommendation
            recommendation = FunctionBlockRecommendation(**result)
            
            self.logger.info(f"Generated {len(recommendation.next_level_function_blocks)} function blocks")
            self.logger.info(f"Satisfied: {recommendation.satisfied}")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error generating function blocks: {e}")
            
            # Log error if agent_logger is available
            if self.agent_logger:
                self.agent_logger.log_selection_process(
                    available_functions=[],
                    requirements={'user_request': user_request},
                    llm_input=llm_input if 'llm_input' in locals() else None,
                    llm_output=None,
                    selected_function=None,
                    selection_reason=None,
                    error=str(e)
                )
            
            # Return empty recommendation on error
            return FunctionBlockRecommendation(
                satisfied=False,
                next_level_function_blocks=[],
                reasoning=f"Error generating function blocks: {str(e)}"
            )
    
    def _convert_to_function_blocks(
        self, 
        recommendation: FunctionBlockRecommendation,
        user_request: str = "",
        task_dir: Optional[Path] = None
    ) -> List[Union[NewFunctionBlock, ExistingFunctionBlock]]:
        """Convert recommendation to function block objects."""
        
        function_blocks = []
        
        for fb_data in recommendation.next_level_function_blocks:
            if isinstance(fb_data, FunctionBlockContent) or (isinstance(fb_data, dict) and fb_data.get('new', False)):
                # New function block - use function creator
                if self.function_creator:
                    # Get task description
                    task_desc = fb_data.task_description if hasattr(fb_data, 'task_description') else fb_data.get('task_description', '')
                    
                    # Create context for function creator
                    creator_context = {
                        'task_description': task_desc,
                        'user_request': user_request,
                        'parameters': fb_data.parameters if hasattr(fb_data, 'parameters') else fb_data.get('parameters', {}),
                        'node_dir': task_dir  # Pass node_dir for logging (renamed from task_dir)
                    }
                    
                    # Create the function block
                    fb = self.function_creator.process(creator_context)
                    if fb:
                        function_blocks.append(fb)
                    else:
                        self.logger.warning(f"Failed to create function block: {fb_data.name if hasattr(fb_data, 'name') else fb_data.get('name')}")
                else:
                    self.logger.warning("No function creator available, skipping new function block")
                
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
    
    def select_function_block(self, requirements: Dict[str, Any]) -> Optional[Any]:
        """Select an existing function block based on requirements.
        
        Args:
            requirements: Dictionary with requirements for function block
            
        Returns:
            Function block if found, None otherwise
        """
        # For testing purposes, always return None to force creation
        # In real implementation, this would search existing function blocks
        return None