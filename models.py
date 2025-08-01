"""Data models for Ragomics Agent Local."""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import uuid


class NodeState(str, Enum):
    """States of analysis nodes."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEBUGGING = "debugging"
    SKIPPED = "skipped"


class FunctionBlockType(str, Enum):
    """Types of function blocks."""
    PYTHON = "python"
    R = "r"


class GenerationMode(str, Enum):
    """Function block generation modes."""
    MIXED = "mixed"  # Can use existing or generate new
    ONLY_NEW = "only_new"  # Only generate new blocks
    ONLY_EXISTING = "only_existing"  # Only use existing blocks


class Arg(BaseModel):
    """Function block argument definition."""
    name: str
    value_type: str
    description: str
    optional: bool = False
    default_value: Optional[Union[str, int, float, bool]] = None
    render_type: str = "text"
    options: Optional[List[Union[str, int]]] = None


class StaticConfig(BaseModel):
    """Static configuration for a function block."""
    args: List[Arg]
    description: str
    tag: str
    document_url: str = ""
    source: str = "generated"
    preset_env: Optional[str] = None


class FunctionBlock(BaseModel):
    """Base class for function blocks."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: FunctionBlockType
    description: str
    
    @field_validator('type', mode='before')
    def validate_type(cls, v):
        if isinstance(v, str):
            return FunctionBlockType(v)
        return v
    parameters: Dict[str, Any] = Field(default_factory=dict)
    static_config: StaticConfig
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class NewFunctionBlock(FunctionBlock):
    """Newly generated function block."""
    code: str
    requirements: str
    rest_task: Optional[str] = None
    new: bool = True


class ExistingFunctionBlock(FunctionBlock):
    """Reference to existing function block."""
    function_block_id: str
    version_id: Optional[str] = None
    rest_task: Optional[str] = None
    new: bool = False


class AnalysisNode(BaseModel):
    """Node in the analysis tree."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    analysis_id: str
    function_block: Union[NewFunctionBlock, ExistingFunctionBlock]
    state: NodeState = NodeState.PENDING
    level: int = 0
    children: List[str] = Field(default_factory=list)
    
    # Execution info
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    
    # Results
    output_data_id: Optional[str] = None
    figures: List[str] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    debug_attempts: int = 0
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True


class AnalysisTree(BaseModel):
    """The hierarchical analysis tree."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    root_node_id: Optional[str] = None
    nodes: Dict[str, AnalysisNode] = Field(default_factory=dict)
    
    # User input
    user_request: str
    input_data_path: str
    
    # Configuration
    max_nodes: int
    max_children_per_node: int
    max_debug_trials: int
    generation_mode: GenerationMode = GenerationMode.MIXED
    llm_model: str = "gpt-4o-2024-08-06"
    
    # State
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        use_enum_values = True
    
    def add_node(self, node: AnalysisNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.id] = node
        self.total_nodes += 1
        if node.parent_id and node.parent_id in self.nodes:
            self.nodes[node.parent_id].children.append(node.id)
    
    def get_node(self, node_id: str) -> Optional[AnalysisNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def update_node_state(self, node_id: str, state: NodeState) -> None:
        """Update node state."""
        if node_id in self.nodes:
            self.nodes[node_id].state = state
            self.nodes[node_id].updated_at = datetime.now()
            
            if state == NodeState.COMPLETED:
                self.completed_nodes += 1
            elif state == NodeState.FAILED:
                self.failed_nodes += 1


class ExecutionResult(BaseModel):
    """Result from executing a function block."""
    success: bool
    output_data_path: Optional[str] = None
    figures: List[str] = Field(default_factory=list)
    logs: str = ""
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))