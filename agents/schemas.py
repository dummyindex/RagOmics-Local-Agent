"""Schema definitions for agent responses."""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

from ..models import StaticConfig


class FunctionBlockContent(BaseModel):
    """Schema for new function block generation."""
    name: str
    task_description: str  # What the function should do
    parameters: Dict[str, Any] = Field(default_factory=dict)
    new: bool = True
    rest_task: Optional[str] = None


class ExistingFunctionBlockRef(BaseModel):
    """Schema for existing function block reference."""
    id: Union[str, int]
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    new: bool = False
    rest_task: Optional[str] = None


class FunctionBlockRecommendation(BaseModel):
    """Schema for function block recommendations."""
    satisfied: bool
    next_level_function_blocks: List[Union[FunctionBlockContent, ExistingFunctionBlockRef]]
    reasoning: str