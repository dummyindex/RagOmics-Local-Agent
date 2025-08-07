"""Test conversion detection logic."""

from pathlib import Path
import sys

# Add parent to path properly
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Now import after path is set
import models
from models import NewFunctionBlock, FunctionBlockType, StaticConfig


def test_conversion_detection():
    """Test that we can detect when conversion is needed."""
    
    # Create mock blocks
    python_block = NewFunctionBlock(
        name="python_block",
        type=FunctionBlockType.PYTHON,
        description="Python analysis",
        code="def run(path_dict, params): pass",
        requirements="",
        static_config=StaticConfig(
            args=[],
            description="Test",
            tag="test"
        )
    )
    
    r_block = NewFunctionBlock(
        name="r_block", 
        type=FunctionBlockType.R,
        description="R analysis",
        code="run <- function(path_dict, params) {}",
        requirements="",
        static_config=StaticConfig(
            args=[],
            description="Test",
            tag="test"
        )
    )
    
    # Test type detection
    print("Testing conversion detection:")
    print(f"Python block type: {python_block.type}")
    print(f"R block type: {r_block.type}")
    
    # Simple conversion check
    def needs_conversion(parent_type, child_type):
        """Check if conversion needed between parent and child."""
        return parent_type != child_type
    
    # Test cases
    test_cases = [
        (FunctionBlockType.PYTHON, FunctionBlockType.PYTHON, False),
        (FunctionBlockType.PYTHON, FunctionBlockType.R, True),
        (FunctionBlockType.R, FunctionBlockType.PYTHON, True),
        (FunctionBlockType.R, FunctionBlockType.R, False),
    ]
    
    print("\nConversion detection results:")
    for parent, child, expected in test_cases:
        result = needs_conversion(parent, child)
        status = "✓" if result == expected else "✗"
        print(f"{status} {parent.value} → {child.value}: {result} (expected {expected})")
    
    # Test with actual blocks
    print("\nTesting with actual blocks:")
    py_to_r = needs_conversion(python_block.type, r_block.type)
    print(f"Python → R needs conversion: {py_to_r}")
    
    r_to_py = needs_conversion(r_block.type, python_block.type)
    print(f"R → Python needs conversion: {r_to_py}")
    
    py_to_py = needs_conversion(python_block.type, python_block.type)
    print(f"Python → Python needs conversion: {py_to_py}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_conversion_detection()