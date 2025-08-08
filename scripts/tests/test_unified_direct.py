#!/usr/bin/env python3
"""Direct test of unified bug fixer without complex imports."""

import os
import sys
from pathlib import Path
import tempfile

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

print("Testing Unified Bug Fixer Directly...")
print("=" * 60)

# Test 1: Import and initialize
try:
    from agents.unified_bug_fixer.unified_bug_fixer import UnifiedBugFixer
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize with GPT model
try:
    bug_fixer = UnifiedBugFixer("gpt-4o-mini", max_turns=3)
    print("✓ Initialized with GPT-4o-mini")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Simple bug fix
print("\nTesting bug fix...")
with tempfile.TemporaryDirectory() as tmpdir:
    workspace = Path(tmpdir)
    
    # Create broken code
    code_file = workspace / "function_block.py"
    code_file.write_text("""
def run(path_dict, params):
    # Missing import
    df = pd.DataFrame({'a': [1, 2, 3]})
    return {"data": df}
""")
    
    try:
        result = bug_fixer.fix_bug(
            node_name="test_node",
            error_message="NameError: name 'pd' is not defined",
            working_directory=workspace
        )
        
        print(f"✓ Fix attempt completed")
        print(f"  Success: {result['success']}")
        print(f"  Turns: {result['turn_count']}")
        print(f"  Cost: ${result.get('cost', 0):.4f}")
        
        if result['success']:
            fixed_code = code_file.read_text()
            if "import pandas as pd" in fixed_code:
                print("✓ Import correctly added!")
            else:
                print("✗ Import not found in fixed code")
        else:
            print(f"✗ Fix failed: {result.get('explanation', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Bug fix error: {e}")
        import traceback
        traceback.print_exc()

print("\nTest complete!")