# Test Output Location Fix - Project Root

## Issue Fixed
Test outputs were being created in `tests/test_outputs/` instead of the project root `test_outputs/` directory.

## Solution
Updated `tests/test_utils.py` to return the project root `test_outputs/` directory:

```python
def get_test_output_dir() -> Path:
    """Get the test output directory path.
    
    Returns the path to test_outputs/ directory in project root.
    """
    # Get the project root (ragomics_agent_local - parent of parent of this file)
    project_root = Path(__file__).parent.parent
    
    # Create test_outputs directory in project root
    test_outputs_dir = project_root / "test_outputs"
    test_outputs_dir.mkdir(exist_ok=True)
    
    return test_outputs_dir
```

## Output Location
All test outputs are now correctly stored in:
```
/ragomics_agent_local/test_outputs/
```

Not in:
```
/ragomics_agent_local/tests/test_outputs/  ❌ (old, incorrect location)
```

## Directory Structure
```
ragomics_agent_local/
├── test_outputs/                    # ✅ All test outputs go here
│   ├── clustering_mock_*/          # Mock clustering benchmarks
│   ├── clustering_openai_*/        # OpenAI clustering benchmarks
│   ├── clustering_structure/*/     # Structure compliance tests
│   ├── clustering_main_agent_v2/*/ # Main agent v2 tests
│   └── manual_tree/                # Manual tree structure tests
├── tests/
│   ├── test_utils.py              # Helper that returns project root test_outputs/
│   ├── test_clustering_benchmark.py
│   ├── verify_output_structures.py
│   └── clustering/
│       ├── test_clustering_structure.py
│       └── test_clustering_main_agent_v2.py
└── [other project files...]
```

## Verification
Tests now output to the correct location:
```bash
# From tests directory
cd tests
python test_clustering_benchmark.py
# Output: /ragomics_agent_local/test_outputs/clustering_mock_*/

# From project root
python tests/test_clustering_benchmark.py
# Output: /ragomics_agent_local/test_outputs/clustering_mock_*/

# Verify structures
python tests/verify_output_structures.py
# Checks: /ragomics_agent_local/test_outputs/
```

## Benefits
1. **Consistency with project structure** - test_outputs/ at same level as other project directories
2. **Easy access** - outputs are at project root, not nested in tests/
3. **Cleaner separation** - test code in tests/, test outputs in test_outputs/
4. **Standard practice** - follows common project organization patterns

## Tests Updated
All tests using `get_test_output_dir()` now output to the correct location:
- ✅ `test_clustering_benchmark.py`
- ✅ `test_correct_output_structure.py`
- ✅ `verify_output_structures.py`
- ✅ `clustering/test_clustering_structure.py`
- ✅ `clustering/test_clustering_main_agent_v2.py`

## Cleanup
```bash
# Remove old test outputs from incorrect location
rm -rf tests/test_outputs/

# Clean project root test outputs
rm -rf test_outputs/

# Keep only recent outputs
ls -dt test_outputs/* | tail -n +10 | xargs rm -rf
```