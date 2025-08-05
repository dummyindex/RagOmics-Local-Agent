#!/usr/bin/env python
"""Simple test to find JSON parsing issue."""

import json

# Test if JSON parsing works correctly
test_json = """
{
    "function_block": {
        "name": "test",
        "description": "test",
        "code": "def run(path_dict, params): pass",
    # Load data from path_dict
    input_path = os.path.join(path_dict["input_dir"], "_node_anndata.h5ad")
    if not os.path.exists(input_path):
        h5ad_files = [f for f in os.listdir(path_dict["input_dir"]) if f.endswith(".h5ad")]
        if h5ad_files:
            input_path = os.path.join(path_dict["input_dir"], h5ad_files[0])
    adata = sc.read_h5ad(input_path) if "sc" in locals() or "sc" in globals() else None

        "requirements": ["scanpy"],
        "parameters": {},
        "static_config": {
            "args": [{
                "name": "test",
                "value_type": "int",
                "description": "test",
                "optional": true,
                "default_value": 10
            }],
            "description": "test",
            "tag": "analysis"
        }
    },
    "reasoning": "test"
}
"""

try:
    result = json.loads(test_json)
    print("JSON parsing successful!")
    print(f"optional value: {result['function_block']['static_config']['args'][0]['optional']}")
    print(f"Type: {type(result['function_block']['static_config']['args'][0]['optional'])}")
except Exception as e:
    print(f"JSON parsing failed: {e}")

# Test what happens if we try to eval a string with 'true'
try:
    eval("true")
    print("eval('true') succeeded")
except NameError as e:
    print(f"eval('true') failed: {e}")

# Test what happens if we define true
true = True
false = False
null = None

try:
    eval("true")
    print(f"eval('true') with true defined: {eval('true')}")
except Exception as e:
    print(f"eval('true') still failed: {e}")