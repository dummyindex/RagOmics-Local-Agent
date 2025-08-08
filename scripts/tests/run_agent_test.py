"""Run agent test from the correct directory."""

import subprocess
import sys
from pathlib import Path

# Get the test file
test_file = Path(__file__).parent / "tests" / "test_agent_conversion_integration.py"

# Run from the parent directory
result = subprocess.run(
    [sys.executable, str(test_file)],
    cwd=Path(__file__).parent,
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

sys.exit(result.returncode)