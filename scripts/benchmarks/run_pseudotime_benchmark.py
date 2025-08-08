#!/usr/bin/env python3
"""Run pseudotime benchmark test."""

import subprocess
import sys
from pathlib import Path

# Run the test from the correct directory
test_file = Path(__file__).parent / "tests" / "test_pseudotime_benchmark.py"

result = subprocess.run(
    [sys.executable, str(test_file)],
    cwd=Path(__file__).parent,
    capture_output=False,
    text=True
)

sys.exit(result.returncode)