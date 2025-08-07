#!/usr/bin/env python3
"""Test package installation in the actual Docker container."""

import docker
import json
from pathlib import Path

def test_in_docker():
    """Test package installation inside the Docker container."""
    client = docker.from_env()
    
    # Python script to run inside container
    test_script = '''
import subprocess
import sys

packages = ["palantir", "scFates", "cellrank", "scikit-misc"]
results = {}

print("Testing package installations in Docker container...")
print("=" * 60)

# Check if meson is available
try:
    result = subprocess.run(["meson", "--version"], capture_output=True, text=True)
    meson_version = result.stdout.strip() if result.returncode == 0 else "Not found"
    print(f"Meson version: {meson_version}")
except Exception as e:
    print(f"Meson check failed: {e}")

# Test each package
for pkg in packages:
    print(f"\\nTesting {pkg}...")
    try:
        # Try to install
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            results[pkg] = "Success"
            print(f"✓ {pkg} installed successfully")
        else:
            results[pkg] = f"Failed: {result.stderr[-500:]}"
            print(f"✗ {pkg} failed to install")
            
    except subprocess.TimeoutExpired:
        results[pkg] = "Timeout"
        print(f"✗ {pkg} installation timed out")
    except Exception as e:
        results[pkg] = f"Error: {str(e)}"
        print(f"✗ {pkg} error: {e}")

print("\\n" + "=" * 60)
print("SUMMARY:")
for pkg, status in results.items():
    print(f"{pkg}: {status}")
'''
    
    # Create temporary directory
    temp_dir = Path("/tmp/docker_test")
    temp_dir.mkdir(exist_ok=True)
    
    # Write test script
    script_path = temp_dir / "test_install.py"
    script_path.write_text(test_script)
    
    print("Running package installation test in Docker container...")
    print(f"Using image: ragomics-python:local")
    print("=" * 80)
    
    try:
        # Run container
        container = client.containers.run(
            "ragomics-python:local",
            command=["python", "/workspace/test_install.py"],
            volumes={
                str(temp_dir): {"bind": "/workspace", "mode": "rw"}
            },
            remove=True,
            stdout=True,
            stderr=True
        )
        
        print(container.decode())
        
    except docker.errors.ContainerError as e:
        print(f"Container error: {e}")
        print(f"Exit code: {e.exit_status}")
        print(f"Output: {e.output.decode()}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_in_docker()