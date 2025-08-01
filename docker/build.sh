#!/bin/bash
# Build Docker images for ragomics-agent-local

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Building ragomics-agent-local Docker images..."
echo "============================================="

# Build Python image
echo "Building Python execution environment..."
docker build -t ragomics-python:local -f "$SCRIPT_DIR/Dockerfile.python" "$SCRIPT_DIR"

if [ $? -eq 0 ]; then
    echo "✓ Python image built successfully"
else
    echo "✗ Failed to build Python image"
    exit 1
fi

# Build R image
echo ""
echo "Building R execution environment..."
docker build -t ragomics-r:local -f "$SCRIPT_DIR/Dockerfile.r" "$SCRIPT_DIR"

if [ $? -eq 0 ]; then
    echo "✓ R image built successfully"
else
    echo "✗ Failed to build R image"
    exit 1
fi

echo ""
echo "Docker images built successfully!"
echo ""
echo "Available images:"
docker images | grep -E "(ragomics-python|ragomics-r)" | head -n 3