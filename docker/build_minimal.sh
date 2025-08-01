#!/bin/bash

echo "Building minimal ragomics-agent-local Docker images..."
echo "============================================="

# Build Python execution environment
echo "Building minimal Python execution environment..."
docker build -f Dockerfile.python.minimal -t ragomics-python:minimal . || exit 1
echo "✓ Python environment built successfully"

# Build R execution environment  
echo -e "\nBuilding minimal R execution environment..."
docker build -f Dockerfile.r.minimal -t ragomics-r:minimal . || exit 1
echo "✓ R environment built successfully"

echo -e "\nAll minimal images built successfully!"
docker images | grep ragomics