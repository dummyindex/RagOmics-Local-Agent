# Pseudotime Package Installation Challenges

## Summary

The main challenge in running all pseudotime methods in a single function block is **package dependency conflicts and missing build tools** in the Docker container environment.

## Key Findings

### 1. Missing Build Dependencies

**Primary Issue**: The `meson` build system is not properly installed in the Docker container, despite being listed in the Dockerfile.
- scikit-misc (required by scFates) needs meson to compile C/C++ extensions
- The symlink `/usr/bin/meson -> /usr/local/bin/meson` exists but target is missing
- This blocks scFates installation completely

### 2. Package Installation Status

| Package | Installation | Issue |
|---------|-------------|-------|
| scanpy | ✓ Pre-installed | None |
| palantir | ✓ Installs successfully | None |
| scFates | ✗ Fails | Requires scikit-misc → needs meson |
| cellrank | ✓ Installs successfully | None |
| scikit-misc | ✗ Fails | Requires meson build system |

### 3. Why It's Hard to Run in One Function Block

1. **Cascading Failures**: When pip tries to install all packages at once:
   ```
   pip install scanpy palantir scFates cellrank
   ```
   If ANY package fails (like scFates), the entire installation fails and the function block cannot execute.

2. **Build System Requirements**: Some packages require compilation:
   - scikit-misc needs meson + ninja build systems
   - These aren't just Python packages but system-level build tools
   - Docker container must have proper build environment

3. **Dependency Conflicts**: Each pseudotime method has its own ecosystem:
   - palantir: Uses mellon, ml_dtypes
   - scFates: Uses elpigraph-python, scikit-misc, adjustText
   - cellrank: Uses pygpcca, pygam, scvelo
   - Different methods may require incompatible versions of shared dependencies

4. **Container Environment Limitations**:
   - Limited to what's pre-installed in the base image
   - Can't install system packages (apt-get) during runtime
   - Must handle all dependencies via pip only

## Technical Details

### Error Analysis

When the DPT+PAGA node tried to install packages:
```
error: subprocess-exited-with-error
× Preparing metadata (pyproject.toml) did not run successfully.
│ exit code: 1
╰─> meson-python: error: meson executable "meson" not found
```

This shows the package uses meson-python build backend, which requires meson to be available in PATH.

### Docker Image Issue

The Dockerfile includes:
```dockerfile
RUN apt-get install -y meson ninja-build
RUN pip install --no-cache-dir meson>=0.63.3
RUN ln -sf /usr/local/bin/meson /usr/bin/meson
```

But the actual container shows:
```
/usr/bin/meson -> /usr/local/bin/meson (broken symlink)
```

This suggests the pip installation of meson failed or was removed during image optimization.

## Solutions

### 1. Fix Docker Image (Recommended)
Rebuild the Docker image with proper meson installation:
```dockerfile
# Install meson from apt (system package)
RUN apt-get install -y meson ninja-build cmake

# Don't override with pip version, use system meson
# Remove the broken symlink step
```

### 2. Pre-install Pseudotime Packages
Add to requirements-sc.txt:
```
palantir>=1.3.0
cellrank>=2.0.0
# Note: scFates may still fail without proper build tools
```

### 3. Use Separate Function Blocks
Instead of one mega-block, create separate blocks for each method:
- Preprocessing block
- DPT/PAGA block (uses only scanpy)
- Palantir block
- CellRank block
- scFates block (if build issues resolved)
- Visualization block

This way, if one method fails, others can still run.

### 4. Alternative Packages
For methods that won't install:
- Replace scFates with other trajectory inference methods
- Use simpler pseudotime methods that don't require compilation

## Conclusion

The fundamental issue is that **bioinformatics packages often require complex build environments** that aren't always available in containerized settings. The combination of:
- Missing build tools (meson)
- Complex dependency trees
- Compilation requirements
- All-or-nothing pip installation behavior

Makes it very difficult to install multiple pseudotime analysis methods in a single function block. The solution requires either fixing the Docker environment or restructuring the workflow to handle each method separately.