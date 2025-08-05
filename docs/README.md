# RagOmics Agent Local Documentation

## Overview

This directory contains the comprehensive documentation for the RagOmics Agent Local system, a hierarchical multi-agent architecture for automated single-cell RNA sequencing analysis using LLM-guided function blocks with Docker container isolation.

## Documentation Structure

### Core Documentation

1. **[AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md)**
   - System architecture overview
   - Agent descriptions and responsibilities
   - Input/Output specifications
   - Analysis tree integration
   - Agent communication patterns
   - Error handling and recovery strategies
   - Directory structure conventions

2. **[ANALYSIS_TREE_STRUCTURE.md](ANALYSIS_TREE_STRUCTURE.md)**
   - Analysis tree specification
   - Directory structure details
   - Component descriptions
   - File formats (JSON schemas)
   - Key principles
   - Usage examples

3. **[ANALYSIS_IMPLEMENTATION.md](ANALYSIS_IMPLEMENTATION.md)**
   - Core architecture details
   - Execution model (Docker-based isolation)
   - Job executors implementation
   - Standard I/O conventions
   - Parallel execution details
   - Job history tracking

### Implementation Guides

4. **[agent_impl/AGENT_IMPLEMENTATION_GUIDE.md](agent_impl/AGENT_IMPLEMENTATION_GUIDE.md)**
   - Detailed implementation guide
   - Agent design patterns
   - Function block framework
   - Output directory structure
   - Agent logging system
   - Error handling
   - LLM integration
   - Testing infrastructure
   - Best practices
   - Configuration options

5. **[agent_impl/PARALLEL_EXECUTION.md](agent_impl/PARALLEL_EXECUTION.md)**
   - Parallel execution architecture
   - Job pool implementation
   - Dependency management
   - Performance optimization

6. **[agent_impl/PARALLEL_EXECUTION_DESIGN.md](agent_impl/PARALLEL_EXECUTION_DESIGN.md)**
   - Design considerations
   - Execution strategies
   - Resource management

### Development Resources

7. **[agent_impl/DEVELOPMENT_ROADMAP.md](agent_impl/DEVELOPMENT_ROADMAP.md)**
   - Future features
   - Enhancement plans
   - Extension points

8. **[agent_impl/OPENAI_SETUP.md](agent_impl/OPENAI_SETUP.md)**
   - OpenAI API configuration
   - Model selection
   - Usage guidelines

### Summary Documents

9. **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)**
   - Current implementation status
   - Completed features
   - System capabilities
   - Recent updates

### Development Archives

10. **[development/](development/)**
    - Historical development notes
    - Update summaries
    - Architecture evolution
    - Test restructuring details

## Quick Reference

### For Users
- Start with the main [README.md](../README.md) for installation and quick start
- Read [AGENT_ARCHITECTURE.md](AGENT_ARCHITECTURE.md) for system overview
- Check [ANALYSIS_TREE_STRUCTURE.md](ANALYSIS_TREE_STRUCTURE.md) for output structure

### For Developers
- [AGENT_IMPLEMENTATION_GUIDE.md](agent_impl/AGENT_IMPLEMENTATION_GUIDE.md) for detailed implementation
- [agent_impl/PARALLEL_EXECUTION.md](agent_impl/PARALLEL_EXECUTION.md) for parallel execution details
- [agent_impl/DEVELOPMENT_ROADMAP.md](agent_impl/DEVELOPMENT_ROADMAP.md) for future development

### For Testing
- See [tests/README.md](../tests/README.md) for test suite documentation
- Check implementation guide for testing patterns

## Related Documentation

- **[agents/FUNCTION_BLOCK_FRAMEWORK.md](../agents/FUNCTION_BLOCK_FRAMEWORK.md)** - Function block specifications
- **[tests/README.md](../tests/README.md)** - Test suite documentation

## Document Maintenance

Last updated: 2025-08-05

All documentation should be kept in sync with code changes. When updating the codebase:
1. Update relevant documentation
2. Ensure examples still work
3. Update this README if new documents are added