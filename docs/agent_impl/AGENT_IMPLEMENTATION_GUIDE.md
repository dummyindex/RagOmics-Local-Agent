# Agent Implementation Guide

## Overview

The Ragomics Agent system implements a hierarchical multi-agent architecture for bioinformatics analysis workflows. This guide covers the implementation details, design patterns, and best practices for the agent system.

## Agent Architecture

### Agent Hierarchy

```
Main Agent (Coordinator)
├── Orchestrator Agents
│   ├── Analysis Planning
│   ├── Workflow Coordination
│   └── Result Aggregation
├── Creator Agents
│   ├── Function Block Generation
│   ├── Code Synthesis
│   └── Parameter Optimization
└── Fixer Agents
    ├── Error Diagnosis
    ├── Code Debugging
    └── Solution Implementation
```

### Agent Responsibilities

#### Main Agent
- **Primary Role**: Overall workflow coordination and user interaction
- **Key Responsibilities**:
  - Parse user requests
  - Create analysis tree structure
  - Delegate tasks to specialized agents
  - Monitor execution progress
  - Aggregate results

#### Orchestrator Agents
- **Primary Role**: Manage specific analysis workflows
- **Key Responsibilities**:
  - Plan analysis sequences
  - Coordinate between creator and fixer agents
  - Manage dependencies
  - Track workflow state

#### Creator Agents
- **Primary Role**: Generate function blocks and code
- **Key Responsibilities**:
  - Create new function blocks
  - Generate analysis code
  - Optimize parameters
  - Validate implementations

#### Fixer Agents
- **Primary Role**: Debug and fix execution errors
- **Key Responsibilities**:
  - Analyze error logs
  - Diagnose issues
  - Implement fixes
  - Retry executions

## Implementation Details

### Agent Base Class

```python
class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, llm_service: LLMService, logger: Logger):
        self.llm_service = llm_service
        self.logger = logger
        self.agent_id = str(uuid.uuid4())
        self.created_at = datetime.now()
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute agent task."""
        raise NotImplementedError
    
    def save_interaction(self, prompt: str, response: str):
        """Save LLM interaction to agent_tasks folder."""
        # Implementation details
```

### Task Management

#### Agent Task Structure
```python
@dataclass
class AgentTask:
    task_id: str
    task_type: AgentTaskType
    input_data: Dict[str, Any]
    context: Dict[str, Any]
    parent_task_id: Optional[str]
    created_at: datetime
    priority: int = 0
```

#### Task Queue Management
- Tasks are queued with priorities
- Parallel execution for independent tasks
- Dependency resolution for sequential tasks
- Automatic retry with exponential backoff

### LLM Integration

#### Service Configuration
```python
class LLMService:
    def __init__(self, model: str = "gpt-4o-2024-08-06"):
        self.model = model
        self.max_tokens = 4096
        self.temperature = 0.7
        self.retry_attempts = 3
```

#### Prompt Templates
Agents use structured prompt templates for consistency:

```python
CREATOR_PROMPT_TEMPLATE = """
You are a bioinformatics code creator agent.

Task: {task_description}
Input Data: {input_data}
Expected Output: {expected_output}

Generate a Python function block that:
1. Processes the input data
2. Performs the required analysis
3. Returns the expected output

Code Requirements:
- Use appropriate bioinformatics libraries
- Include error handling
- Add logging statements
- Document parameters
"""
```

### Error Handling

#### Debug Strategy
1. **Initial Execution**: Run function block
2. **Error Detection**: Capture and parse errors
3. **Diagnosis**: Fixer agent analyzes error
4. **Fix Generation**: Create solution
5. **Retry**: Execute fixed code
6. **Iteration**: Repeat up to max_debug_trials

#### Error Categories
- **Syntax Errors**: Code syntax issues
- **Import Errors**: Missing dependencies
- **Runtime Errors**: Execution failures
- **Data Errors**: Input/output mismatches
- **Resource Errors**: Memory/compute limits

## Folder Structure

### Agent Task Organization

```
output_dir/
├── main_TIMESTAMP_TREEID/
│   ├── agent_info.json
│   ├── user_request.txt
│   └── orchestrator_tasks/
│       ├── orch_TIMESTAMP_ID/
│       │   ├── task_info.json
│       │   ├── llm_interactions/
│       │   │   ├── prompt_001.txt
│       │   │   └── response_001.json
│       │   └── results/
│       └── ...
└── tree_TREEID/
    └── nodes/
        └── node_ID/
            ├── agent_tasks/     # LLM interactions only
            │   ├── creator_TIMESTAMP/
            │   │   ├── prompt.txt
            │   │   └── response.json
            │   └── fixer_TIMESTAMP/
            │       ├── error_analysis.json
            │       ├── fix_prompt.txt
            │       └── fix_response.json
            ├── jobs/           # Actual execution
            └── outputs/        # Results
```

### Key Design Principles

1. **Separation of Concerns**
   - `agent_tasks/`: LLM service interactions
   - `jobs/`: Computational execution
   - `outputs/`: Final results

2. **Traceability**
   - All LLM interactions logged
   - Timestamp-based organization
   - Complete audit trail

3. **Isolation**
   - Each agent task in separate folder
   - No cross-contamination
   - Clean rollback capability

## Communication Patterns

### Inter-Agent Communication

```python
class AgentCommunicator:
    """Manages communication between agents."""
    
    async def send_message(
        self, 
        from_agent: str, 
        to_agent: str, 
        message: AgentMessage
    ):
        """Send message between agents."""
        # Queue message
        # Notify recipient
        # Log communication
    
    async def broadcast(
        self,
        from_agent: str,
        message: AgentMessage
    ):
        """Broadcast to all agents."""
        # Send to all registered agents
```

### Message Types

1. **Task Assignment**: Main → Orchestrator
2. **Progress Update**: Any → Main
3. **Error Report**: Executor → Fixer
4. **Solution Ready**: Fixer → Executor
5. **Result Available**: Any → Orchestrator

## State Management

### Agent State

```python
class AgentState(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"
```

### State Transitions

```
IDLE → PROCESSING → COMPLETED
     ↓           ↑
     → ERROR → PROCESSING
```

### Persistence

- State saved to `agent_state.json`
- Checkpoint after each major operation
- Recovery from last checkpoint on failure

## Performance Optimization

### Parallel Execution
- Independent nodes execute in parallel
- Agent tasks queued with priorities
- Resource pool management

### Caching Strategy
- LLM responses cached by hash(prompt)
- Function blocks cached by parameters
- Results cached with TTL

### Resource Management
- Docker container pooling
- Memory limit enforcement
- Timeout handling

## Testing Strategy

### Unit Tests
```python
# tests/test_agents/test_creator_agent.py
def test_creator_agent_generates_valid_code():
    agent = CreatorAgent(mock_llm_service)
    task = create_test_task()
    result = agent.execute(task)
    assert validate_function_block(result.function_block)
```

### Integration Tests
```python
# tests/test_agents/test_agent_coordination.py
def test_main_agent_coordinates_workflow():
    main_agent = MainAgent()
    request = "Perform clustering analysis"
    result = main_agent.process_request(request)
    assert result.tree.completed_nodes > 0
```

### Mock Agents
```python
class MockCreatorAgent(CreatorAgent):
    """Mock creator for testing."""
    
    def execute(self, task):
        return create_mock_function_block()
```

## Debugging Tools

### Agent Inspector
```bash
python -m ragomics_agent_local.tools.agent_inspector \
    --agent-id <id> \
    --show-interactions
```

### LLM Interaction Viewer
```bash
python -m ragomics_agent_local.tools.llm_viewer \
    --task-dir agent_tasks/creator_TIMESTAMP
```

### State Debugger
```bash
python -m ragomics_agent_local.tools.state_debugger \
    --tree-id <id> \
    --show-transitions
```

## Best Practices

### 1. Agent Design
- Keep agents focused on single responsibility
- Use composition over inheritance
- Implement proper error boundaries

### 2. LLM Interactions
- Use structured prompts
- Validate responses
- Implement retry logic
- Cache when possible

### 3. State Management
- Always persist state changes
- Use transactions for multi-step operations
- Implement rollback capability

### 4. Error Handling
- Categorize errors clearly
- Provide actionable error messages
- Log all error details
- Implement graceful degradation

### 5. Testing
- Test each agent in isolation
- Mock external dependencies
- Test error scenarios
- Validate state transitions

## Common Patterns

### Pattern 1: Retry with Backoff
```python
async def execute_with_retry(self, task, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await self.execute(task)
        except Exception as e:
            wait_time = 2 ** attempt
            await asyncio.sleep(wait_time)
            if attempt == max_retries - 1:
                raise
```

### Pattern 2: Pipeline Processing
```python
class AgentPipeline:
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
    
    async def process(self, initial_input):
        result = initial_input
        for agent in self.agents:
            result = await agent.execute(result)
        return result
```

### Pattern 3: Event-Driven Coordination
```python
class EventDrivenCoordinator:
    def __init__(self):
        self.event_handlers = {}
    
    def on(self, event: str, handler: Callable):
        self.event_handlers[event] = handler
    
    async def emit(self, event: str, data: Any):
        if event in self.event_handlers:
            await self.event_handlers[event](data)
```

## Monitoring and Metrics

### Key Metrics
- Task completion rate
- Average execution time
- LLM token usage
- Error rates by type
- Agent utilization

### Logging Standards
```python
# Structured logging format
logger.info("Agent task completed", extra={
    "agent_id": self.agent_id,
    "task_id": task.task_id,
    "duration": duration,
    "status": "success"
})
```

### Health Checks
```python
class AgentHealthCheck:
    async def check_agent_health(self, agent_id: str) -> HealthStatus:
        # Check agent responsiveness
        # Verify resource availability
        # Test LLM connectivity
        return HealthStatus(healthy=True)
```

## Future Enhancements

### Planned Features
1. **Multi-Model Support**: Support for different LLM providers
2. **Agent Learning**: Improve performance based on past executions
3. **Distributed Execution**: Scale across multiple machines
4. **Real-time Collaboration**: Multiple users working together
5. **Visual Workflow Designer**: GUI for creating workflows

### Extension Points
- Custom agent types via plugin system
- User-defined prompt templates
- Custom error handlers
- External tool integration

## References

- [Analysis Tree Structure](ANALYSIS_TREE_STRUCTURE.md)
- [Node Execution Flow](NODE_EXECUTION_FLOW.md)
- [Function Block Specification](FUNCTION_BLOCK_SPEC.md)
- [API Documentation](API.md)