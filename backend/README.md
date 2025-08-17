
# DSPY Boss System

A comprehensive autonomous agent management framework built with DSPY, featuring domain-agnostic design for trading, business operations, and general task automation.

## Features

### Core System
- **DSPY Boss with State Machine**: Internal states (idle, awake, restart, stop, executing, researching, thinking, rethink, reflecting)
- **Domain-Agnostic Design**: Handles trading, business operations, and general automation tasks
- **Pydantic BaseModel Classes**: All models are strongly typed with Pydantic
- **Self-Diagnosis**: Uses DSPY's native Python interpreter for system health monitoring

### Agent Management
- **Dual Agent Types**:
  - **Agentic Agents**: Numbered versions of the boss running autonomously using DSPY
  - **Human Agents**: Loaded from YAML configuration with MCP communication
- **Dynamic Scaling**: Boss analyzes workload and creates new workers as needed
- **Autonomous Threading**: Agents run independently on separate threads

### Task Management
- **Async Task Queue**: Built on the `async-queue-manager` library
- **Autonomous Loop**: Task manager runs continuously without closing
- **Mixed Task Types**: Handles both human-awaited and autonomous agent tasks
- **Failure Handling**: Comprehensive retry logic and failure tracking

### Configuration System
- **MCP Servers**: JSON configuration for external service integration
- **Agent Definitions**: YAML configuration for both human and agentic agents
- **Prompt Signatures**: YAML-based DSPY prompt configurations with React agent support

### Integrated MCP Servers
- **Close CRM**: Customer relationship management
- **Slack**: Team communication and notifications
- **LinkedIn**: Professional networking automation
- **Deep Research**: Advanced web research using Firecrawl
- **Web Crawl**: Web search and data extraction

## Installation

1. **Clone and Setup**:
```bash
cd /home/ubuntu/dspy_boss_system
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # (if you create one, or install manually)
```

2. **Install Dependencies**:
```bash
pip install dspy-ai pydantic rich PyYAML loguru requests httpx aiohttp async-queue-manager
```

## Quick Start

### Basic Usage
```bash
# Run the system
python -m dspy_boss.main

# Dry run for testing
python -m dspy_boss.main --dry-run

# Custom config directory
python -m dspy_boss.main --config-dir /path/to/configs
```

### Programmatic Usage
```python
import asyncio
from dspy_boss import DSPYBoss, TaskPriority

async def main():
    # Initialize the boss
    boss = DSPYBoss("configs")
    await boss.start()
    
    # Add a task
    task_id = await boss.add_task(
        name="Market Research",
        description="Research current AI market trends",
        function_name="research",
        parameters={"query": "AI market 2024", "depth": "comprehensive"},
        priority=TaskPriority.HIGH
    )
    
    # Check system status
    status = boss.get_system_status()
    print(f"System Status: {status}")
    
    # Run diagnosis
    health = await boss.run_diagnosis("health")
    print(f"System Health: {health.status}")
    
    # Keep running
    await boss.run_forever()

asyncio.run(main())
```

## Configuration

### MCP Servers (`configs/mcp_servers.json`)
```json
{
  "close_crm": {
    "name": "Close CRM",
    "url": "https://close-mcp-server.klavis.ai/mcp/",
    "instance_id": "7a69be3d-40df-4bfe-8cdc-ca8d15d6d6f0",
    "api_key": "E33hYGfyM2LN5KRkuswV7G765Ga/yoZVctUevP4mG98=",
    "capabilities": ["crm", "contacts", "deals"]
  }
}
```

### Agents (`configs/agents.yaml`)
```yaml
data_analyst:
  name: Data Analyst Agent
  type: agentic
  description: Autonomous agent for data analysis
  capabilities: [data_analysis, visualization, reporting]
  model_name: gpt-4
  prompt_signature: data_analysis_react

research_specialist:
  name: Research Specialist
  type: human
  description: Human expert in research
  capabilities: [research, analysis, reporting]
  contact_method: slack
  contact_details:
    channel: "#research-team"
    user: "@researcher"
```

### Prompt Signatures (`configs/prompts.yaml`)
```yaml
data_analysis_react:
  name: Data Analysis React Agent
  signature: question -> analysis
  description: React agent for data analysis tasks
  is_react_agent: true
  react_steps: 5
  react_tools: [python, pandas, matplotlib]
  input_fields: [question, data]
  output_fields: [analysis, insights, recommendations]
```

## Architecture

### State Machine
The boss operates through a sophisticated state machine:

- **IDLE**: Waiting for tasks
- **AWAKE**: Assessing workload and available resources
- **THINKING**: Analyzing task distribution and agent allocation
- **EXECUTING**: Managing active task execution
- **RESEARCHING**: Investigating issues or gathering information
- **REFLECTING**: Analyzing performance and optimizing operations
- **RETHINK**: Reconsidering current strategy
- **RESTART**: System restart sequence
- **STOP**: Graceful shutdown

### Agent Types

#### Agentic Agents
- Autonomous DSPY-powered agents
- Use configured prompt signatures
- Support both ChainOfThought and ReAct patterns
- Run independently on separate threads
- Auto-scaling based on workload

#### Human Agents
- Interface with humans via MCP servers
- Support Slack and Close CRM communication
- Task assignment with timeout handling
- Response parsing and result integration

### Task Flow
1. **Task Creation**: Tasks defined with priority, timeout, and requirements
2. **Agent Assignment**: Boss selects best available agent based on capabilities
3. **Execution**: Agent processes task using DSPY or human communication
4. **Result Handling**: Success/failure tracking with retry logic
5. **State Updates**: System state updated based on task outcomes

## Self-Diagnosis

The system includes comprehensive self-diagnosis capabilities:

### Health Monitoring
- CPU, memory, and disk usage tracking
- MCP server connection health
- Agent performance metrics
- Automatic issue detection and alerting

### Performance Analysis
- Task completion rates and duration analysis
- Agent utilization and efficiency metrics
- Bottleneck identification
- Optimization recommendations

### Error Investigation
- Pattern recognition in system errors
- Root cause analysis
- Automated remediation suggestions
- Historical trend analysis

## API Reference

### DSPYBoss Class

#### Methods
- `start()`: Initialize and start the system
- `shutdown()`: Graceful system shutdown
- `add_task(name, description, function_name, parameters, priority, timeout, requires_human)`: Add new task
- `get_system_status()`: Get comprehensive system status
- `run_diagnosis(diagnosis_type)`: Run system diagnosis
- `run_forever()`: Run system indefinitely

### Task Management
- Automatic task queuing and prioritization
- Agent assignment based on capabilities and availability
- Retry logic with exponential backoff
- Comprehensive failure tracking and reporting

### Agent Management
- Dynamic agent spawning based on workload
- Idle agent cleanup
- Performance tracking and optimization
- Human-agent communication via MCP servers

## Monitoring and Logging

### Logging
- Structured logging with loguru
- Multiple log levels and outputs
- Automatic log rotation and retention
- Performance and error tracking

### Metrics
- Real-time system metrics collection
- Historical performance data
- Automated alerting on threshold breaches
- Export capabilities for external monitoring

## Development

### Adding New Task Functions
```python
async def custom_task(param1: str, param2: int) -> dict:
    # Your task logic here
    return {"result": "success", "data": param1}

# Register the function
boss.task_manager.register_task_function("custom_task", custom_task)
```

### Adding New Agent Types
Extend the `BaseAgent` class and implement the `execute_task` method:

```python
class CustomAgent(BaseAgent):
    async def execute_task(self, task: TaskDefinition) -> Any:
        # Custom task execution logic
        return {"status": "completed", "result": "custom_result"}
```

### Custom MCP Servers
Add new MCP server configurations to `mcp_servers.json` and implement corresponding capabilities.

## Troubleshooting

### Common Issues
1. **MCP Connection Failures**: Check API keys and network connectivity
2. **Agent Spawn Issues**: Verify DSPY model configuration and API access
3. **Task Queue Stalls**: Review task function implementations for blocking operations
4. **Memory Issues**: Monitor task history cleanup and agent lifecycle management

### Debug Mode
Run with debug logging for detailed system information:
```bash
python -m dspy_boss.main --log-level DEBUG
```

### Health Checks
Use the built-in diagnosis system to identify issues:
```python
health_result = await boss.run_diagnosis("comprehensive")
print(health_result.summary)
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review system logs in the `logs/` directory
- Use the self-diagnosis system for automated issue detection
- Submit issues with detailed system status output
