
# DSPY Boss System

A comprehensive, domain-agnostic task management and agent orchestration system with integrated MCP (Model Context Protocol) server support and real-time dashboard monitoring.

## 🏗️ Architecture

This system consists of two main components:

### Backend (`/backend`)
- **DSPY Boss System**: Python-based task management and agent orchestration
- **MCP Integration**: Support for multiple Model Context Protocol servers
- **State Management**: Internal state machine with persistence
- **Agent Management**: Dynamic agent creation, scaling, and monitoring
- **Task Management**: Priority-based task queue with automatic distribution

### Frontend (`/frontend`)
- **Next.js Dashboard**: Real-time monitoring and control interface
- **Agent Monitoring**: Live agent status and performance metrics
- **Task Management**: Visual task queue management and assignment
- **MCP Server Status**: Real-time MCP server health monitoring
- **System Metrics**: Performance graphs and system health indicators

## 🚀 Features

### Core Capabilities
- ✅ Domain-agnostic workflow management (trading, business operations, etc.)
- ✅ Dynamic agent scaling and load balancing
- ✅ MCP server integration for external tool access
- ✅ Real-time dashboard monitoring
- ✅ Task priority management and automatic distribution
- ✅ Comprehensive logging and error handling
- ✅ Configuration management (JSON/YAML support)
- ✅ Self-diagnosis and system health monitoring

### Dashboard Features
- 📊 Real-time system metrics and performance charts
- 🤖 Agent status grid with live updates
- 📝 Task management with priority visualization
- 🌐 MCP server status monitoring
- ⚙️ System configuration management
- 📈 Performance analytics and trend analysis
- 🔍 Comprehensive logging interface

## 📋 Requirements

### Backend Requirements
```
Python 3.8+
```

### Frontend Requirements
```
Node.js 16+
npm or yarn
```

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd dspy_boss_complete
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
# or
yarn install
```

### 4. Configuration
1. Copy example configurations:
```bash
cp backend/configs/default_config.json backend/configs/config.json
cp backend/configs/agents_config.yaml backend/configs/my_agents.yaml
```

2. Update configurations with your specific settings
3. Configure MCP server URLs in the config files

## 🏃 Running the System

### Start Backend
```bash
cd backend
python -m dspy_boss.main
```

### Start Frontend Dashboard
```bash
cd frontend
npm run dev
# or
yarn dev
```

The dashboard will be available at `http://localhost:3000`

## 📁 Project Structure

```
dspy_boss_complete/
├── backend/
│   ├── dspy_boss/
│   │   ├── __init__.py
│   │   ├── main.py           # Main entry point
│   │   ├── boss.py           # Core DSPY Boss system
│   │   ├── state_machine.py  # Internal state management
│   │   ├── agents.py         # Agent management
│   │   ├── task_manager.py   # Task queue and distribution
│   │   ├── mcp.py           # MCP server integration
│   │   ├── models.py        # Pydantic data models
│   │   ├── config.py        # Configuration management
│   │   └── self_diagnosis.py # System health monitoring
│   ├── configs/
│   │   ├── default_config.json
│   │   └── agents_config.yaml
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Dashboard homepage
│   │   ├── layout.tsx       # App layout
│   │   ├── agents/          # Agent management pages
│   │   ├── tasks/           # Task management pages
│   │   ├── boss/            # Boss system control
│   │   └── api/             # API routes
│   ├── components/
│   │   ├── ui/              # Reusable UI components
│   │   ├── dashboard-card.tsx
│   │   ├── agent-status-grid.tsx
│   │   ├── task-management.tsx
│   │   └── mcp-server-status.tsx
│   ├── package.json
│   └── next.config.js
└── README.md
```

## 🔧 Configuration

### Backend Configuration (`backend/configs/config.json`)
```json
{
  "max_agents": 10,
  "task_queue_size": 1000,
  "mcp_servers": [
    "https://api.example.com/mcp",
    "wss://mcp.example.com/ws"
  ],
  "logging_level": "INFO",
  "state_persistence": true
}
```

### MCP Server Configuration (`backend/configs/agents_config.yaml`)
```yaml
agents:
  - name: "trading_agent"
    type: "trading"
    max_concurrent_tasks: 5
  - name: "analysis_agent"
    type: "analysis"
    max_concurrent_tasks: 3
```

## 🌐 MCP Server Integration

The system supports multiple MCP servers for:
- External API access
- Tool integration
- Data source connections
- Real-time market feeds
- Business system integrations

Configured MCP servers are automatically monitored for health and availability.

## 📈 Monitoring and Logging

- **Real-time Dashboard**: Monitor system status, agent performance, and task queues
- **Comprehensive Logging**: Structured logging with configurable levels
- **Health Checks**: Automatic system diagnostics and error reporting
- **Performance Metrics**: Track throughput, response times, and resource usage

## 🔄 Development Workflow

1. **Backend Development**: Modify Python modules in `/backend/dspy_boss/`
2. **Frontend Development**: Update React components in `/frontend/components/`
3. **Configuration Changes**: Update configs in `/backend/configs/`
4. **Testing**: Run both backend and frontend for integration testing

## 🚦 System States

The DSPY Boss system operates in several states:
- **IDLE**: Waiting for tasks
- **ACTIVE**: Processing tasks
- **SCALING**: Adjusting agent capacity
- **MAINTENANCE**: System diagnostics
- **ERROR**: Error handling and recovery

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test both backend and frontend
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the system logs in the dashboard
2. Review configuration files
3. Consult the diagnostic output
4. Open an issue with system details

## 🚀 Deployment

### Production Deployment
1. Configure production MCP server URLs
2. Set up proper environment variables
3. Use process managers (PM2, systemd) for backend
4. Deploy frontend with proper build optimization
5. Set up monitoring and alerting

### Environment Variables
- `DSPY_CONFIG_PATH`: Path to configuration files
- `MCP_SERVER_URLS`: Comma-separated MCP server URLs
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `NEXTAUTH_URL`: Frontend authentication URL
- `NEXTAUTH_SECRET`: Frontend authentication secret

---

*Built with ❤️ for scalable, intelligent task management*
