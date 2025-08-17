
// Mock data for development and testing when Python backend is not available

import { 
  SystemOverview, 
  TaskDefinition, 
  AgentConfig, 
  MCPServerConfig,
  ReportEntry,
  FailureEntry,
  DiagnosisResult,
  BossState,
  TaskQueueSummary,
  AgentSummary,
  MCPServerStatus,
  AgentType,
  TaskStatus,
  TaskPriority
} from './types';

export const mockSystemOverview: SystemOverview = {
  boss_state: BossState.AWAKE,
  state_data: {
    current_workload: 7,
    active_agents: ['agent-1', 'agent-2', 'agent-3'],
    pending_tasks: ['task-1', 'task-2'],
    completed_tasks: ['task-3', 'task-4', 'task-5'],
    failed_tasks: ['task-6'],
    total_tasks_processed: 156,
    success_rate: 92.3,
    average_task_duration: 45.2,
    last_health_check: new Date().toISOString(),
    system_errors: [],
    last_reflection: new Date().toISOString(),
    reflection_notes: 'System performance is optimal',
    improvement_actions: ['Optimize task routing', 'Update agent signatures']
  },
  metrics: {
    timestamp: new Date().toISOString(),
    tasks_per_minute: 3.2,
    average_task_completion_time: 45.2,
    task_success_rate: 92.3,
    active_agents_count: 3,
    agent_utilization: 78.5,
    memory_usage_mb: 512.7,
    cpu_usage_percent: 23.1,
    active_mcp_connections: 5,
    mcp_response_time_avg: 120.5
  },
  health_status: 'healthy',
  uptime: 86400 // 24 hours in seconds
};

export const mockTasks: TaskDefinition[] = [
  {
    id: 'task-1',
    name: 'Market Research Analysis',
    description: 'Analyze current market trends in AI technology',
    priority: TaskPriority.HIGH,
    status: TaskStatus.RUNNING,
    function_name: 'research_market_trends',
    parameters: { industry: 'AI', timeframe: '2024' },
    assigned_agent_id: 'agent-1',
    requires_human: false,
    created_at: new Date().toISOString(),
    started_at: new Date().toISOString(),
    retry_count: 0,
    max_retries: 3
  },
  {
    id: 'task-2',
    name: 'Customer Data Sync',
    description: 'Synchronize customer data from CRM',
    priority: TaskPriority.MEDIUM,
    status: TaskStatus.PENDING,
    function_name: 'sync_customer_data',
    parameters: { source: 'close_crm' },
    requires_human: false,
    created_at: new Date().toISOString(),
    retry_count: 0,
    max_retries: 3
  },
  {
    id: 'task-3',
    name: 'Sales Report Generation',
    description: 'Generate monthly sales performance report',
    priority: TaskPriority.LOW,
    status: TaskStatus.COMPLETED,
    function_name: 'generate_sales_report',
    parameters: { month: 'August', year: 2024 },
    assigned_agent_id: 'agent-2',
    requires_human: true,
    created_at: new Date().toISOString(),
    started_at: new Date().toISOString(),
    completed_at: new Date().toISOString(),
    result: { report_url: '/reports/sales-august-2024.pdf' },
    retry_count: 0,
    max_retries: 3
  }
];

export const mockAgents: AgentConfig[] = [
  {
    id: 'agent-1',
    name: 'Research Specialist',
    type: AgentType.HUMAN,
    description: 'Human expert in research and analysis',
    capabilities: ['research', 'analysis', 'reporting'],
    max_concurrent_tasks: 3,
    is_available: true,
    contact_method: 'slack',
    contact_details: { channel: '#research-team', user: '@researcher' },
    created_at: new Date().toISOString(),
    last_active: new Date().toISOString()
  },
  {
    id: 'agent-2',
    name: 'Sales Manager',
    type: AgentType.HUMAN,
    description: 'Human sales manager for deal management',
    capabilities: ['sales', 'crm', 'customer_relations'],
    max_concurrent_tasks: 3,
    is_available: true,
    contact_method: 'close_crm',
    contact_details: { user_id: 'sales_manager_001' },
    created_at: new Date().toISOString(),
    last_active: new Date().toISOString()
  },
  {
    id: 'agent-3',
    name: 'Data Analyst Agent',
    type: AgentType.AGENTIC,
    description: 'Autonomous agent for data analysis tasks',
    capabilities: ['data_analysis', 'visualization', 'reporting'],
    max_concurrent_tasks: 3,
    is_available: true,
    model_name: 'gpt-4',
    prompt_signature: 'data_analysis_react',
    created_at: new Date().toISOString(),
    last_active: new Date().toISOString()
  }
];

export const mockMCPServers: MCPServerStatus[] = [
  {
    name: 'Close CRM',
    status: 'connected',
    last_connected: new Date().toISOString(),
    response_time: 120
  },
  {
    name: 'Slack',
    status: 'connected',
    last_connected: new Date().toISOString(),
    response_time: 85
  },
  {
    name: 'LinkedIn',
    status: 'disconnected',
    error_message: 'Connection timeout'
  },
  {
    name: 'Deep Research',
    status: 'connected',
    last_connected: new Date().toISOString(),
    response_time: 340
  },
  {
    name: 'Web Crawl',
    status: 'connected',
    last_connected: new Date().toISOString(),
    response_time: 200
  }
];

export const mockLogs: ReportEntry[] = [
  {
    id: 'log-1',
    timestamp: new Date().toISOString(),
    level: 'info',
    category: 'task',
    message: 'Task completed successfully',
    details: { task_id: 'task-3', duration: 45.2 },
    task_id: 'task-3',
    agent_id: 'agent-2'
  },
  {
    id: 'log-2',
    timestamp: new Date().toISOString(),
    level: 'warning',
    category: 'mcp',
    message: 'LinkedIn MCP connection unstable',
    mcp_server: 'linkedin'
  },
  {
    id: 'log-3',
    timestamp: new Date().toISOString(),
    level: 'error',
    category: 'system',
    message: 'Memory usage approaching limit',
    details: { memory_usage_mb: 890.5, limit_mb: 1024 }
  }
];

export const mockFailures: FailureEntry[] = [
  {
    id: 'failure-1',
    timestamp: new Date().toISOString(),
    failure_type: 'mcp_connection',
    description: 'LinkedIn MCP server connection failed',
    error_details: 'Connection timeout after 30 seconds',
    mcp_server: 'linkedin',
    is_resolved: false
  },
  {
    id: 'failure-2',
    timestamp: new Date().toISOString(),
    failure_type: 'task_failure',
    description: 'Task execution failed due to invalid parameters',
    error_details: 'Missing required parameter: industry',
    task_id: 'task-failed-1',
    is_resolved: true,
    resolution_notes: 'Fixed parameter validation',
    resolved_at: new Date().toISOString()
  }
];

export const mockMetricsHistory = Array.from({ length: 24 }, (_, i) => ({
  timestamp: new Date(Date.now() - (23 - i) * 60 * 60 * 1000).toISOString(),
  tasks_per_minute: 2 + Math.random() * 3,
  cpu_usage: 15 + Math.random() * 20,
  memory_usage: 400 + Math.random() * 200,
  active_agents: 2 + Math.floor(Math.random() * 3),
  success_rate: 85 + Math.random() * 10
}));
