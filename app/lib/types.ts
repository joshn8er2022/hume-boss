
// Types matching the Python DSPY Boss system models

export enum AgentType {
  AGENTIC = 'agentic',
  HUMAN = 'human'
}

export enum TaskStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export enum TaskPriority {
  CRITICAL = 1,
  HIGH = 2,
  MEDIUM = 3,
  LOW = 4
}

export enum BossState {
  IDLE = 'idle',
  AWAKE = 'awake',
  RESTART = 'restart',
  STOP = 'stop',
  EXECUTING = 'executing',
  RESEARCHING = 'researching',
  THINKING = 'thinking',
  RETHINK = 'rethink',
  REFLECTING = 'reflecting'
}

export interface AgentConfig {
  id: string;
  name: string;
  type: AgentType;
  description?: string;
  capabilities: string[];
  max_concurrent_tasks: number;
  is_available: boolean;
  contact_method?: string;
  contact_details?: Record<string, any>;
  model_name?: string;
  prompt_signature?: string;
  thread_id?: string;
  created_at: string;
  last_active?: string;
}

export interface TaskDefinition {
  id: string;
  name: string;
  description: string;
  priority: TaskPriority;
  status: TaskStatus;
  function_name: string;
  parameters: Record<string, any>;
  timeout?: number;
  assigned_agent_id?: string;
  requires_human: boolean;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  result?: any;
  error_message?: string;
  retry_count: number;
  max_retries: number;
}

export interface MCPServerConfig {
  name: string;
  url: string;
  api_key?: string;
  instance_id?: string;
  description?: string;
  capabilities: string[];
  is_active: boolean;
  connection_timeout: number;
  retry_attempts: number;
  headers: Record<string, string>;
  auth_method?: string;
  created_at: string;
  last_connected?: string;
}

export interface ReportEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  category: string;
  message: string;
  details?: Record<string, any>;
  task_id?: string;
  agent_id?: string;
  mcp_server?: string;
}

export interface FailureEntry {
  id: string;
  timestamp: string;
  failure_type: string;
  description: string;
  error_details?: string;
  stack_trace?: string;
  task_id?: string;
  agent_id?: string;
  mcp_server?: string;
  is_resolved: boolean;
  resolution_notes?: string;
  resolved_at?: string;
}

export interface BossStateData {
  current_workload: number;
  active_agents: string[];
  pending_tasks: string[];
  completed_tasks: string[];
  failed_tasks: string[];
  total_tasks_processed: number;
  success_rate: number;
  average_task_duration: number;
  last_health_check?: string;
  system_errors: string[];
  last_reflection?: string;
  reflection_notes?: string;
  improvement_actions: string[];
}

export interface SystemMetrics {
  timestamp: string;
  tasks_per_minute: number;
  average_task_completion_time: number;
  task_success_rate: number;
  active_agents_count: number;
  agent_utilization: number;
  memory_usage_mb: number;
  cpu_usage_percent: number;
  active_mcp_connections: number;
  mcp_response_time_avg: number;
}

export interface DiagnosisResult {
  timestamp: string;
  diagnosis_type: string;
  status: 'healthy' | 'warning' | 'critical';
  summary: string;
  details: Record<string, any>;
  recommendations: string[];
  action_items: string[];
  code_executed?: string;
  execution_output?: string;
  execution_error?: string;
}

// Dashboard specific types
export interface SystemOverview {
  boss_state: BossState;
  state_data: BossStateData;
  metrics: SystemMetrics;
  health_status: 'healthy' | 'warning' | 'critical';
  uptime: number;
}

export interface TaskQueueSummary {
  pending: number;
  running: number;
  completed: number;
  failed: number;
  total: number;
}

export interface AgentSummary {
  total: number;
  active: number;
  human: number;
  agentic: number;
  available: number;
}

export interface MCPServerStatus {
  name: string;
  status: 'connected' | 'disconnected' | 'error';
  last_connected?: string;
  response_time?: number;
  error_message?: string;
}
