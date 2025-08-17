
// API client for communicating with the Python DSPY Boss system

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
  MCPServerStatus
} from './types';

// Configuration for the Python backend
const PYTHON_API_BASE = process.env.NEXT_PUBLIC_PYTHON_API_URL || 'http://localhost:8001';

class APIClient {
  private baseURL: string;

  constructor(baseURL: string = PYTHON_API_BASE) {
    this.baseURL = baseURL;
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  // System Overview and Status
  async getSystemOverview(): Promise<SystemOverview> {
    return this.request<SystemOverview>('/api/system/overview');
  }

  async getBossState(): Promise<{ state: BossState; data: any }> {
    return this.request('/api/system/boss-state');
  }

  async setBossState(state: BossState): Promise<{ success: boolean }> {
    return this.request('/api/system/boss-state', {
      method: 'POST',
      body: JSON.stringify({ state }),
    });
  }

  async getSystemHealth(): Promise<DiagnosisResult> {
    return this.request<DiagnosisResult>('/api/system/health');
  }

  async restartSystem(): Promise<{ success: boolean; message: string }> {
    return this.request('/api/system/restart', { method: 'POST' });
  }

  async stopSystem(): Promise<{ success: boolean; message: string }> {
    return this.request('/api/system/stop', { method: 'POST' });
  }

  // Task Management
  async getTasks(status?: string, limit?: number): Promise<TaskDefinition[]> {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    if (limit) params.append('limit', limit.toString());
    
    return this.request<TaskDefinition[]>(`/api/tasks?${params}`);
  }

  async getTaskQueue(): Promise<TaskQueueSummary> {
    return this.request<TaskQueueSummary>('/api/tasks/queue-summary');
  }

  async getTask(taskId: string): Promise<TaskDefinition> {
    return this.request<TaskDefinition>(`/api/tasks/${taskId}`);
  }

  async createTask(task: Partial<TaskDefinition>): Promise<TaskDefinition> {
    return this.request<TaskDefinition>('/api/tasks', {
      method: 'POST',
      body: JSON.stringify(task),
    });
  }

  async cancelTask(taskId: string): Promise<{ success: boolean }> {
    return this.request(`/api/tasks/${taskId}/cancel`, { method: 'POST' });
  }

  async retryTask(taskId: string): Promise<{ success: boolean }> {
    return this.request(`/api/tasks/${taskId}/retry`, { method: 'POST' });
  }

  // Agent Management
  async getAgents(): Promise<AgentConfig[]> {
    return this.request<AgentConfig[]>('/api/agents');
  }

  async getAgentSummary(): Promise<AgentSummary> {
    return this.request<AgentSummary>('/api/agents/summary');
  }

  async getAgent(agentId: string): Promise<AgentConfig> {
    return this.request<AgentConfig>(`/api/agents/${agentId}`);
  }

  async updateAgent(agentId: string, updates: Partial<AgentConfig>): Promise<AgentConfig> {
    return this.request<AgentConfig>(`/api/agents/${agentId}`, {
      method: 'PUT',
      body: JSON.stringify(updates),
    });
  }

  async createAgent(agent: Partial<AgentConfig>): Promise<AgentConfig> {
    return this.request<AgentConfig>('/api/agents', {
      method: 'POST',
      body: JSON.stringify(agent),
    });
  }

  async deleteAgent(agentId: string): Promise<{ success: boolean }> {
    return this.request(`/api/agents/${agentId}`, { method: 'DELETE' });
  }

  async getAgentTasks(agentId: string): Promise<TaskDefinition[]> {
    return this.request<TaskDefinition[]>(`/api/agents/${agentId}/tasks`);
  }

  // MCP Server Management
  async getMCPServers(): Promise<MCPServerConfig[]> {
    return this.request<MCPServerConfig[]>('/api/mcp-servers');
  }

  async getMCPServerStatus(): Promise<MCPServerStatus[]> {
    return this.request<MCPServerStatus[]>('/api/mcp-servers/status');
  }

  async testMCPConnection(serverName: string): Promise<{ success: boolean; message: string }> {
    return this.request(`/api/mcp-servers/${serverName}/test`, { method: 'POST' });
  }

  async restartMCPServer(serverName: string): Promise<{ success: boolean; message: string }> {
    return this.request(`/api/mcp-servers/${serverName}/restart`, { method: 'POST' });
  }

  // Logs and Reports
  async getLogs(level?: string, category?: string, limit?: number): Promise<ReportEntry[]> {
    const params = new URLSearchParams();
    if (level) params.append('level', level);
    if (category) params.append('category', category);
    if (limit) params.append('limit', limit.toString());
    
    return this.request<ReportEntry[]>(`/api/logs?${params}`);
  }

  async getFailures(resolved?: boolean, limit?: number): Promise<FailureEntry[]> {
    const params = new URLSearchParams();
    if (resolved !== undefined) params.append('resolved', resolved.toString());
    if (limit) params.append('limit', limit.toString());
    
    return this.request<FailureEntry[]>(`/api/failures?${params}`);
  }

  async resolveFailure(failureId: string, notes?: string): Promise<{ success: boolean }> {
    return this.request(`/api/failures/${failureId}/resolve`, {
      method: 'POST',
      body: JSON.stringify({ notes }),
    });
  }

  // Metrics and Performance
  async getMetricsHistory(hours?: number): Promise<any[]> {
    const params = new URLSearchParams();
    if (hours) params.append('hours', hours.toString());
    
    return this.request(`/api/metrics/history?${params}`);
  }

  async getPerformanceReport(): Promise<any> {
    return this.request('/api/metrics/performance');
  }

  // Configuration
  async getConfiguration(): Promise<any> {
    return this.request('/api/config');
  }

  async updateConfiguration(config: any): Promise<{ success: boolean }> {
    return this.request('/api/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    });
  }

  // Self-Diagnosis
  async runDiagnosis(type?: string): Promise<DiagnosisResult> {
    return this.request('/api/diagnosis/run', {
      method: 'POST',
      body: JSON.stringify({ type }),
    });
  }
}

// Export singleton instance
export const apiClient = new APIClient();
export default apiClient;
