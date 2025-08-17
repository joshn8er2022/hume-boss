
'use client';

import { useState, useEffect } from 'react';
import { 
  Activity, 
  Users, 
  ListTodo, 
  Server, 
  Brain, 
  Zap,
  TrendingUp,
  Clock,
  CheckCircle,
  AlertTriangle
} from 'lucide-react';

import DashboardCard from '@/components/dashboard-card';
import SystemMetricsChart from '@/components/system-metrics-chart';
import TaskDistributionChart from '@/components/task-distribution-chart';
import AgentStatusGrid from '@/components/agent-status-grid';
import MCPServerStatus from '@/components/mcp-server-status';
import DynamicStatusIndicator, { useSystemStatus, SystemState } from '@/components/dynamic-status-indicator';
import { toast } from 'sonner';
import { getHealthScore, formatDuration, formatBytes } from '@/lib/dashboard-utils';

export default function SystemOverviewDashboard() {
  const [overview, setOverview] = useState<any>(null);
  const [tasks, setTasks] = useState<any[]>([]);
  const [agents, setAgents] = useState<any[]>([]);
  const [mcpServers, setMcpServers] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting');
  
  // Use the real-time system status hook
  const { status: systemStatus, lastUpdate } = useSystemStatus();

  // Fetch initial data and set up real-time updates
  useEffect(() => {
    fetchSystemData();
    
    // Set up periodic data refresh
    const interval = setInterval(fetchSystemData, 10000); // Every 10 seconds
    
    return () => clearInterval(interval);
  }, []);

  const fetchSystemData = async () => {
    try {
      setConnectionStatus('connecting');
      
      // Fetch system overview
      const overviewResponse = await fetch('/api/system/overview');
      if (overviewResponse.ok) {
        const overviewData = await overviewResponse.json();
        setOverview(overviewData);
      }
      
      // Fetch agents
      const agentsResponse = await fetch('/api/agents');
      if (agentsResponse.ok) {
        const agentsData = await agentsResponse.json();
        setAgents(agentsData);
      }
      
      // Fetch tasks
      const tasksResponse = await fetch('/api/tasks');
      if (tasksResponse.ok) {
        const tasksData = await tasksResponse.json();
        setTasks(tasksData);
      }
      
      // Fetch MCP servers
      const mcpResponse = await fetch('/api/mcp-servers');
      if (mcpResponse.ok) {
        const mcpData = await mcpResponse.json();
        setMcpServers(mcpData);
      }
      
      setConnectionStatus('connected');
      setIsLoading(false);
      
    } catch (error) {
      console.error('Error fetching system data:', error);
      setConnectionStatus('disconnected');
      toast.error('Failed to connect to backend system');
    }
  };

  const healthScore = getHealthScore(overview);
  const connectedMCPServers = mcpServers?.filter(server => server?.status === 'connected')?.length || 0;
  const taskStatusCounts = tasks?.reduce((acc, task) => {
    acc[task?.status] = (acc[task?.status] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading system data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* System Status Header */}
      <div className="bg-white rounded-lg shadow-sm border p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <DynamicStatusIndicator 
              state={systemStatus as SystemState} 
              size="lg" 
              showLabel={true}
            />
            <div>
              <h2 className="text-2xl font-bold text-gray-900">DSPY Boss System</h2>
              <p className="text-gray-600">
                Last updated: {lastUpdate.toLocaleTimeString()} • 
                Connection: <span className={`font-medium ${
                  connectionStatus === 'connected' ? 'text-green-600' : 
                  connectionStatus === 'connecting' ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  {connectionStatus}
                </span>
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-500">Health Score</div>
            <div className={`text-2xl font-bold ${
              (overview?.health_score || 0) > 80 ? 'text-green-600' : 
              (overview?.health_score || 0) > 60 ? 'text-yellow-600' : 'text-red-600'
            }`}>
              {overview?.health_score || 0}%
            </div>
          </div>
        </div>
      </div>

      {/* Top Stats Row */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        <DashboardCard
          title="System Health"
          value={`${healthScore}%`}
          description="Overall system health score"
          icon={Activity}
          color={healthScore > 80 ? 'green' : healthScore > 60 ? 'yellow' : 'red'}
          trend={{
            value: 5.2,
            label: 'vs last hour',
            positive: true
          }}
        />

        <DashboardCard
          title="Boss State"
          value={overview?.boss_state?.toUpperCase() || 'UNKNOWN'}
          description={`Uptime: ${formatDuration(overview?.uptime || 0)}`}
          icon={Brain}
          color="blue"
        />

        <DashboardCard
          title="Active Tasks"
          value={overview?.state_data?.current_workload || 0}
          description={`${taskStatusCounts?.completed || 0} completed today`}
          icon={ListTodo}
          color="purple"
          trend={{
            value: 12,
            label: 'vs yesterday',
            positive: true
          }}
        />

        <DashboardCard
          title="Success Rate"
          value={`${overview?.state_data?.success_rate?.toFixed(1) || '0.0'}%`}
          description="Task completion success rate"
          icon={CheckCircle}
          color="green"
          trend={{
            value: 2.1,
            label: 'vs last week',
            positive: true
          }}
        />
      </div>

      {/* Second Stats Row */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        <DashboardCard
          title="Active Agents"
          value={overview?.state_data?.active_agents?.length || 0}
          description={`${agents?.length || 0} total configured`}
          icon={Users}
          color="blue"
        />

        <DashboardCard
          title="MCP Servers"
          value={`${connectedMCPServers}/${mcpServers?.length || 0}`}
          description="Connected MCP servers"
          icon={Server}
          color={connectedMCPServers === mcpServers?.length ? 'green' : 'yellow'}
        />

        <DashboardCard
          title="Avg Task Time"
          value={`${overview?.state_data?.average_task_duration?.toFixed(1) || '0.0'}s`}
          description="Average task completion time"
          icon={Clock}
          color="gray"
        />

        <DashboardCard
          title="Memory Usage"
          value={formatBytes((overview?.metrics?.memory_usage_mb || 0) * 1024 * 1024)}
          description={`CPU: ${overview?.metrics?.cpu_usage_percent?.toFixed(1) || '0.0'}%`}
          icon={Activity}
          color={overview?.metrics?.memory_usage_mb > 800 ? 'red' : 'green'}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">System Metrics</h3>
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <TrendingUp className="h-4 w-4" />
              Last 24 hours
            </div>
          </div>
          <SystemMetricsChart />
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">Task Distribution</h3>
            <div className="text-sm text-gray-500">
              {tasks?.length || 0} total tasks
            </div>
          </div>
          <TaskDistributionChart tasks={tasks} />
        </div>
      </div>

      {/* Agent Status Grid */}
      <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Agent Status</h3>
          <div className="text-sm text-gray-500">
            {agents?.filter(a => a?.is_available)?.length || 0} available
          </div>
        </div>
        <AgentStatusGrid agents={agents} />
      </div>

      {/* MCP Server Status */}
      <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">MCP Server Status</h3>
          <div className="text-sm text-gray-500">
            {connectedMCPServers} of {mcpServers?.length || 0} connected
          </div>
        </div>
        <MCPServerStatus servers={mcpServers} />
      </div>

      {/* Recent Activity */}
      <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
          <div className="text-sm text-blue-600 hover:text-blue-700 cursor-pointer">
            View all logs
          </div>
        </div>
        <div className="space-y-3">
          <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg border border-green-100">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">Task completed successfully</p>
              <p className="text-xs text-gray-500">Sales Report Generation - 2 minutes ago</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg border border-blue-100">
            <Zap className="h-4 w-4 text-blue-600" />
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">New task assigned</p>
              <p className="text-xs text-gray-500">Market Research Analysis to Research Specialist - 5 minutes ago</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3 p-3 bg-yellow-50 rounded-lg border border-yellow-100">
            <AlertTriangle className="h-4 w-4 text-yellow-600" />
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">MCP connection warning</p>
              <p className="text-xs text-gray-500">LinkedIn MCP server response time high - 8 minutes ago</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
