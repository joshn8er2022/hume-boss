
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
import { mockSystemOverview, mockTasks, mockAgents, mockMCPServers } from '@/lib/mock-data';
import { formatDuration, formatBytes, getHealthScore } from '@/lib/dashboard-utils';

export default function SystemOverviewDashboard() {
  const [overview, setOverview] = useState(mockSystemOverview);
  const [tasks, setTasks] = useState(mockTasks);
  const [agents, setAgents] = useState(mockAgents);
  const [mcpServers, setMcpServers] = useState(mockMCPServers);
  const [isLoading, setIsLoading] = useState(false);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate data updates
      setOverview(prev => ({
        ...prev,
        metrics: {
          ...prev.metrics,
          timestamp: new Date().toISOString(),
          tasks_per_minute: 2 + Math.random() * 2,
          cpu_usage_percent: 15 + Math.random() * 15,
          memory_usage_mb: 400 + Math.random() * 200
        }
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const healthScore = getHealthScore(overview);
  const connectedMCPServers = mcpServers?.filter(server => server?.status === 'connected')?.length || 0;
  const taskStatusCounts = tasks?.reduce((acc, task) => {
    acc[task?.status] = (acc[task?.status] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <div className="space-y-6">
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
