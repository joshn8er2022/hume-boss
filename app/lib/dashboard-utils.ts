
// Utility functions for the dashboard

import { TaskStatus, BossState, AgentType } from './types';

// Status color mappings
export const getStatusColor = (status: string): string => {
  switch (status?.toLowerCase()) {
    case 'healthy':
    case 'connected':
    case 'completed':
    case 'success':
      return 'text-green-600 bg-green-100 border-green-200';
    case 'warning':
    case 'pending':
    case 'thinking':
    case 'researching':
      return 'text-yellow-600 bg-yellow-100 border-yellow-200';
    case 'error':
    case 'critical':
    case 'failed':
    case 'disconnected':
      return 'text-red-600 bg-red-100 border-red-200';
    case 'running':
    case 'executing':
    case 'awake':
      return 'text-blue-600 bg-blue-100 border-blue-200';
    case 'idle':
    case 'stop':
      return 'text-gray-600 bg-gray-100 border-gray-200';
    default:
      return 'text-gray-600 bg-gray-100 border-gray-200';
  }
};

// Status icon mappings
export const getStatusIcon = (status: string): string => {
  switch (status?.toLowerCase()) {
    case 'healthy':
    case 'connected':
    case 'completed':
      return 'CheckCircle';
    case 'warning':
    case 'pending':
      return 'AlertCircle';
    case 'error':
    case 'critical':
    case 'failed':
    case 'disconnected':
      return 'XCircle';
    case 'running':
    case 'executing':
    case 'awake':
      return 'Play';
    case 'thinking':
    case 'researching':
      return 'Brain';
    case 'idle':
      return 'Pause';
    case 'stop':
      return 'Square';
    default:
      return 'Circle';
  }
};

// Format duration in seconds to human readable
export const formatDuration = (seconds: number): string => {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86400)}d`;
};

// Format bytes to human readable
export const formatBytes = (bytes: number): string => {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return `${Math.round(size * 100) / 100} ${units[unitIndex]}`;
};

// Format percentage
export const formatPercentage = (value: number): string => {
  return `${Math.round(value * 100) / 100}%`;
};

// Get priority label
export const getPriorityLabel = (priority: number): string => {
  switch (priority) {
    case 1: return 'Critical';
    case 2: return 'High';
    case 3: return 'Medium';
    case 4: return 'Low';
    default: return 'Unknown';
  }
};

// Get priority color
export const getPriorityColor = (priority: number): string => {
  switch (priority) {
    case 1: return 'text-red-600 bg-red-100';
    case 2: return 'text-orange-600 bg-orange-100';
    case 3: return 'text-blue-600 bg-blue-100';
    case 4: return 'text-gray-600 bg-gray-100';
    default: return 'text-gray-600 bg-gray-100';
  }
};

// Capitalize first letter
export const capitalize = (str: string): string => {
  return str?.charAt(0)?.toUpperCase() + str?.slice(1);
};

// Format relative time
export const formatRelativeTime = (dateString: string): string => {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  
  const seconds = Math.floor(diffMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (seconds < 60) return `${seconds}s ago`;
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return `${days}d ago`;
};

// Get task status distribution
export const getTaskStatusDistribution = (tasks: any[]) => {
  return tasks?.reduce((acc, task) => {
    acc[task?.status] = (acc[task?.status] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
};

// Get agent type distribution
export const getAgentTypeDistribution = (agents: any[]) => {
  return agents?.reduce((acc, agent) => {
    acc[agent?.type] = (acc[agent?.type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
};

// Check if system needs attention
export const needsAttention = (overview: any): boolean => {
  if (!overview) return false;
  
  return (
    overview?.health_status === 'warning' ||
    overview?.health_status === 'critical' ||
    overview?.state_data?.system_errors?.length > 0 ||
    overview?.metrics?.cpu_usage_percent > 80 ||
    overview?.metrics?.memory_usage_mb > 800
  );
};

// Get health score based on various metrics
export const getHealthScore = (overview: any): number => {
  if (!overview) return 0;
  
  let score = 100;
  
  // Deduct points for system issues
  if (overview?.health_status === 'warning') score -= 20;
  if (overview?.health_status === 'critical') score -= 50;
  
  // Deduct points for high resource usage
  if (overview?.metrics?.cpu_usage_percent > 80) score -= 15;
  if (overview?.metrics?.memory_usage_mb > 800) score -= 15;
  
  // Deduct points for low success rate
  if (overview?.state_data?.success_rate < 90) score -= 10;
  if (overview?.state_data?.success_rate < 80) score -= 20;
  
  // Deduct points for system errors
  score -= (overview?.state_data?.system_errors?.length || 0) * 5;
  
  return Math.max(0, score);
};
