
'use client';

import { useState, useEffect } from 'react';
import { 
  Users, 
  Bot, 
  Plus, 
  Edit, 
  Trash2, 
  MessageCircle, 
  Clock, 
  Settings,
  CheckCircle,
  XCircle,
  Play,
  Pause
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import DashboardCard from '@/components/dashboard-card';
import { AgentConfig, AgentType } from '@/lib/types';
import { mockAgents, mockTasks } from '@/lib/mock-data';
import { formatRelativeTime, getStatusColor, capitalize } from '@/lib/dashboard-utils';
import { useToast } from '@/hooks/use-toast';

export default function AgentManagement() {
  const { toast } = useToast();
  const [agents, setAgents] = useState(mockAgents);
  const [tasks, setTasks] = useState(mockTasks);
  const [selectedAgent, setSelectedAgent] = useState<AgentConfig | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const humanAgents = agents?.filter(agent => agent?.type === AgentType.HUMAN) || [];
  const agenticAgents = agents?.filter(agent => agent?.type === AgentType.AGENTIC) || [];
  const availableAgents = agents?.filter(agent => agent?.is_available) || [];

  const getAgentTasks = (agentId: string) => {
    return tasks?.filter(task => task?.assigned_agent_id === agentId) || [];
  };

  const getAgentCurrentWorkload = (agentId: string) => {
    return tasks?.filter(task => 
      task?.assigned_agent_id === agentId && 
      (task?.status === 'running' || task?.status === 'pending')
    )?.length || 0;
  };

  const handleToggleAvailability = async (agentId: string) => {
    setIsLoading(true);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setAgents(prev => prev?.map(agent => 
        agent?.id === agentId 
          ? { ...agent, is_available: !agent?.is_available }
          : agent
      ));

      const agent = agents?.find(a => a?.id === agentId);
      toast({
        title: "Agent Updated",
        description: `${agent?.name} is now ${!agent?.is_available ? 'available' : 'unavailable'}`,
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Update Failed",
        description: "Failed to update agent availability",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteAgent = async (agentId: string) => {
    setIsLoading(true);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setAgents(prev => prev?.filter(agent => agent?.id !== agentId));

      toast({
        title: "Agent Deleted",
        description: "Agent has been removed from the system",
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Delete Failed",
        description: "Failed to delete agent",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Stats Row */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        <DashboardCard
          title="Total Agents"
          value={agents?.length || 0}
          description="All configured agents"
          icon={Users}
          color="blue"
        />
        
        <DashboardCard
          title="Human Agents"
          value={humanAgents?.length || 0}
          description="Human team members"
          icon={Users}
          color="green"
        />
        
        <DashboardCard
          title="Agentic Agents"
          value={agenticAgents?.length || 0}
          description="AI-powered agents"
          icon={Bot}
          color="purple"
        />
        
        <DashboardCard
          title="Available"
          value={availableAgents?.length || 0}
          description="Ready for tasks"
          icon={CheckCircle}
          color="green"
        />
      </div>

      {/* Actions Bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-semibold text-gray-900">All Agents</h2>
          <Badge variant="outline" className="bg-blue-50 text-blue-700">
            {agents?.length} total
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={() => console.log('Add agent')}>
            <Plus className="h-4 w-4 mr-2" />
            Add Agent
          </Button>
        </div>
      </div>

      {/* Agent Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {agents?.map((agent) => {
          const currentWorkload = getAgentCurrentWorkload(agent?.id);
          const agentTasks = getAgentTasks(agent?.id);
          const completedTasks = agentTasks?.filter(task => task?.status === 'completed')?.length || 0;
          
          return (
            <div 
              key={agent?.id}
              className="p-6 bg-white border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-shadow"
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  {agent?.type === AgentType.HUMAN ? (
                    <div className="flex items-center justify-center w-10 h-10 rounded-full bg-blue-100">
                      <Users className="h-5 w-5 text-blue-600" />
                    </div>
                  ) : (
                    <div className="flex items-center justify-center w-10 h-10 rounded-full bg-purple-100">
                      <Bot className="h-5 w-5 text-purple-600" />
                    </div>
                  )}
                  <div>
                    <h3 className="font-semibold text-gray-900">{agent?.name}</h3>
                    <p className="text-sm text-gray-500 capitalize">{agent?.type} Agent</p>
                  </div>
                </div>
                
                <Badge 
                  variant="outline"
                  className={getStatusColor(agent?.is_available ? 'available' : 'unavailable')}
                >
                  {agent?.is_available ? 'Available' : 'Unavailable'}
                </Badge>
              </div>

              {/* Description */}
              {agent?.description && (
                <p className="text-sm text-gray-600 mb-4">{agent?.description}</p>
              )}

              {/* Capabilities */}
              <div className="mb-4">
                <div className="text-xs text-gray-500 mb-2">Capabilities</div>
                <div className="flex flex-wrap gap-1">
                  {agent?.capabilities?.slice(0, 3)?.map((capability) => (
                    <span
                      key={capability}
                      className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-700"
                    >
                      {capability?.replace('_', ' ')}
                    </span>
                  ))}
                  {agent?.capabilities?.length > 3 && (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      +{agent?.capabilities?.length - 3}
                    </span>
                  )}
                </div>
              </div>

              {/* Workload */}
              <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <div className="text-gray-500">Current Load</div>
                    <div className="font-semibold text-gray-900">
                      {currentWorkload}/{agent?.max_concurrent_tasks}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500">Completed</div>
                    <div className="font-semibold text-green-600">{completedTasks}</div>
                  </div>
                </div>
              </div>

              {/* Agent Specific Info */}
              <div className="mb-4 space-y-2 text-xs text-gray-500">
                {agent?.type === AgentType.HUMAN && agent?.contact_method && (
                  <div className="flex items-center gap-2">
                    <MessageCircle className="h-3 w-3" />
                    <span>Contact via {agent?.contact_method}</span>
                  </div>
                )}
                
                {agent?.type === AgentType.AGENTIC && agent?.model_name && (
                  <div className="flex items-center justify-between">
                    <span>Model:</span>
                    <span className="font-medium text-gray-700">{agent?.model_name}</span>
                  </div>
                )}
                
                {agent?.last_active && (
                  <div className="flex items-center gap-2">
                    <Clock className="h-3 w-3" />
                    <span>Last active {formatRelativeTime(agent?.last_active)}</span>
                  </div>
                )}
              </div>

              {/* Actions */}
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleToggleAvailability(agent?.id)}
                  disabled={isLoading}
                  className="flex-1"
                >
                  {agent?.is_available ? (
                    <>
                      <Pause className="h-3 w-3 mr-1" />
                      Disable
                    </>
                  ) : (
                    <>
                      <Play className="h-3 w-3 mr-1" />
                      Enable
                    </>
                  )}
                </Button>
                
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setSelectedAgent(agent)}
                  className="flex items-center gap-1"
                >
                  <Edit className="h-3 w-3" />
                  Edit
                </Button>
                
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleDeleteAgent(agent?.id)}
                  disabled={isLoading || currentWorkload > 0}
                  className="text-red-600 hover:text-red-700"
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Empty State */}
      {agents?.length === 0 && (
        <div className="flex items-center justify-center h-64 bg-white rounded-xl border border-gray-200">
          <div className="text-center">
            <Users className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No agents configured</h3>
            <p className="text-gray-500 mb-4">Get started by adding your first agent</p>
            <Button onClick={() => console.log('Add first agent')}>
              <Plus className="h-4 w-4 mr-2" />
              Add Agent
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
