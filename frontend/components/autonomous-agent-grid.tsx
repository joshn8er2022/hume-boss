
"use client";

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { 
  Users, 
  Crown, 
  Bot, 
  Activity, 
  Clock, 
  CheckCircle, 
  Plus,
  Minus,
  BarChart3,
  Zap,
  Target
} from "lucide-react";

interface AgentInfo {
  agent_id: number;
  display_name: string;
  agent_name: string;
  role: "boss" | "subordinate";
  status: string;
  completed_tasks: number;
  capabilities: string[];
  performance_score: number;
  last_active: string;
  current_task?: string | null;
}

interface AgentHierarchy {
  boss_agent: AgentInfo;
  subordinate_agents: AgentInfo[];
  total_agents: number;
  next_agent_id: number;
  hierarchy_stats: {
    active_agents: number;
    idle_agents: number;
    busy_agents: number;
    average_performance: number;
    total_completed_tasks: number;
  };
}

export default function AutonomousAgentGrid() {
  const [hierarchy, setHierarchy] = useState<AgentHierarchy | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [newAgentName, setNewAgentName] = useState('');
  const [targetAgentCount, setTargetAgentCount] = useState<number>(0);

  useEffect(() => {
    fetchHierarchy();
    
    // Auto-refresh every 5 seconds
    const interval = setInterval(fetchHierarchy, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchHierarchy = async () => {
    try {
      const response = await fetch('/api/agents/hierarchy');
      const data = await response.json();
      
      if (data.success) {
        setHierarchy(data.data);
        setTargetAgentCount(data.data.subordinate_agents.length);
      }
    } catch (error) {
      console.error('Error fetching agent hierarchy:', error);
      toast.error('Failed to fetch agent hierarchy');
    } finally {
      setLoading(false);
    }
  };

  const createAgent = async () => {
    if (!newAgentName.trim()) {
      toast.error('Please enter an agent name');
      return;
    }

    setActionLoading('create');
    try {
      const response = await fetch('/api/agents/hierarchy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'create_agent',
          agent_name: newAgentName
        })
      });

      const data = await response.json();

      if (data.success) {
        setHierarchy(data.data);
        setNewAgentName('');
        toast.success(data.message);
      } else {
        toast.error(data.message || 'Failed to create agent');
      }
    } catch (error) {
      console.error('Error creating agent:', error);
      toast.error('Failed to create agent');
    } finally {
      setActionLoading(null);
    }
  };

  const scaleAgents = async () => {
    if (targetAgentCount < 0) {
      toast.error('Agent count must be positive');
      return;
    }

    setActionLoading('scale');
    try {
      const response = await fetch('/api/agents/hierarchy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'scale_agents',
          target_count: targetAgentCount
        })
      });

      const data = await response.json();

      if (data.success) {
        setHierarchy(data.data);
        toast.success(data.message);
      } else {
        toast.error(data.message || 'Failed to scale agents');
      }
    } catch (error) {
      console.error('Error scaling agents:', error);
      toast.error('Failed to scale agents');
    } finally {
      setActionLoading(null);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active': return 'bg-green-100 text-green-800 border-green-200';
      case 'busy': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'idle': return 'bg-gray-100 text-gray-800 border-gray-200';
      case 'error': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getPerformanceColor = (score: number) => {
    if (score >= 0.9) return 'text-green-600';
    if (score >= 0.7) return 'text-blue-600';
    if (score >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Agent Hierarchy
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center p-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!hierarchy) return null;

  return (
    <div className="space-y-6">
      {/* Hierarchy Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Agent Hierarchy
          </CardTitle>
          <CardDescription>
            Boss (Agent 0) and subordinate agents. Numbers are displayed in human-readable format.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Stats Overview */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{hierarchy.total_agents}</div>
              <div className="text-sm text-muted-foreground">Total Agents</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{hierarchy.hierarchy_stats.active_agents}</div>
              <div className="text-sm text-muted-foreground">Active</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{hierarchy.hierarchy_stats.busy_agents}</div>
              <div className="text-sm text-muted-foreground">Busy</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-600">{hierarchy.hierarchy_stats.idle_agents}</div>
              <div className="text-sm text-muted-foreground">Idle</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {Math.round(hierarchy.hierarchy_stats.average_performance * 100)}%
              </div>
              <div className="text-sm text-muted-foreground">Avg Performance</div>
            </div>
          </div>

          <Separator className="my-6" />

          {/* Boss Agent (Agent 0) */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Crown className="h-5 w-5 text-yellow-500" />
              Boss Agent
            </h3>
            
            <Card className="border-yellow-200 bg-yellow-50/30">
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-yellow-500 text-white flex items-center justify-center font-bold">
                      0
                    </div>
                    <div>
                      <h4 className="font-semibold text-yellow-800">Boss (Agent 0)</h4>
                      <p className="text-sm text-yellow-600">{hierarchy.boss_agent.agent_name}</p>
                    </div>
                  </div>
                  <Badge className={`border ${getStatusColor(hierarchy.boss_agent.status)}`}>
                    {hierarchy.boss_agent.status}
                  </Badge>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Completed Tasks</div>
                    <div className="font-semibold">{hierarchy.boss_agent.completed_tasks}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Performance</div>
                    <div className={`font-semibold ${getPerformanceColor(hierarchy.boss_agent.performance_score)}`}>
                      {Math.round(hierarchy.boss_agent.performance_score * 100)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Capabilities</div>
                    <div className="font-semibold">{hierarchy.boss_agent.capabilities.length}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Last Active</div>
                    <div className="font-semibold text-xs">
                      {new Date(hierarchy.boss_agent.last_active).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
                
                {hierarchy.boss_agent.current_task && (
                  <div className="mt-3 p-2 bg-yellow-100 rounded text-sm">
                    <div className="text-yellow-600 font-medium">Current Task:</div>
                    <div className="text-yellow-800">{hierarchy.boss_agent.current_task}</div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Subordinate Agents */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Bot className="h-5 w-5 text-blue-500" />
                Subordinate Agents
              </h3>
              
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-2">
                  <Label htmlFor="agent-count" className="text-sm">Target Count:</Label>
                  <Input
                    id="agent-count"
                    type="number"
                    min="0"
                    max="20"
                    value={targetAgentCount}
                    onChange={(e) => setTargetAgentCount(parseInt(e.target.value) || 0)}
                    className="w-20 h-8"
                  />
                  <Button
                    size="sm"
                    onClick={scaleAgents}
                    disabled={actionLoading === 'scale'}
                    className="h-8"
                  >
                    {actionLoading === 'scale' ? 'Scaling...' : 'Scale'}
                  </Button>
                </div>
              </div>
            </div>

            {hierarchy.subordinate_agents.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {hierarchy.subordinate_agents.map((agent) => (
                  <Card key={agent.agent_id} className="hover:shadow-md transition-shadow">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-full bg-blue-500 text-white flex items-center justify-center font-bold text-sm">
                            {agent.agent_id}
                          </div>
                          <div>
                            <h4 className="font-semibold">{agent.display_name}</h4>
                            <p className="text-xs text-muted-foreground">{agent.agent_name}</p>
                          </div>
                        </div>
                        <Badge variant="outline" className={getStatusColor(agent.status)}>
                          {agent.status}
                        </Badge>
                      </div>
                      
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Tasks Completed:</span>
                          <span className="font-medium">{agent.completed_tasks}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Performance:</span>
                          <span className={`font-medium ${getPerformanceColor(agent.performance_score)}`}>
                            {Math.round(agent.performance_score * 100)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Capabilities:</span>
                          <span className="font-medium">{agent.capabilities.length}</span>
                        </div>
                      </div>
                      
                      <Progress 
                        value={agent.performance_score * 100} 
                        className="mt-3 h-2"
                      />
                      
                      {agent.current_task && (
                        <div className="mt-3 p-2 bg-blue-50 rounded text-xs">
                          <div className="text-blue-600 font-medium">Current Task:</div>
                          <div className="text-blue-800">{agent.current_task}</div>
                        </div>
                      )}
                      
                      <div className="mt-2 text-xs text-muted-foreground">
                        Last active: {new Date(agent.last_active).toLocaleTimeString()}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card className="border-dashed">
                <CardContent className="p-8 text-center">
                  <Bot className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                  <p className="text-muted-foreground mb-4">No subordinate agents yet</p>
                  <p className="text-sm text-muted-foreground mb-4">
                    The Boss will automatically create agents during autonomous operation, or you can create them manually.
                  </p>
                  
                  <div className="flex items-center gap-2 justify-center max-w-md mx-auto">
                    <Input
                      placeholder="Agent name (e.g., Trading Agent)"
                      value={newAgentName}
                      onChange={(e) => setNewAgentName(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && createAgent()}
                    />
                    <Button 
                      onClick={createAgent}
                      disabled={actionLoading === 'create' || !newAgentName.trim()}
                    >
                      <Plus className="h-4 w-4 mr-1" />
                      {actionLoading === 'create' ? 'Creating...' : 'Create'}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
