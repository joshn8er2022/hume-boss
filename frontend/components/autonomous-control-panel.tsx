
"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { 
  Play, 
  Square, 
  Brain, 
  Settings, 
  Activity, 
  Clock, 
  Zap,
  CheckCircle,
  AlertCircle,
  Users,
  TrendingUp
} from "lucide-react";

interface AutonomousStatus {
  is_autonomous: boolean;
  iteration_count: number;
  current_phase: string | null;
  current_iteration_id: number | null;
  system_state: {
    autonomous_mode: boolean;
    timestamp: string;
  };
  recent_decisions: Array<{
    iteration_id: number;
    decision: string;
    reasoning: string;
    timestamp: string;
  }>;
  agent_hierarchy: {
    boss_agent: {
      agent_id: number;
      agent_name: string;
      status: string;
      role: string;
      completed_tasks: number;
      capabilities: string[];
    };
    subordinate_agents: Array<any>;
    total_agents: number;
  };
  llm_providers: {
    active_provider: string | null;
    providers: Record<string, {
      is_active: boolean;
      has_api_key: boolean;
      is_initialized: boolean;
    }>;
  };
}

const PHASE_INFO = {
  pre_processing: { name: "Pre-Processing", icon: "⚡", color: "blue" },
  boss_decision: { name: "Boss Decision", icon: "🧠", color: "purple" },
  execution: { name: "Execution", icon: "⚙️", color: "green" },
  error_handling: { name: "Error Handling", icon: "🚨", color: "red" },
  next_iteration_prep: { name: "Next Iteration Prep", icon: "🔄", color: "orange" }
};

export default function AutonomousControlPanel() {
  const [status, setStatus] = useState<AutonomousStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus();
    
    // Auto-refresh status when autonomous
    const interval = setInterval(() => {
      if (status?.is_autonomous) {
        fetchStatus();
      }
    }, 2000); // Update every 2 seconds when autonomous
    
    return () => clearInterval(interval);
  }, [status?.is_autonomous]);

  const fetchStatus = async () => {
    try {
      const response = await fetch('/api/autonomous/status');
      const data = await response.json();
      
      if (data.success) {
        setStatus(data.data);
      }
    } catch (error) {
      console.error('Error fetching autonomous status:', error);
      toast.error('Failed to fetch autonomous status');
    } finally {
      setLoading(false);
    }
  };

  const startAutonomous = async () => {
    setActionLoading('start');
    try {
      const response = await fetch('/api/autonomous/start', {
        method: 'POST'
      });

      const data = await response.json();

      if (data.success) {
        setStatus(data.status);
        toast.success('🚀 Autonomous mode started! DSPY signatures are now in control.');
      } else {
        toast.error(data.message || 'Failed to start autonomous mode');
      }
    } catch (error) {
      console.error('Error starting autonomous mode:', error);
      toast.error('Failed to start autonomous mode');
    } finally {
      setActionLoading(null);
    }
  };

  const stopAutonomous = async () => {
    setActionLoading('stop');
    try {
      const response = await fetch('/api/autonomous/stop', {
        method: 'POST'
      });

      const data = await response.json();

      if (data.success) {
        setStatus(data.status);
        toast.success('⏹️ Autonomous mode stopped.');
      } else {
        toast.error(data.message || 'Failed to stop autonomous mode');
      }
    } catch (error) {
      console.error('Error stopping autonomous mode:', error);
      toast.error('Failed to stop autonomous mode');
    } finally {
      setActionLoading(null);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Autonomous Control Panel
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

  if (!status) return null;

  const hasActiveProvider = status.llm_providers.active_provider !== null;
  const canStart = !status.is_autonomous && hasActiveProvider;

  return (
    <div className="space-y-6">
      {/* Main Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Autonomous DSPY Engine
          </CardTitle>
          <CardDescription>
            Fully autonomous decision-making powered by DSPY signatures. The system makes all decisions without manual intervention.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Status Overview */}
          <div className="flex items-center justify-between p-4 border rounded-lg">
            <div className="flex items-center gap-4">
              <div className={`w-3 h-3 rounded-full ${status.is_autonomous ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`} />
              <div>
                <p className="font-semibold text-lg">
                  {status.is_autonomous ? 'AUTONOMOUS MODE ACTIVE' : 'Manual Mode'}
                </p>
                <p className="text-sm text-muted-foreground">
                  {status.is_autonomous 
                    ? 'DSPY signatures are making decisions autonomously' 
                    : 'System awaiting manual control'}
                </p>
              </div>
            </div>
            
            <div className="flex gap-2">
              {status.is_autonomous ? (
                <Button
                  onClick={stopAutonomous}
                  disabled={actionLoading === 'stop'}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  <Square className="h-4 w-4" />
                  {actionLoading === 'stop' ? 'Stopping...' : 'Stop'}
                </Button>
              ) : (
                <Button
                  onClick={startAutonomous}
                  disabled={!canStart || actionLoading === 'start'}
                  className="flex items-center gap-2"
                >
                  <Play className="h-4 w-4" />
                  {actionLoading === 'start' ? 'Starting...' : 'Start Autonomous'}
                </Button>
              )}
            </div>
          </div>

          {/* Prerequisites Check */}
          {!hasActiveProvider && (
            <Alert className="border-yellow-200 bg-yellow-50">
              <AlertCircle className="h-4 w-4 text-yellow-600" />
              <AlertDescription className="text-yellow-800">
                <strong>LLM Provider Required:</strong> Configure at least one LLM provider with an API key to enable autonomous mode.
              </AlertDescription>
            </Alert>
          )}

          {/* Current Status Details */}
          {status.is_autonomous && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label className="text-sm font-medium text-muted-foreground">Iterations</Label>
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-blue-500" />
                  <span className="text-lg font-semibold">{status.iteration_count}</span>
                </div>
              </div>
              
              <div className="space-y-2">
                <Label className="text-sm font-medium text-muted-foreground">Current Phase</Label>
                <div className="flex items-center gap-2">
                  <span className="text-lg">
                    {status.current_phase ? PHASE_INFO[status.current_phase as keyof typeof PHASE_INFO]?.icon : '⏸️'}
                  </span>
                  <span className="font-semibold">
                    {status.current_phase 
                      ? PHASE_INFO[status.current_phase as keyof typeof PHASE_INFO]?.name 
                      : 'Idle'}
                  </span>
                </div>
              </div>
              
              <div className="space-y-2">
                <Label className="text-sm font-medium text-muted-foreground">Active LLM</Label>
                <div className="flex items-center gap-2">
                  <Zap className="h-4 w-4 text-green-500" />
                  <span className="font-semibold capitalize">
                    {status.llm_providers.active_provider || 'None'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Agent Hierarchy Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Agent Hierarchy
          </CardTitle>
          <CardDescription>
            Boss (Agent 0) and subordinate agents managed by the autonomous system
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Boss Agent */}
            <div className="flex items-center justify-between p-3 border rounded-lg bg-blue-50/50 border-blue-200">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-semibold text-sm">
                  0
                </div>
                <div>
                  <p className="font-semibold text-blue-800">Boss Agent (Agent 0)</p>
                  <p className="text-sm text-blue-600">Strategic Decision Making</p>
                </div>
              </div>
              <Badge variant="outline" className="text-blue-600 border-blue-200">
                {status.agent_hierarchy.boss_agent.status}
              </Badge>
            </div>

            {/* Subordinate Agents */}
            {status.agent_hierarchy.subordinate_agents.map((agent, index) => (
              <div key={agent.agent_id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-gray-600 text-white flex items-center justify-center font-semibold text-sm">
                    {agent.agent_id}
                  </div>
                  <div>
                    <p className="font-semibold">Agent {agent.agent_id}</p>
                    <p className="text-sm text-muted-foreground">{agent.agent_name}</p>
                  </div>
                </div>
                <Badge variant={agent.status === 'busy' ? 'default' : 'outline'}>
                  {agent.status}
                </Badge>
              </div>
            ))}

            {status.agent_hierarchy.subordinate_agents.length === 0 && (
              <div className="text-center py-6 text-muted-foreground">
                <p>No subordinate agents created yet</p>
                <p className="text-sm mt-1">The Boss will create agents as needed during autonomous operation</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Recent Decisions */}
      {status.recent_decisions && status.recent_decisions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Recent Autonomous Decisions
            </CardTitle>
            <CardDescription>
              Latest decisions made by the DSPY-driven autonomous system
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {status.recent_decisions.slice(0, 5).map((decision, index) => (
                <div key={decision.iteration_id || index} className="p-3 border rounded-lg space-y-2">
                  <div className="flex items-center justify-between">
                    <Badge variant="outline" className="text-xs">
                      Iteration {decision.iteration_id}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {new Date(decision.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="font-medium text-sm">{decision.decision}</p>
                  <p className="text-xs text-muted-foreground">{decision.reasoning}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
