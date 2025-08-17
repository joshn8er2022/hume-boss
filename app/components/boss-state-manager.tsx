
'use client';

import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  Play, 
  Pause, 
  Square, 
  RotateCcw, 
  Zap,
  Clock,
  Activity,
  CheckCircle,
  AlertCircle
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import DashboardCard from '@/components/dashboard-card';
import { BossState } from '@/lib/types';
import { mockSystemOverview } from '@/lib/mock-data';
import { getStatusColor, formatDuration, formatRelativeTime } from '@/lib/dashboard-utils';
import { useToast } from '@/hooks/use-toast';

const STATE_DESCRIPTIONS = {
  [BossState.IDLE]: 'System is idle and waiting for new tasks or commands',
  [BossState.AWAKE]: 'System is awake and actively monitoring for work',
  [BossState.THINKING]: 'Boss is analyzing the current situation and planning next actions',
  [BossState.RETHINK]: 'Boss is reconsidering the current approach and strategy',
  [BossState.EXECUTING]: 'Boss is actively executing tasks and managing agents',
  [BossState.RESEARCHING]: 'Boss is conducting research to inform decision making',
  [BossState.REFLECTING]: 'Boss is reflecting on recent performance and learning',
  [BossState.RESTART]: 'System is restarting and reinitializing components',
  [BossState.STOP]: 'System is stopped and not processing any tasks'
};

const STATE_ACTIONS = {
  [BossState.IDLE]: [BossState.AWAKE, BossState.STOP, BossState.RESTART],
  [BossState.AWAKE]: [BossState.THINKING, BossState.EXECUTING, BossState.RESEARCHING, BossState.IDLE, BossState.STOP],
  [BossState.THINKING]: [BossState.EXECUTING, BossState.RESEARCHING, BossState.RETHINK, BossState.REFLECTING, BossState.IDLE],
  [BossState.RETHINK]: [BossState.THINKING, BossState.EXECUTING, BossState.REFLECTING, BossState.IDLE],
  [BossState.EXECUTING]: [BossState.THINKING, BossState.REFLECTING, BossState.IDLE, BossState.AWAKE],
  [BossState.RESEARCHING]: [BossState.THINKING, BossState.EXECUTING, BossState.REFLECTING, BossState.IDLE],
  [BossState.REFLECTING]: [BossState.THINKING, BossState.IDLE, BossState.AWAKE],
  [BossState.RESTART]: [BossState.IDLE, BossState.AWAKE],
  [BossState.STOP]: []
};

export default function BossStateManager() {
  const { toast } = useToast();
  const [overview, setOverview] = useState(mockSystemOverview);
  const [isLoading, setIsLoading] = useState(false);
  const [stateHistory, setStateHistory] = useState([
    { state: BossState.IDLE, timestamp: new Date(Date.now() - 3600000).toISOString(), duration: 1800 },
    { state: BossState.AWAKE, timestamp: new Date(Date.now() - 1800000).toISOString(), duration: 1800 },
    { state: BossState.THINKING, timestamp: new Date(Date.now() - 900000).toISOString(), duration: 600 },
    { state: BossState.EXECUTING, timestamp: new Date(Date.now() - 300000).toISOString(), duration: 300 },
  ]);

  const currentState = overview?.boss_state;
  const stateData = overview?.state_data;
  const possibleActions = STATE_ACTIONS[currentState as BossState] || [];

  const handleStateChange = async (newState: BossState) => {
    setIsLoading(true);
    
    try {
      // In real implementation, this would call the API
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setOverview(prev => ({
        ...prev,
        boss_state: newState
      }));

      setStateHistory(prev => [
        ...prev,
        {
          state: newState,
          timestamp: new Date().toISOString(),
          duration: 0
        }
      ]);

      toast({
        title: "State Changed",
        description: `Boss state changed to ${newState.toUpperCase()}`,
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "State Change Failed",
        description: "Failed to change boss state",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getStateIcon = (state: BossState) => {
    switch (state) {
      case BossState.IDLE: return Pause;
      case BossState.AWAKE: return Activity;
      case BossState.THINKING: 
      case BossState.RETHINK: return Brain;
      case BossState.EXECUTING: return Play;
      case BossState.RESEARCHING: return Zap;
      case BossState.REFLECTING: return CheckCircle;
      case BossState.RESTART: return RotateCcw;
      case BossState.STOP: return Square;
      default: return Brain;
    }
  };

  return (
    <div className="space-y-6">
      {/* Current State Overview */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-4 mb-6">
              <div className="flex items-center justify-center w-16 h-16 rounded-full bg-blue-100">
                {React.createElement(getStateIcon(currentState as BossState), {
                  className: "h-8 w-8 text-blue-600"
                })}
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900">{currentState?.toUpperCase()}</h2>
                <p className="text-gray-600 mt-1">
                  {STATE_DESCRIPTIONS[currentState as BossState] || 'Unknown state'}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-500">Current Workload</div>
                <div className="text-2xl font-bold text-gray-900">{stateData?.current_workload}</div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-500">Success Rate</div>
                <div className="text-2xl font-bold text-green-600">{stateData?.success_rate?.toFixed(1)}%</div>
              </div>
            </div>

            {/* State Actions */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900">Available Actions</h3>
              <div className="flex flex-wrap gap-2">
                {possibleActions?.map((action) => {
                  const ActionIcon = getStateIcon(action);
                  return (
                    <Button
                      key={action}
                      onClick={() => handleStateChange(action)}
                      disabled={isLoading}
                      className="flex items-center gap-2"
                      variant={action === BossState.STOP ? 'destructive' : 'default'}
                    >
                      <ActionIcon className="h-4 w-4" />
                      {action?.toUpperCase()}
                    </Button>
                  );
                })}
              </div>
              {possibleActions?.length === 0 && (
                <p className="text-gray-500 text-sm">No actions available from current state</p>
              )}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <DashboardCard
            title="System Uptime"
            value={formatDuration(overview?.uptime || 0)}
            description="Since last restart"
            icon={Clock}
            color="blue"
          />
          
          <DashboardCard
            title="Active Agents"
            value={stateData?.active_agents?.length || 0}
            description="Currently running"
            icon={Activity}
            color="green"
          />
          
          <DashboardCard
            title="Pending Tasks"
            value={stateData?.pending_tasks?.length || 0}
            description="Awaiting execution"
            icon={Clock}
            color="yellow"
          />
        </div>
      </div>

      {/* State History */}
      <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">State History</h3>
        <div className="space-y-3">
          {stateHistory?.slice(-10)?.reverse()?.map((entry, index) => {
            const StateIcon = getStateIcon(entry?.state);
            return (
              <div key={index} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                <StateIcon className="h-5 w-5 text-gray-600" />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Badge 
                      variant="outline"
                      className={getStatusColor(entry?.state)}
                    >
                      {entry?.state?.toUpperCase()}
                    </Badge>
                    <span className="text-sm text-gray-600">
                      {formatRelativeTime(entry?.timestamp)}
                    </span>
                  </div>
                </div>
                <div className="text-sm text-gray-500">
                  {formatDuration(entry?.duration)}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-lg">
            <div className="flex items-center gap-3">
              <CheckCircle className="h-8 w-8 text-green-600" />
              <div>
                <div className="text-sm text-gray-600">Tasks Completed</div>
                <div className="text-2xl font-bold text-green-700">{stateData?.completed_tasks?.length}</div>
              </div>
            </div>
          </div>
          
          <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-lg">
            <div className="flex items-center gap-3">
              <Clock className="h-8 w-8 text-blue-600" />
              <div>
                <div className="text-sm text-gray-600">Avg Task Duration</div>
                <div className="text-2xl font-bold text-blue-700">{stateData?.average_task_duration?.toFixed(1)}s</div>
              </div>
            </div>
          </div>
          
          <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 rounded-lg">
            <div className="flex items-center gap-3">
              <Brain className="h-8 w-8 text-purple-600" />
              <div>
                <div className="text-sm text-gray-600">Total Processed</div>
                <div className="text-2xl font-bold text-purple-700">{stateData?.total_tasks_processed}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Reflections */}
      {stateData?.reflection_notes && (
        <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Reflection</h3>
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-start gap-3">
              <Brain className="h-5 w-5 text-blue-600 mt-1" />
              <div>
                <p className="text-gray-700">{stateData?.reflection_notes}</p>
                {stateData?.last_reflection && (
                  <p className="text-sm text-gray-500 mt-2">
                    {formatRelativeTime(stateData?.last_reflection)}
                  </p>
                )}
              </div>
            </div>
          </div>
          
          {stateData?.improvement_actions?.length > 0 && (
            <div className="mt-4">
              <h4 className="font-medium text-gray-900 mb-2">Improvement Actions</h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
                {stateData?.improvement_actions?.map((action, index) => (
                  <li key={index}>{action}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
