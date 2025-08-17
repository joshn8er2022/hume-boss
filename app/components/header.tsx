
'use client';

import { useState, useEffect } from 'react';
import { Bell, RefreshCw, AlertTriangle, CheckCircle, Activity } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';
import { formatRelativeTime, getStatusColor } from '@/lib/dashboard-utils';
import { mockSystemOverview } from '@/lib/mock-data';

interface HeaderProps {
  title: string;
  description?: string;
}

export default function Header({ title, description }: HeaderProps) {
  const { toast } = useToast();
  const [systemStatus, setSystemStatus] = useState(mockSystemOverview);
  const [notifications, setNotifications] = useState(3);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  const refreshData = async () => {
    try {
      // In real implementation, this would call the API
      setLastUpdate(new Date());
      toast({
        title: "Data Refreshed",
        description: "System data has been updated",
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Refresh Failed",
        description: "Failed to refresh system data",
      });
    }
  };

  const getSystemStatusInfo = () => {
    if (!systemStatus) return { text: 'Unknown', color: 'text-gray-600' };
    
    const { health_status, boss_state } = systemStatus;
    
    if (health_status === 'critical') {
      return { text: 'Critical', color: 'text-red-600', icon: AlertTriangle };
    } else if (health_status === 'warning') {
      return { text: 'Warning', color: 'text-yellow-600', icon: AlertTriangle };
    } else if (boss_state === 'awake') {
      return { text: 'Active', color: 'text-green-600', icon: Activity };
    } else {
      return { text: 'Healthy', color: 'text-green-600', icon: CheckCircle };
    }
  };

  const statusInfo = getSystemStatusInfo();
  const StatusIcon = statusInfo?.icon || CheckCircle;

  return (
    <div className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-gray-900">{title}</h1>
          {description && (
            <p className="mt-1 text-sm text-gray-600">{description}</p>
          )}
        </div>
        
        <div className="flex items-center gap-4">
          {/* System Status */}
          <div className="flex items-center gap-2">
            <StatusIcon className={`h-4 w-4 ${statusInfo?.color}`} />
            <span className={`text-sm font-medium ${statusInfo?.color}`}>
              {statusInfo?.text}
            </span>
          </div>

          {/* Last Update */}
          <div className="text-xs text-gray-500 hidden sm:block">
            Updated {formatRelativeTime(lastUpdate?.toISOString())}
          </div>

          {/* Notifications */}
          <div className="relative">
            <Button 
              variant="ghost" 
              size="sm" 
              className="relative h-9 w-9 p-0"
              onClick={() => setNotifications(0)}
            >
              <Bell className="h-4 w-4" />
              {notifications > 0 && (
                <Badge 
                  variant="destructive" 
                  className="absolute -top-1 -right-1 h-5 w-5 rounded-full p-0 text-xs flex items-center justify-center"
                >
                  {notifications}
                </Badge>
              )}
            </Button>
          </div>

          {/* Refresh Button */}
          <Button 
            variant="outline" 
            size="sm" 
            onClick={refreshData}
            className="flex items-center gap-2"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Refresh</span>
          </Button>
        </div>
      </div>
      
      {/* Boss State Info */}
      <div className="mt-3 flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1.5">
          <span className="text-gray-500">Boss State:</span>
          <Badge 
            variant="outline" 
            className={`${getStatusColor(systemStatus?.boss_state)} px-2 py-0.5`}
          >
            {systemStatus?.boss_state?.toUpperCase()}
          </Badge>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-gray-500">Active Agents:</span>
          <span className="font-medium text-gray-900">
            {systemStatus?.state_data?.active_agents?.length || 0}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-gray-500">Success Rate:</span>
          <span className="font-medium text-green-600">
            {systemStatus?.state_data?.success_rate?.toFixed(1) || '0'}%
          </span>
        </div>
      </div>
    </div>
  );
}
