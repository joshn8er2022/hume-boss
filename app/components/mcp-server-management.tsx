
'use client';

import { useState, useEffect } from 'react';
import { 
  Server, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Wifi, 
  WifiOff,
  RefreshCw,
  Settings,
  Zap,
  AlertTriangle
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import DashboardCard from '@/components/dashboard-card';
import { MCPServerStatus } from '@/lib/types';
import { mockMCPServers } from '@/lib/mock-data';
import { formatRelativeTime, getStatusColor } from '@/lib/dashboard-utils';
import { useToast } from '@/hooks/use-toast';

const MCP_SERVER_DESCRIPTIONS = {
  'Close CRM': 'Customer relationship management and sales pipeline integration',
  'Slack': 'Team communication and notification system',
  'LinkedIn': 'Professional networking and content management',
  'Deep Research': 'Advanced web research and analysis capabilities',
  'Web Crawl': 'Web scraping and data extraction services'
};

const MCP_SERVER_CAPABILITIES = {
  'Close CRM': ['Customer Management', 'Deal Tracking', 'Communication History'],
  'Slack': ['Messaging', 'Channel Management', 'Notifications'],
  'LinkedIn': ['Content Posting', 'Network Management', 'Profile Updates'],
  'Deep Research': ['Web Analysis', 'Content Extraction', 'Research Reports'],
  'Web Crawl': ['Data Scraping', 'URL Processing', 'Content Mining']
};

export default function MCPServerManagement() {
  const { toast } = useToast();
  const [servers, setServers] = useState(mockMCPServers);
  const [isLoading, setIsLoading] = useState(false);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setServers(prev => prev?.map(server => ({
        ...server,
        response_time: server?.status === 'connected' 
          ? Math.max(50, server?.response_time || 100 + (Math.random() - 0.5) * 50)
          : undefined
      })));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const connectedServers = servers?.filter(s => s?.status === 'connected')?.length || 0;
  const disconnectedServers = servers?.filter(s => s?.status === 'disconnected')?.length || 0;
  const errorServers = servers?.filter(s => s?.status === 'error')?.length || 0;
  const avgResponseTime = servers
    ?.filter(s => s?.response_time)
    ?.reduce((acc, s) => acc + (s?.response_time || 0), 0) / Math.max(1, connectedServers);

  const handleTestConnection = async (serverName: string) => {
    setIsLoading(true);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setServers(prev => prev?.map(server =>
        server?.name === serverName
          ? {
              ...server,
              status: Math.random() > 0.3 ? 'connected' : 'error',
              last_connected: Math.random() > 0.3 ? new Date().toISOString() : server?.last_connected,
              response_time: Math.random() > 0.3 ? 100 + Math.random() * 200 : undefined,
              error_message: Math.random() > 0.3 ? undefined : 'Connection timeout'
            }
          : server
      ));

      toast({
        title: "Connection Test Complete",
        description: `Connection test for ${serverName} completed`,
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Test Failed",
        description: "Failed to test connection",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleRestartServer = async (serverName: string) => {
    setIsLoading(true);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      setServers(prev => prev?.map(server =>
        server?.name === serverName
          ? {
              ...server,
              status: 'connected',
              last_connected: new Date().toISOString(),
              response_time: 80 + Math.random() * 40,
              error_message: undefined
            }
          : server
      ));

      toast({
        title: "Server Restarted",
        description: `${serverName} has been successfully restarted`,
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Restart Failed",
        description: "Failed to restart server",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return CheckCircle;
      case 'disconnected': return WifiOff;
      case 'error': return XCircle;
      default: return Server;
    }
  };

  const getResponseTimeColor = (responseTime?: number) => {
    if (!responseTime) return 'text-gray-500';
    if (responseTime < 150) return 'text-green-600';
    if (responseTime < 300) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-6">
      {/* Stats Row */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        <DashboardCard
          title="Connected"
          value={connectedServers}
          description={`${servers?.length || 0} total servers`}
          icon={CheckCircle}
          color="green"
        />
        
        <DashboardCard
          title="Disconnected"
          value={disconnectedServers}
          description="Servers offline"
          icon={WifiOff}
          color="gray"
        />
        
        <DashboardCard
          title="Errors"
          value={errorServers}
          description="Connection issues"
          icon={AlertTriangle}
          color="red"
        />
        
        <DashboardCard
          title="Avg Response"
          value={`${Math.round(avgResponseTime || 0)}ms`}
          description="Average response time"
          icon={Zap}
          color="blue"
        />
      </div>

      {/* Server Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {servers?.map((server) => {
          const StatusIcon = getStatusIcon(server?.status);
          const capabilities = MCP_SERVER_CAPABILITIES[server?.name as keyof typeof MCP_SERVER_CAPABILITIES] || [];
          const description = MCP_SERVER_DESCRIPTIONS[server?.name as keyof typeof MCP_SERVER_DESCRIPTIONS] || 'MCP Server';
          
          return (
            <div 
              key={server?.name}
              className="p-6 bg-white border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-shadow"
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className={`flex items-center justify-center w-12 h-12 rounded-lg ${
                    server?.status === 'connected' 
                      ? 'bg-green-100' 
                      : server?.status === 'error'
                      ? 'bg-red-100'
                      : 'bg-gray-100'
                  }`}>
                    <StatusIcon className={`h-6 w-6 ${
                      server?.status === 'connected' 
                        ? 'text-green-600' 
                        : server?.status === 'error'
                        ? 'text-red-600'
                        : 'text-gray-600'
                    }`} />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">{server?.name}</h3>
                    <p className="text-sm text-gray-500">{description}</p>
                  </div>
                </div>
                
                <Badge 
                  variant="outline"
                  className={getStatusColor(server?.status)}
                >
                  {server?.status?.toUpperCase()}
                </Badge>
              </div>

              {/* Connection Details */}
              <div className="space-y-3 mb-4">
                {server?.last_connected && (
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-500 flex items-center gap-2">
                      <Clock className="h-4 w-4" />
                      Last Connected
                    </span>
                    <span className="text-gray-700">
                      {formatRelativeTime(server?.last_connected)}
                    </span>
                  </div>
                )}
                
                {server?.response_time && (
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-500">Response Time</span>
                    <span className={`font-medium ${getResponseTimeColor(server?.response_time)}`}>
                      {server?.response_time?.toFixed(0)}ms
                    </span>
                  </div>
                )}
                
                {server?.error_message && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-start gap-2">
                      <AlertTriangle className="h-4 w-4 text-red-600 mt-0.5" />
                      <div>
                        <div className="text-sm font-medium text-red-800">Connection Error</div>
                        <div className="text-xs text-red-600 mt-1">{server?.error_message}</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Capabilities */}
              <div className="mb-4">
                <div className="text-sm text-gray-500 mb-2">Capabilities</div>
                <div className="flex flex-wrap gap-1">
                  {capabilities?.map((capability) => (
                    <span
                      key={capability}
                      className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-700"
                    >
                      {capability}
                    </span>
                  ))}
                </div>
              </div>

              {/* Connection Health */}
              <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Connection Health</span>
                  <div className="flex items-center gap-2">
                    {server?.status === 'connected' ? (
                      <Wifi className="h-4 w-4 text-green-600" />
                    ) : (
                      <WifiOff className="h-4 w-4 text-red-600" />
                    )}
                  </div>
                </div>
                
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${
                      server?.status === 'connected' 
                        ? 'bg-green-500' 
                        : server?.status === 'error'
                        ? 'bg-red-500'
                        : 'bg-gray-400'
                    }`}
                    style={{ 
                      width: server?.status === 'connected' 
                        ? '100%' 
                        : server?.status === 'error'
                        ? '25%'
                        : '50%'
                    }}
                  />
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleTestConnection(server?.name)}
                  disabled={isLoading}
                  className="flex-1"
                >
                  <Zap className="h-3 w-3 mr-1" />
                  Test Connection
                </Button>
                
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleRestartServer(server?.name)}
                  disabled={isLoading || server?.status === 'connected'}
                  className="flex-1"
                >
                  <RefreshCw className="h-3 w-3 mr-1" />
                  Restart
                </Button>
                
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => console.log(`Configure ${server?.name}`)}
                  className="flex items-center gap-1"
                >
                  <Settings className="h-3 w-3" />
                  Config
                </Button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Server Status Summary */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Connection Summary</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="text-2xl font-bold text-green-600">{connectedServers}</div>
            <div className="text-sm text-gray-600">Healthy Connections</div>
          </div>
          
          <div className="text-center p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="text-2xl font-bold text-yellow-600">{Math.round(avgResponseTime || 0)}ms</div>
            <div className="text-sm text-gray-600">Average Response Time</div>
          </div>
          
          <div className="text-center p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="text-2xl font-bold text-red-600">{disconnectedServers + errorServers}</div>
            <div className="text-sm text-gray-600">Issues Detected</div>
          </div>
        </div>
      </div>
    </div>
  );
}
