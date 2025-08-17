
'use client';

import { Server, CheckCircle, XCircle, Clock } from 'lucide-react';
import { MCPServerStatus } from '@/lib/types';
import { formatRelativeTime, getStatusColor } from '@/lib/dashboard-utils';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

interface MCPServerStatusProps {
  servers: MCPServerStatus[];
}

export default function MCPServerStatus({ servers }: MCPServerStatusProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
        return CheckCircle;
      case 'disconnected':
      case 'error':
        return XCircle;
      default:
        return Server;
    }
  };

  const handleTestConnection = async (serverName: string) => {
    // In real implementation, this would call the API
    console.log(`Testing connection for ${serverName}`);
  };

  const handleRestart = async (serverName: string) => {
    // In real implementation, this would call the API
    console.log(`Restarting ${serverName}`);
  };

  if (!servers?.length) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500">
        <div className="text-center">
          <Server className="h-12 w-12 mx-auto mb-3 text-gray-300" />
          <p>No MCP servers configured</p>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {servers?.map((server) => {
        const StatusIcon = getStatusIcon(server?.status);
        
        return (
          <div 
            key={server?.name}
            className="p-4 bg-gradient-to-br from-white to-gray-50 border border-gray-200 rounded-lg hover:shadow-md transition-shadow"
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <StatusIcon 
                  className={`h-5 w-5 ${
                    server?.status === 'connected' 
                      ? 'text-green-600' 
                      : 'text-red-600'
                  }`} 
                />
                <h4 className="font-medium text-gray-900">{server?.name}</h4>
              </div>
              <Badge 
                variant="outline"
                className={getStatusColor(server?.status)}
              >
                {server?.status?.toUpperCase()}
              </Badge>
            </div>

            {/* Connection Details */}
            <div className="space-y-2 mb-4">
              {server?.last_connected && (
                <div className="flex items-center gap-1 text-xs text-gray-500">
                  <Clock className="h-3 w-3" />
                  <span>Connected {formatRelativeTime(server?.last_connected)}</span>
                </div>
              )}
              
              {server?.response_time && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Response Time:</span>
                  <span className={`font-medium ${
                    server?.response_time > 300 ? 'text-red-600' : 
                    server?.response_time > 150 ? 'text-yellow-600' : 'text-green-600'
                  }`}>
                    {server?.response_time}ms
                  </span>
                </div>
              )}
              
              {server?.error_message && (
                <div className="p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                  {server?.error_message}
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleTestConnection(server?.name)}
                className="flex-1 text-xs"
              >
                Test
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => handleRestart(server?.name)}
                className="flex-1 text-xs"
                disabled={server?.status === 'connected'}
              >
                Restart
              </Button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
