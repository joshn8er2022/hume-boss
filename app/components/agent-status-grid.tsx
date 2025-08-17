
'use client';

import { Users, Bot, MessageCircle, Clock } from 'lucide-react';
import { AgentConfig, AgentType } from '@/lib/types';
import { formatRelativeTime, getStatusColor } from '@/lib/dashboard-utils';
import { Badge } from '@/components/ui/badge';

interface AgentStatusGridProps {
  agents: AgentConfig[];
}

export default function AgentStatusGrid({ agents }: AgentStatusGridProps) {
  if (!agents?.length) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500">
        <div className="text-center">
          <Users className="h-12 w-12 mx-auto mb-3 text-gray-300" />
          <p>No agents configured</p>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {agents?.map((agent) => (
        <div 
          key={agent?.id}
          className="p-4 bg-gradient-to-br from-white to-gray-50 border border-gray-200 rounded-lg hover:shadow-md transition-shadow"
        >
          {/* Header */}
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-2">
              {agent?.type === AgentType.HUMAN ? (
                <Users className="h-5 w-5 text-blue-600" />
              ) : (
                <Bot className="h-5 w-5 text-purple-600" />
              )}
              <h4 className="font-medium text-gray-900">{agent?.name}</h4>
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
            <p className="text-sm text-gray-600 mb-3">{agent?.description}</p>
          )}

          {/* Capabilities */}
          <div className="mb-3">
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
                  +{agent?.capabilities?.length - 3} more
                </span>
              )}
            </div>
          </div>

          {/* Details */}
          <div className="space-y-2 text-xs text-gray-500">
            <div className="flex items-center justify-between">
              <span>Max Tasks:</span>
              <span className="font-medium">{agent?.max_concurrent_tasks}</span>
            </div>
            
            {agent?.type === AgentType.HUMAN && agent?.contact_method && (
              <div className="flex items-center gap-1">
                <MessageCircle className="h-3 w-3" />
                <span>{agent?.contact_method}</span>
              </div>
            )}
            
            {agent?.type === AgentType.AGENTIC && agent?.model_name && (
              <div className="flex items-center justify-between">
                <span>Model:</span>
                <span className="font-medium">{agent?.model_name}</span>
              </div>
            )}
            
            {agent?.last_active && (
              <div className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                <span>Last active {formatRelativeTime(agent?.last_active)}</span>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
