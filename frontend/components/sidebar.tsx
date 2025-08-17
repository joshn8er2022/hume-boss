
'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import {
  LayoutDashboard,
  Users,
  ListTodo,
  Server,
  BarChart3,
  AlertTriangle,
  Settings,
  FileText,
  Activity,
  Brain,
  Zap
} from 'lucide-react';

const navigation = [
  {
    name: 'Overview',
    href: '/',
    icon: LayoutDashboard,
    description: 'System status and key metrics'
  },
  {
    name: 'Autonomous Engine',
    href: '/autonomous',
    icon: Zap,
    description: 'Autonomous DSPY-driven operation control'
  },
  {
    name: 'Boss State',
    href: '/boss',
    icon: Brain,
    description: 'DSPY Boss state and controls'
  },
  {
    name: 'Agents',
    href: '/agents',
    icon: Users,
    description: 'Agent management and status'
  },
  {
    name: 'Task Queue',
    href: '/tasks',
    icon: ListTodo,
    description: 'Task monitoring and management'
  },
  {
    name: 'MCP Servers',
    href: '/mcp-servers',
    icon: Server,
    description: 'MCP server status and connections'
  },
  {
    name: 'Performance',
    href: '/performance',
    icon: BarChart3,
    description: 'Performance metrics and analytics'
  },
  {
    name: 'Failures',
    href: '/failures',
    icon: AlertTriangle,
    description: 'Error tracking and diagnostics'
  },
  {
    name: 'Logs',
    href: '/logs',
    icon: FileText,
    description: 'System logs and reports'
  },
  {
    name: 'Health Check',
    href: '/health',
    icon: Activity,
    description: 'System health and diagnostics'
  },
  {
    name: 'Configuration',
    href: '/config',
    icon: Settings,
    description: 'System configuration management'
  }
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="flex h-full w-64 flex-col fixed left-0 top-0 border-r border-gray-200 bg-white shadow-lg z-50">
      {/* Header */}
      <div className="flex h-16 items-center gap-2 px-4 border-b border-gray-200 bg-gradient-to-r from-blue-600 to-purple-600">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-white/20 backdrop-blur-sm">
          <Zap className="h-5 w-5 text-white" />
        </div>
        <div className="flex flex-col">
          <span className="text-lg font-bold text-white">DSPY Boss</span>
          <span className="text-xs text-blue-100">Dashboard</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-2 py-4 overflow-y-auto">
        {navigation?.map((item) => {
          const Icon = item?.icon;
          const isActive = pathname === item?.href;
          
          return (
            <Link
              key={item?.name}
              href={item?.href}
              className={cn(
                'group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200 hover:bg-gray-50',
                isActive
                  ? 'bg-blue-50 text-blue-700 border-r-2 border-blue-600'
                  : 'text-gray-700 hover:text-gray-900'
              )}
              title={item?.description}
            >
              <Icon 
                className={cn(
                  'h-5 w-5 transition-colors',
                  isActive ? 'text-blue-600' : 'text-gray-500 group-hover:text-gray-700'
                )} 
              />
              <span className="flex-1">{item?.name}</span>
              {isActive && (
                <div className="w-1.5 h-1.5 rounded-full bg-blue-600" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-gray-200 bg-gray-50">
        <div className="text-xs text-gray-500 text-center">
          DSPY Boss System v1.0
        </div>
      </div>
    </div>
  );
}
