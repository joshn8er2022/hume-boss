
'use client';

import React from 'react';
import { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DashboardCardProps {
  title: string;
  value?: string | number;
  description?: string;
  icon?: LucideIcon;
  color?: string;
  trend?: {
    value: number;
    label: string;
    positive?: boolean;
  };
  children?: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

export default function DashboardCard({
  title,
  value,
  description,
  icon: Icon,
  color = 'blue',
  trend,
  children,
  className,
  onClick
}: DashboardCardProps) {
  const colorClasses = {
    blue: 'border-blue-200 bg-gradient-to-br from-blue-50 to-blue-100',
    green: 'border-green-200 bg-gradient-to-br from-green-50 to-green-100',
    yellow: 'border-yellow-200 bg-gradient-to-br from-yellow-50 to-yellow-100',
    red: 'border-red-200 bg-gradient-to-br from-red-50 to-red-100',
    purple: 'border-purple-200 bg-gradient-to-br from-purple-50 to-purple-100',
    gray: 'border-gray-200 bg-gradient-to-br from-gray-50 to-gray-100'
  };

  const iconColorClasses = {
    blue: 'text-blue-600 bg-blue-100',
    green: 'text-green-600 bg-green-100',
    yellow: 'text-yellow-600 bg-yellow-100',
    red: 'text-red-600 bg-red-100',
    purple: 'text-purple-600 bg-purple-100',
    gray: 'text-gray-600 bg-gray-100'
  };

  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-xl border p-6 shadow-sm transition-all duration-200 hover:shadow-lg',
        colorClasses[color as keyof typeof colorClasses] || colorClasses.blue,
        onClick ? 'cursor-pointer hover:scale-105' : '',
        className
      )}
      onClick={onClick}
    >
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute -right-4 -top-4 h-32 w-32 rounded-full bg-white"></div>
        <div className="absolute -bottom-4 -left-4 h-24 w-24 rounded-full bg-white"></div>
      </div>

      <div className="relative">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-600">{title}</p>
            {value !== undefined && (
              <p className="mt-2 text-3xl font-bold text-gray-900">
                {typeof value === 'number' ? value.toLocaleString() : value}
              </p>
            )}
            {description && (
              <p className="mt-1 text-sm text-gray-500">{description}</p>
            )}
          </div>
          
          {Icon && (
            <div className={cn(
              'flex h-12 w-12 items-center justify-center rounded-lg',
              iconColorClasses[color as keyof typeof iconColorClasses] || iconColorClasses.blue
            )}>
              <Icon className="h-6 w-6" />
            </div>
          )}
        </div>

        {/* Trend */}
        {trend && (
          <div className="mt-4 flex items-center gap-2">
            <span className={cn(
              'inline-flex items-center rounded-full px-2 py-1 text-xs font-medium',
              trend?.positive 
                ? 'bg-green-100 text-green-800' 
                : 'bg-red-100 text-red-800'
            )}>
              {trend?.positive ? '↗' : '↘'} {Math.abs(trend?.value)}%
            </span>
            <span className="text-xs text-gray-500">{trend?.label}</span>
          </div>
        )}

        {/* Custom Content */}
        {children && (
          <div className="mt-4">
            {children}
          </div>
        )}
      </div>
    </div>
  );
}
