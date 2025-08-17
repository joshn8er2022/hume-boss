'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Power, 
  Activity, 
  Pause, 
  AlertTriangle, 
  Settings,
  Brain,
  Zap
} from 'lucide-react';

export type SystemState = 
  | 'WAKE' 
  | 'READY' 
  | 'ACTIVE' 
  | 'SCALING' 
  | 'MAINTENANCE' 
  | 'ERROR' 
  | 'IDLE'
  | 'PAUSED';

interface DynamicStatusIndicatorProps {
  state: SystemState;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  showLabel?: boolean;
  className?: string;
}

const stateConfig = {
  WAKE: {
    color: '#3B82F6', // Blue
    bgColor: 'bg-blue-500',
    textColor: 'text-blue-600',
    borderColor: 'border-blue-500',
    icon: Power,
    label: 'Waking Up',
    description: 'System is initializing and waiting for instructions',
    animation: 'blink',
    glowColor: 'shadow-blue-500/50'
  },
  READY: {
    color: '#10B981', // Green
    bgColor: 'bg-green-500',
    textColor: 'text-green-600',
    borderColor: 'border-green-500',
    icon: Activity,
    label: 'Ready',
    description: 'System is active and ready to process tasks',
    animation: 'blink',
    glowColor: 'shadow-green-500/50'
  },
  ACTIVE: {
    color: '#10B981', // Green
    bgColor: 'bg-green-500',
    textColor: 'text-green-600',
    borderColor: 'border-green-500',
    icon: Zap,
    label: 'Active',
    description: 'System is actively processing tasks',
    animation: 'pulse',
    glowColor: 'shadow-green-500/50'
  },
  IDLE: {
    color: '#10B981', // Static Green
    bgColor: 'bg-green-500',
    textColor: 'text-green-600',
    borderColor: 'border-green-500',
    icon: Pause,
    label: 'Idle',
    description: 'System is paused but ready',
    animation: 'static',
    glowColor: 'shadow-green-500/30'
  },
  PAUSED: {
    color: '#F59E0B', // Amber
    bgColor: 'bg-amber-500',
    textColor: 'text-amber-600',
    borderColor: 'border-amber-500',
    icon: Pause,
    label: 'Paused',
    description: 'System is temporarily paused',
    animation: 'static',
    glowColor: 'shadow-amber-500/30'
  },
  SCALING: {
    color: '#8B5CF6', // Purple
    bgColor: 'bg-purple-500',
    textColor: 'text-purple-600',
    borderColor: 'border-purple-500',
    icon: Brain,
    label: 'Scaling',
    description: 'System is adjusting capacity',
    animation: 'pulse',
    glowColor: 'shadow-purple-500/50'
  },
  MAINTENANCE: {
    color: '#F59E0B', // Amber
    bgColor: 'bg-amber-500',
    textColor: 'text-amber-600',
    borderColor: 'border-amber-500',
    icon: Settings,
    label: 'Maintenance',
    description: 'System is performing maintenance tasks',
    animation: 'pulse',
    glowColor: 'shadow-amber-500/50'
  },
  ERROR: {
    color: '#EF4444', // Red
    bgColor: 'bg-red-500',
    textColor: 'text-red-600',
    borderColor: 'border-red-500',
    icon: AlertTriangle,
    label: 'Error',
    description: 'System has encountered an error',
    animation: 'blink',
    glowColor: 'shadow-red-500/50'
  }
};

const sizeConfig = {
  sm: {
    container: 'w-8 h-8',
    icon: 'w-4 h-4',
    text: 'text-xs',
    glow: 'shadow-sm'
  },
  md: {
    container: 'w-12 h-12',
    icon: 'w-6 h-6',
    text: 'text-sm',
    glow: 'shadow-md'
  },
  lg: {
    container: 'w-16 h-16',
    icon: 'w-8 h-8',
    text: 'text-base',
    glow: 'shadow-lg'
  },
  xl: {
    container: 'w-20 h-20',
    icon: 'w-10 h-10',
    text: 'text-lg',
    glow: 'shadow-xl'
  }
};

export default function DynamicStatusIndicator({ 
  state, 
  size = 'md', 
  showLabel = true,
  className = '' 
}: DynamicStatusIndicatorProps) {
  const config = stateConfig[state] || stateConfig['IDLE'];
  const sizeStyles = sizeConfig[size];
  const IconComponent = config.icon;

  const getAnimationVariants = () => {
    switch (config.animation) {
      case 'blink':
        return {
          animate: {
            opacity: [1, 0.3, 1],
            scale: [1, 0.95, 1],
          },
          transition: {
            duration: 1.5,
            repeat: Infinity,
            ease: "easeInOut"
          }
        };
      case 'pulse':
        return {
          animate: {
            scale: [1, 1.1, 1],
            opacity: [0.8, 1, 0.8],
          },
          transition: {
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }
        };
      case 'static':
      default:
        return {
          animate: {},
          transition: {}
        };
    }
  };

  const getGlowAnimation = () => {
    if (config.animation === 'static') {
      return {};
    }
    
    return {
      animate: {
        boxShadow: [
          `0 0 20px ${config.color}40`,
          `0 0 30px ${config.color}60`,
          `0 0 20px ${config.color}40`
        ]
      },
      transition: {
        duration: config.animation === 'blink' ? 1.5 : 2,
        repeat: Infinity,
        ease: "easeInOut"
      }
    };
  };

  return (
    <div className={`flex items-center space-x-3 ${className}`}>
      {/* Status Indicator Circle */}
      <motion.div
        className={`
          relative flex items-center justify-center rounded-full
          ${sizeStyles.container} ${config.bgColor} ${config.borderColor}
          border-2 ${sizeStyles.glow} ${config.glowColor}
        `}
        {...getAnimationVariants()}
        {...getGlowAnimation()}
      >
        {/* Pulsing Ring for Active States */}
        <AnimatePresence>
          {(config.animation === 'blink' || config.animation === 'pulse') && (
            <motion.div
              className={`
                absolute inset-0 rounded-full border-2 ${config.borderColor}
                opacity-30
              `}
              initial={{ scale: 1, opacity: 0.3 }}
              animate={{ 
                scale: [1, 1.5, 1], 
                opacity: [0.3, 0, 0.3] 
              }}
              transition={{
                duration: config.animation === 'blink' ? 1.5 : 2,
                repeat: Infinity,
                ease: "easeOut"
              }}
            />
          )}
        </AnimatePresence>

        {/* Icon */}
        <IconComponent 
          className={`${sizeStyles.icon} text-white`}
        />
      </motion.div>

      {/* Label and Description */}
      {showLabel && (
        <div className="flex flex-col">
          <span className={`font-semibold ${config.textColor} ${sizeStyles.text}`}>
            {config.label}
          </span>
          {size !== 'sm' && (
            <span className="text-xs text-gray-500 max-w-xs">
              {config.description}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// Additional component for status badge
export function StatusBadge({ 
  state, 
  className = '' 
}: { 
  state: SystemState; 
  className?: string; 
}) {
  const config = stateConfig[state];
  
  return (
    <motion.div
      className={`
        inline-flex items-center px-2 py-1 rounded-full text-xs font-medium
        ${config.bgColor} text-white ${className}
      `}
      {...(config.animation !== 'static' ? {
        animate: { opacity: [1, 0.7, 1] },
        transition: { duration: 2, repeat: Infinity }
      } : {})}
    >
      <config.icon className="w-3 h-3 mr-1" />
      {config.label}
    </motion.div>
  );
}

// Hook for real-time status updates
export function useSystemStatus() {
  const [status, setStatus] = useState<SystemState>('WAKE');
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  useEffect(() => {
    // WebSocket connection for real-time updates
    const ws = new WebSocket(`ws://localhost:8000/ws`);
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'system_update' && data.data.boss_state) {
          setStatus(data.data.boss_state as SystemState);
          setLastUpdate(new Date());
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    // Fallback: Poll API if WebSocket fails
    const fallbackInterval = setInterval(async () => {
      try {
        const response = await fetch('/api/system/boss-state');
        if (response.ok) {
          const data = await response.json();
          setStatus(data.state as SystemState);
          setLastUpdate(new Date());
        }
      } catch (error) {
        console.error('Error fetching system status:', error);
      }
    }, 5000);

    return () => {
      ws.close();
      clearInterval(fallbackInterval);
    };
  }, []);

  return { status, lastUpdate };
}