
'use client';

import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { mockMetricsHistory } from '@/lib/mock-data';

export default function SystemMetricsChart() {
  const [data, setData] = useState(mockMetricsHistory);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setData(prevData => {
        const newData = [...prevData];
        const lastPoint = newData[newData.length - 1];
        
        // Add new point
        newData.push({
          timestamp: new Date().toISOString(),
          tasks_per_minute: Math.max(0, lastPoint.tasks_per_minute + (Math.random() - 0.5)),
          cpu_usage: Math.max(0, Math.min(100, lastPoint.cpu_usage + (Math.random() - 0.5) * 5)),
          memory_usage: Math.max(0, lastPoint.memory_usage + (Math.random() - 0.5) * 20),
          active_agents: Math.max(0, lastPoint.active_agents + Math.floor((Math.random() - 0.5) * 2)),
          success_rate: Math.max(0, Math.min(100, lastPoint.success_rate + (Math.random() - 0.5) * 2))
        });
        
        // Keep only last 24 points
        if (newData.length > 24) {
          newData.shift();
        }
        
        return newData;
      });
    }, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <XAxis 
            dataKey="timestamp" 
            tickFormatter={formatTime}
            tick={{ fontSize: 10 }}
            interval="preserveStartEnd"
          />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip 
            labelFormatter={(value) => formatTime(value as string)}
            formatter={(value: number, name: string) => [
              `${value?.toFixed(1)}${name?.includes('usage') ? '%' : ''}`,
              name?.replace('_', ' ')?.replace(/\b\w/g, l => l?.toUpperCase())
            ]}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line 
            type="monotone" 
            dataKey="tasks_per_minute" 
            stroke="#60B5FF" 
            strokeWidth={2}
            dot={false}
            name="Tasks/Min"
          />
          <Line 
            type="monotone" 
            dataKey="cpu_usage" 
            stroke="#FF9149" 
            strokeWidth={2}
            dot={false}
            name="CPU %"
          />
          <Line 
            type="monotone" 
            dataKey="success_rate" 
            stroke="#72BF78" 
            strokeWidth={2}
            dot={false}
            name="Success Rate"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
