
'use client';

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { TaskDefinition } from '@/lib/types';

interface TaskDistributionChartProps {
  tasks: TaskDefinition[];
}

const COLORS = {
  pending: '#FF9149',
  running: '#60B5FF', 
  completed: '#72BF78',
  failed: '#FF6363',
  cancelled: '#A19AD3'
};

const STATUS_LABELS = {
  pending: 'Pending',
  running: 'Running',
  completed: 'Completed', 
  failed: 'Failed',
  cancelled: 'Cancelled'
};

export default function TaskDistributionChart({ tasks }: TaskDistributionChartProps) {
  const taskCounts = tasks?.reduce((acc, task) => {
    acc[task?.status] = (acc[task?.status] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const chartData = Object?.entries(taskCounts)?.map(([status, count]) => ({
    name: STATUS_LABELS[status as keyof typeof STATUS_LABELS] || status,
    value: count,
    color: COLORS[status as keyof typeof COLORS] || '#A19AD3'
  }));

  const renderTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium">{data?.name}</p>
          <p className="text-sm text-gray-600">{data?.value} tasks</p>
          <p className="text-xs text-gray-500">
            {((data?.value / tasks?.length) * 100)?.toFixed(1)}% of total
          </p>
        </div>
      );
    }
    return null;
  };

  if (!tasks?.length) {
    return (
      <div className="h-80 flex items-center justify-center text-gray-500">
        <div className="text-center">
          <div className="text-2xl mb-2">📋</div>
          <p>No tasks available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            outerRadius={100}
            innerRadius={40}
            paddingAngle={2}
            dataKey="value"
            label={({ name, percent }) => `${name}: ${(percent * 100)?.toFixed(0)}%`}
            labelLine={false}
          >
            {chartData?.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry?.color} />
            ))}
          </Pie>
          <Tooltip content={renderTooltip} />
          <Legend 
            verticalAlign="top"
            height={36}
            wrapperStyle={{ fontSize: 11 }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
