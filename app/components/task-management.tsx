
'use client';

import { useState, useEffect } from 'react';
import { 
  ListTodo, 
  Plus, 
  Play, 
  Pause, 
  RotateCcw, 
  X, 
  Clock,
  CheckCircle,
  AlertCircle,
  Filter,
  Search
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import DashboardCard from '@/components/dashboard-card';
import { TaskDefinition, TaskStatus, TaskPriority } from '@/lib/types';
import { mockTasks } from '@/lib/mock-data';
import { 
  formatRelativeTime, 
  getStatusColor, 
  getPriorityLabel, 
  getPriorityColor,
  formatDuration
} from '@/lib/dashboard-utils';
import { useToast } from '@/hooks/use-toast';

export default function TaskManagement() {
  const { toast } = useToast();
  const [tasks, setTasks] = useState(mockTasks);
  const [filteredTasks, setFilteredTasks] = useState(mockTasks);
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Filter tasks based on status and search
  useEffect(() => {
    let filtered = tasks;
    
    if (selectedStatus !== 'all') {
      filtered = filtered?.filter(task => task?.status === selectedStatus);
    }
    
    if (searchQuery) {
      filtered = filtered?.filter(task =>
        task?.name?.toLowerCase()?.includes(searchQuery?.toLowerCase()) ||
        task?.description?.toLowerCase()?.includes(searchQuery?.toLowerCase())
      );
    }
    
    setFilteredTasks(filtered);
  }, [tasks, selectedStatus, searchQuery]);

  const taskCounts = {
    all: tasks?.length || 0,
    pending: tasks?.filter(t => t?.status === TaskStatus.PENDING)?.length || 0,
    running: tasks?.filter(t => t?.status === TaskStatus.RUNNING)?.length || 0,
    completed: tasks?.filter(t => t?.status === TaskStatus.COMPLETED)?.length || 0,
    failed: tasks?.filter(t => t?.status === TaskStatus.FAILED)?.length || 0,
  };

  const handleTaskAction = async (taskId: string, action: 'retry' | 'cancel') => {
    setIsLoading(true);
    
    try {
      await new Promise(resolve => setTimeout(resolve, 500));
      
      if (action === 'retry') {
        setTasks(prev => prev?.map(task => 
          task?.id === taskId 
            ? { ...task, status: TaskStatus.PENDING, retry_count: task?.retry_count + 1 }
            : task
        ));
      } else if (action === 'cancel') {
        setTasks(prev => prev?.map(task => 
          task?.id === taskId 
            ? { ...task, status: TaskStatus.CANCELLED }
            : task
        ));
      }

      toast({
        title: "Task Updated",
        description: `Task has been ${action === 'retry' ? 'retried' : 'cancelled'}`,
      });
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Action Failed",
        description: `Failed to ${action} task`,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusIcon = (status: TaskStatus) => {
    switch (status) {
      case TaskStatus.PENDING: return Clock;
      case TaskStatus.RUNNING: return Play;
      case TaskStatus.COMPLETED: return CheckCircle;
      case TaskStatus.FAILED: return AlertCircle;
      case TaskStatus.CANCELLED: return X;
      default: return Clock;
    }
  };

  return (
    <div className="space-y-6">
      {/* Stats Row */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-5">
        <DashboardCard
          title="All Tasks"
          value={taskCounts.all}
          description="Total tasks"
          icon={ListTodo}
          color="blue"
        />
        
        <DashboardCard
          title="Pending"
          value={taskCounts.pending}
          description="Waiting to start"
          icon={Clock}
          color="yellow"
        />
        
        <DashboardCard
          title="Running"
          value={taskCounts.running}
          description="Currently executing"
          icon={Play}
          color="blue"
        />
        
        <DashboardCard
          title="Completed"
          value={taskCounts.completed}
          description="Successfully finished"
          icon={CheckCircle}
          color="green"
        />
        
        <DashboardCard
          title="Failed"
          value={taskCounts.failed}
          description="Execution failed"
          icon={AlertCircle}
          color="red"
        />
      </div>

      {/* Filters and Search */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="relative">
            <Search className="h-4 w-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <Input
              placeholder="Search tasks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 w-64"
            />
          </div>
          
          <div className="flex items-center gap-2">
            <Filter className="h-4 w-4 text-gray-500" />
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Status</option>
              <option value="pending">Pending</option>
              <option value="running">Running</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="cancelled">Cancelled</option>
            </select>
          </div>
        </div>

        <Button onClick={() => console.log('Add task')}>
          <Plus className="h-4 w-4 mr-2" />
          Add Task
        </Button>
      </div>

      {/* Task List */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">
            Tasks ({filteredTasks?.length})
          </h2>
        </div>
        
        <div className="divide-y divide-gray-200">
          {filteredTasks?.map((task) => {
            const StatusIcon = getStatusIcon(task?.status);
            const duration = task?.started_at && task?.completed_at
              ? (new Date(task.completed_at).getTime() - new Date(task.started_at).getTime()) / 1000
              : null;
            
            return (
              <div key={task?.id} className="p-6 hover:bg-gray-50">
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-4 flex-1">
                    <div className="flex items-center justify-center w-10 h-10 rounded-full bg-gray-100">
                      <StatusIcon className="h-5 w-5 text-gray-600" />
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="font-semibold text-gray-900">{task?.name}</h3>
                        <Badge 
                          variant="outline"
                          className={getStatusColor(task?.status)}
                        >
                          {task?.status?.toUpperCase()}
                        </Badge>
                        <Badge 
                          variant="outline"
                          className={getPriorityColor(task?.priority)}
                        >
                          {getPriorityLabel(task?.priority)}
                        </Badge>
                      </div>
                      
                      <p className="text-sm text-gray-600 mb-3">{task?.description}</p>
                      
                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        <span>Created {formatRelativeTime(task?.created_at)}</span>
                        {task?.assigned_agent_id && (
                          <span>Assigned to Agent {task?.assigned_agent_id}</span>
                        )}
                        {duration && (
                          <span>Duration: {formatDuration(duration)}</span>
                        )}
                        {task?.retry_count > 0 && (
                          <span>Retries: {task?.retry_count}</span>
                        )}
                      </div>
                      
                      {task?.error_message && (
                        <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                          {task?.error_message}
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 ml-4">
                    {task?.status === TaskStatus.FAILED && task?.retry_count < task?.max_retries && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleTaskAction(task?.id, 'retry')}
                        disabled={isLoading}
                      >
                        <RotateCcw className="h-3 w-3 mr-1" />
                        Retry
                      </Button>
                    )}
                    
                    {(task?.status === TaskStatus.PENDING || task?.status === TaskStatus.RUNNING) && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => handleTaskAction(task?.id, 'cancel')}
                        disabled={isLoading}
                        className="text-red-600 hover:text-red-700"
                      >
                        <X className="h-3 w-3 mr-1" />
                        Cancel
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {filteredTasks?.length === 0 && (
          <div className="p-12 text-center">
            <ListTodo className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              {searchQuery || selectedStatus !== 'all' ? 'No tasks found' : 'No tasks yet'}
            </h3>
            <p className="text-gray-500 mb-4">
              {searchQuery || selectedStatus !== 'all' 
                ? 'Try adjusting your filters or search query'
                : 'Tasks will appear here once they are created'
              }
            </p>
            {(!searchQuery && selectedStatus === 'all') && (
              <Button onClick={() => console.log('Add first task')}>
                <Plus className="h-4 w-4 mr-2" />
                Add Task
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
