
import Header from '@/components/header';
import TaskManagement from '@/components/task-management';

export default function TasksPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        title="Task Queue Management" 
        description="Monitor and manage all tasks in the system queue"
      />
      <div className="p-6">
        <TaskManagement />
      </div>
    </div>
  );
}
