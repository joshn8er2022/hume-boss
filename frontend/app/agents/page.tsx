
import Header from '@/components/header';
import AgentManagement from '@/components/agent-management';

export default function AgentsPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        title="Agent Management" 
        description="Monitor and manage all agentic and human agents in the system"
      />
      <div className="p-6">
        <AgentManagement />
      </div>
    </div>
  );
}
