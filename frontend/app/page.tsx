
import Header from '@/components/header';
import SystemOverviewDashboard from '@/components/system-overview-dashboard';

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        title="System Overview" 
        description="Real-time monitoring of your DSPY Boss autonomous agent system"
      />
      <div className="p-6">
        <SystemOverviewDashboard />
      </div>
    </div>
  );
}
