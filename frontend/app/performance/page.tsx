
import Header from '@/components/header';

export default function PerformancePage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        title="Performance Analytics" 
        description="System performance metrics and analytics dashboard"
      />
      <div className="p-6">
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-8 text-center">
          <div className="text-6xl mb-4">📊</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Performance Analytics</h2>
          <p className="text-gray-600 mb-4">
            Detailed performance metrics, charts, and analytics will be displayed here.
          </p>
          <div className="text-sm text-gray-500">
            This page will show task completion rates, agent utilization, system resource usage, and performance trends.
          </div>
        </div>
      </div>
    </div>
  );
}
