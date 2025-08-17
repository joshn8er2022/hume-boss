
import Header from '@/components/header';

export default function LogsPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        title="System Logs" 
        description="Real-time system logs and activity monitoring"
      />
      <div className="p-6">
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-8 text-center">
          <div className="text-6xl mb-4">📝</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">System Logs</h2>
          <p className="text-gray-600 mb-4">
            Real-time system logs, activity monitoring, and log analysis will be displayed here.
          </p>
          <div className="text-sm text-gray-500">
            This page will show filtered logs, search capabilities, and log level filtering.
          </div>
        </div>
      </div>
    </div>
  );
}
