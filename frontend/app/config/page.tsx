
import Header from '@/components/header';

export default function ConfigPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        title="System Configuration" 
        description="System configuration management and settings"
      />
      <div className="p-6">
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-8 text-center">
          <div className="text-6xl mb-4">⚙️</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">System Configuration</h2>
          <p className="text-gray-600 mb-4">
            System settings, configuration management, and environment variables will be displayed here.
          </p>
          <div className="text-sm text-gray-500">
            This page will show MCP server configurations, agent settings, and system parameters.
          </div>
        </div>
      </div>
    </div>
  );
}
