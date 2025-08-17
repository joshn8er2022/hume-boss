
import Header from '@/components/header';
import MCPServerManagement from '@/components/mcp-server-management';

export default function MCPServersPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        title="MCP Server Management" 
        description="Monitor and manage all Model Context Protocol server connections"
      />
      <div className="p-6">
        <MCPServerManagement />
      </div>
    </div>
  );
}
