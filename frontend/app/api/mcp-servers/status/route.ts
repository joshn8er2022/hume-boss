
import { NextRequest, NextResponse } from 'next/server';
import { mockMCPServers } from '@/lib/mock-data';

// GET /api/mcp-servers/status
export async function GET() {
  try {
    // In production, this would check actual MCP server statuses
    return NextResponse.json(mockMCPServers);
  } catch (error) {
    console.error('Failed to get MCP server status:', error);
    return NextResponse.json(
      { error: 'Failed to get MCP server status' }, 
      { status: 500 }
    );
  }
}
