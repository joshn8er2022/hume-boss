
import { NextRequest, NextResponse } from 'next/server';
import { mockAgents } from '@/lib/mock-data';

// GET /api/agents
export async function GET() {
  try {
    // In production, this would connect to the Python DSPY Boss system
    return NextResponse.json(mockAgents);
  } catch (error) {
    console.error('Failed to get agents:', error);
    return NextResponse.json(
      { error: 'Failed to get agents' }, 
      { status: 500 }
    );
  }
}

// POST /api/agents
export async function POST(request: NextRequest) {
  try {
    const agentData = await request.json();
    
    // In production, this would create an agent in the Python DSPY Boss system
    console.log('Creating agent:', agentData);
    
    const newAgent = {
      id: `agent-${Date.now()}`,
      created_at: new Date().toISOString(),
      is_available: true,
      max_concurrent_tasks: 3,
      capabilities: [],
      ...agentData
    };
    
    return NextResponse.json(newAgent, { status: 201 });
  } catch (error) {
    console.error('Failed to create agent:', error);
    return NextResponse.json(
      { error: 'Failed to create agent' }, 
      { status: 500 }
    );
  }
}
