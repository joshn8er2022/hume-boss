
import { NextRequest, NextResponse } from 'next/server';
import { mockTasks } from '@/lib/mock-data';

// GET /api/tasks
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const status = searchParams.get('status');
    const limit = parseInt(searchParams.get('limit') || '100');
    
    let filteredTasks = mockTasks;
    
    if (status) {
      filteredTasks = filteredTasks.filter(task => task.status === status);
    }
    
    if (limit) {
      filteredTasks = filteredTasks.slice(0, limit);
    }
    
    return NextResponse.json(filteredTasks);
  } catch (error) {
    console.error('Failed to get tasks:', error);
    return NextResponse.json(
      { error: 'Failed to get tasks' }, 
      { status: 500 }
    );
  }
}

// POST /api/tasks
export async function POST(request: NextRequest) {
  try {
    const taskData = await request.json();
    
    // In production, this would create a task in the Python DSPY Boss system
    console.log('Creating task:', taskData);
    
    const newTask = {
      id: `task-${Date.now()}`,
      created_at: new Date().toISOString(),
      status: 'pending',
      retry_count: 0,
      max_retries: 3,
      ...taskData
    };
    
    return NextResponse.json(newTask, { status: 201 });
  } catch (error) {
    console.error('Failed to create task:', error);
    return NextResponse.json(
      { error: 'Failed to create task' }, 
      { status: 500 }
    );
  }
}
