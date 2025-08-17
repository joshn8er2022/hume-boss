
import { NextRequest, NextResponse } from 'next/server';
import { BossState } from '@/lib/types';
import { mockSystemOverview } from '@/lib/mock-data';

// GET /api/system/boss-state
export async function GET() {
  try {
    // In production, this would connect to the Python DSPY Boss system
    return NextResponse.json({
      state: mockSystemOverview.boss_state,
      data: mockSystemOverview.state_data
    });
  } catch (error) {
    console.error('Failed to get boss state:', error);
    return NextResponse.json(
      { error: 'Failed to get boss state' }, 
      { status: 500 }
    );
  }
}

// POST /api/system/boss-state
export async function POST(request: NextRequest) {
  try {
    const { state } = await request.json();
    
    // Validate state
    if (!Object.values(BossState).includes(state)) {
      return NextResponse.json(
        { error: 'Invalid boss state' }, 
        { status: 400 }
      );
    }
    
    // In production, this would call the Python DSPY Boss API
    console.log(`Setting boss state to: ${state}`);
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Failed to set boss state:', error);
    return NextResponse.json(
      { error: 'Failed to set boss state' }, 
      { status: 500 }
    );
  }
}
