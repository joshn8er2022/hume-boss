
import { NextRequest, NextResponse } from 'next/server';
import { mockSystemOverview } from '@/lib/mock-data';

// GET /api/system/overview
export async function GET() {
  try {
    // In production, this would connect to the Python DSPY Boss system
    // For now, return mock data
    return NextResponse.json(mockSystemOverview);
  } catch (error) {
    console.error('Failed to get system overview:', error);
    return NextResponse.json(
      { error: 'Failed to get system overview' }, 
      { status: 500 }
    );
  }
}
