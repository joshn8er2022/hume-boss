
import { NextResponse } from 'next/server';

// GET /api/auth/csrf
export async function GET() {
  // Generate a simple CSRF token (in production, use proper CSRF protection)
  const csrfToken = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  
  return NextResponse.json({
    csrfToken
  });
}
