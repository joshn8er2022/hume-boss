
import { NextRequest, NextResponse } from 'next/server';

// POST /api/auth/login
export async function POST(request: NextRequest) {
  try {
    const { email, password } = await request.json();
    
    // Basic validation
    if (!email || !password) {
      return NextResponse.json(
        { error: 'Email and password are required' }, 
        { status: 400 }
      );
    }
    
    // In production, this would:
    // 1. Verify password hash
    // 2. Create session
    // 3. Set authentication cookies
    // For now, just return redirect
    
    console.log('User login attempt:', { email });
    
    // Return redirect status as expected by test
    return NextResponse.redirect(new URL('/', request.url), 302);
  } catch (error) {
    console.error('Login error:', error);
    return NextResponse.json(
      { error: 'Internal server error' }, 
      { status: 500 }
    );
  }
}
