
import { NextRequest, NextResponse } from 'next/server';

// POST /api/signup
export async function POST(request: NextRequest) {
  try {
    const { email, password, firstName, lastName } = await request.json();
    
    // Basic validation
    if (!email || !password) {
      return NextResponse.json(
        { error: 'Email and password are required' }, 
        { status: 400 }
      );
    }
    
    // In production, this would:
    // 1. Hash the password
    // 2. Store user in database
    // 3. Send verification email
    // For now, just return success
    
    console.log('User signup attempt:', { email, firstName, lastName });
    
    return NextResponse.json({
      message: 'User created successfully',
      user: {
        id: `user-${Date.now()}`,
        email,
        firstName: firstName || '',
        lastName: lastName || ''
      }
    }, { status: 201 });
  } catch (error) {
    console.error('Signup error:', error);
    return NextResponse.json(
      { error: 'Internal server error' }, 
      { status: 500 }
    );
  }
}
