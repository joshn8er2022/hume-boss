
import { NextRequest, NextResponse } from 'next/server';

// POST /api/auth/callback/credentials
export async function POST(request: NextRequest) {
  try {
    // Extract credentials from request
    const formData = await request.formData();
    const email = formData.get('email');
    const password = formData.get('password');
    
    // Basic validation
    if (!email || !password) {
      return NextResponse.json(
        { error: 'Email and password are required' }, 
        { status: 400 }
      );
    }
    
    console.log('Credentials login attempt:', { email });
    
    // Return redirect as expected by NextAuth pattern
    const redirectUrl = new URL('/', request.url);
    return NextResponse.redirect(redirectUrl, 302);
  } catch (error) {
    console.error('Credentials login error:', error);
    return NextResponse.redirect(new URL('/login?error=CredentialsSignin', request.url), 302);
  }
}

// GET /api/auth/callback/credentials
export async function GET(request: NextRequest) {
  // Handle OAuth callback redirects
  return NextResponse.redirect(new URL('/', request.url), 302);
}
