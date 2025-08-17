
import { NextRequest, NextResponse } from 'next/server';

// Mock LLM provider data
let mockLLMProviders = {
  active_provider: null,
  providers: {
    openai: { 
      is_active: false, 
      has_api_key: false, 
      is_initialized: false,
      model: "gpt-4-turbo-preview",
      base_url: "https://api.openai.com/v1"
    },
    grok: { 
      is_active: false, 
      has_api_key: false, 
      is_initialized: false,
      model: "grok-beta",
      base_url: "https://api.x.ai/v1"
    },
    ollama: { 
      is_active: false, 
      has_api_key: false, 
      is_initialized: false,
      model: "llama3.1",
      base_url: "http://localhost:11434"
    },
    google: { 
      is_active: false, 
      has_api_key: false, 
      is_initialized: false,
      model: "gemini-1.5-pro",
      base_url: "https://generativelanguage.googleapis.com/v1beta"
    },
    openrouter: { 
      is_active: false, 
      has_api_key: false, 
      is_initialized: false,
      model: "anthropic/claude-3-haiku",
      base_url: "https://openrouter.ai/api/v1"
    }
  }
};

export async function GET(request: NextRequest) {
  try {
    return NextResponse.json({
      success: true,
      data: mockLLMProviders
    });
  } catch (error) {
    console.error('Error getting LLM providers:', error);
    return NextResponse.json({
      success: false,
      message: "Internal server error"
    }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, provider, api_key, config } = body;

    if (action === 'set_api_key') {
      if (!provider || !api_key) {
        return NextResponse.json({
          success: false,
          message: "Provider and API key are required"
        }, { status: 400 });
      }

      // Update mock data  
      if (provider in mockLLMProviders.providers) {
        const providerData = mockLLMProviders.providers[provider as keyof typeof mockLLMProviders.providers];
        providerData.has_api_key = true;
        providerData.is_active = true;
        providerData.is_initialized = true;
        
        // Set as active provider if none is set
        if (!mockLLMProviders.active_provider) {
          mockLLMProviders.active_provider = provider;
        }
      }

      return NextResponse.json({
        success: true,
        message: `API key set for ${provider}`,
        data: mockLLMProviders
      });
    }

    if (action === 'switch_provider') {
      if (!provider) {
        return NextResponse.json({
          success: false,
          message: "Provider is required"
        }, { status: 400 });
      }

      if (provider in mockLLMProviders.providers && mockLLMProviders.providers[provider as keyof typeof mockLLMProviders.providers]?.is_initialized) {
        mockLLMProviders.active_provider = provider;
        
        return NextResponse.json({
          success: true,
          message: `Switched to ${provider}`,
          data: mockLLMProviders
        });
      } else {
        return NextResponse.json({
          success: false,
          message: `Provider ${provider} is not initialized`
        }, { status: 400 });
      }
    }

    if (action === 'test_provider') {
      if (!provider) {
        return NextResponse.json({
          success: false,
          message: "Provider is required"
        }, { status: 400 });
      }

      // Mock test result
      const providerData = provider in mockLLMProviders.providers 
        ? mockLLMProviders.providers[provider as keyof typeof mockLLMProviders.providers] 
        : null;
        
      const testResult = {
        status: providerData?.has_api_key ? "success" : "error",
        message: providerData?.has_api_key 
          ? `Provider ${provider} is working correctly`
          : `Provider ${provider} needs API key`,
        test_response: providerData?.has_api_key 
          ? "Hello! I'm working correctly."
          : undefined
      };

      return NextResponse.json({
        success: true,
        data: testResult
      });
    }

    return NextResponse.json({
      success: false,
      message: "Invalid action"
    }, { status: 400 });

  } catch (error) {
    console.error('Error managing LLM providers:', error);
    return NextResponse.json({
      success: false,
      message: "Internal server error"
    }, { status: 500 });
  }
}
