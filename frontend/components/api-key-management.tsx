
"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { toast } from "sonner";
import { Eye, EyeOff, Check, X, TestTube, Settings, Key } from "lucide-react";

interface LLMProvider {
  is_active: boolean;
  has_api_key: boolean;
  is_initialized: boolean;
  model: string;
  base_url: string;
  max_tokens?: number;
  temperature?: number;
}

interface LLMProviderStatus {
  active_provider: string | null;
  providers: Record<string, LLMProvider>;
}

const PROVIDER_INFO = {
  openai: {
    name: "OpenAI",
    description: "GPT-4 and GPT-3.5 models",
    icon: "🤖",
    placeholder: "sk-..."
  },
  grok: {
    name: "Grok (X.AI)",
    description: "Grok AI models by X",
    icon: "🚀",
    placeholder: "xai-..."
  },
  ollama: {
    name: "Ollama",
    description: "Local LLM server",
    icon: "🦙",
    placeholder: "Not required for local"
  },
  google: {
    name: "Google AI",
    description: "Gemini models",
    icon: "🔍",
    placeholder: "AIza..."
  },
  openrouter: {
    name: "OpenRouter",
    description: "Multiple model access",
    icon: "🌐",
    placeholder: "sk-or-v1-..."
  }
};

export default function APIKeyManagement() {
  const [providers, setProviders] = useState<LLMProviderStatus>({
    active_provider: null,
    providers: {}
  });
  const [loading, setLoading] = useState(true);
  const [apiKeys, setApiKeys] = useState<Record<string, string>>({});
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});
  const [testResults, setTestResults] = useState<Record<string, any>>({});

  useEffect(() => {
    fetchProviderStatus();
  }, []);

  const fetchProviderStatus = async () => {
    try {
      const response = await fetch('/api/llm-providers');
      const data = await response.json();
      
      if (data.success) {
        setProviders(data.data);
      }
    } catch (error) {
      console.error('Error fetching provider status:', error);
      toast.error('Failed to fetch provider status');
    } finally {
      setLoading(false);
    }
  };

  const setAPIKey = async (provider: string, apiKey: string) => {
    if (!apiKey.trim()) {
      toast.error('API key cannot be empty');
      return;
    }

    try {
      const response = await fetch('/api/llm-providers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'set_api_key',
          provider,
          api_key: apiKey
        })
      });

      const data = await response.json();

      if (data.success) {
        setProviders(data.data);
        setApiKeys(prev => ({ ...prev, [provider]: '' })); // Clear input
        toast.success(`API key set for ${PROVIDER_INFO[provider as keyof typeof PROVIDER_INFO]?.name}`);
      } else {
        toast.error(data.message || 'Failed to set API key');
      }
    } catch (error) {
      console.error('Error setting API key:', error);
      toast.error('Failed to set API key');
    }
  };

  const testProvider = async (provider: string) => {
    try {
      const response = await fetch('/api/llm-providers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'test_provider',
          provider
        })
      });

      const data = await response.json();

      if (data.success) {
        setTestResults(prev => ({ ...prev, [provider]: data.data }));
        
        if (data.data.status === 'success') {
          toast.success(`${PROVIDER_INFO[provider as keyof typeof PROVIDER_INFO]?.name} test successful`);
        } else {
          toast.error(data.data.message);
        }
      }
    } catch (error) {
      console.error('Error testing provider:', error);
      toast.error('Failed to test provider');
    }
  };

  const switchProvider = async (provider: string) => {
    try {
      const response = await fetch('/api/llm-providers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'switch_provider',
          provider
        })
      });

      const data = await response.json();

      if (data.success) {
        setProviders(data.data);
        toast.success(`Switched to ${PROVIDER_INFO[provider as keyof typeof PROVIDER_INFO]?.name}`);
      } else {
        toast.error(data.message || 'Failed to switch provider');
      }
    } catch (error) {
      console.error('Error switching provider:', error);
      toast.error('Failed to switch provider');
    }
  };

  const toggleKeyVisibility = (provider: string) => {
    setShowKeys(prev => ({ ...prev, [provider]: !prev[provider] }));
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="h-5 w-5" />
            API Key Management
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center p-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="h-5 w-5" />
            LLM Provider Management
          </CardTitle>
          <CardDescription>
            Configure API keys for different LLM providers. The autonomous DSPY engine requires at least one provider to operate.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {providers.active_provider && (
            <Alert className="mb-6 border-green-200 bg-green-50">
              <Check className="h-4 w-4 text-green-600" />
              <AlertDescription className="text-green-800">
                Active Provider: <strong>{PROVIDER_INFO[providers.active_provider as keyof typeof PROVIDER_INFO]?.name}</strong>
              </AlertDescription>
            </Alert>
          )}

          <Tabs defaultValue="configure" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="configure">Configure</TabsTrigger>
              <TabsTrigger value="test">Test & Switch</TabsTrigger>
              <TabsTrigger value="status">Status</TabsTrigger>
            </TabsList>

            <TabsContent value="configure" className="space-y-4">
              <div className="grid gap-6">
                {Object.entries(PROVIDER_INFO).map(([key, info]) => {
                  const provider = providers.providers[key];
                  return (
                    <Card key={key} className={provider?.is_active ? "border-green-200" : ""}>
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <span className="text-2xl">{info.icon}</span>
                            <div>
                              <CardTitle className="text-lg">{info.name}</CardTitle>
                              <CardDescription>{info.description}</CardDescription>
                            </div>
                          </div>
                          <div className="flex gap-2">
                            {provider?.has_api_key && (
                              <Badge variant="outline" className="text-green-600 border-green-200">
                                <Check className="h-3 w-3 mr-1" />
                                Configured
                              </Badge>
                            )}
                            {provider?.is_initialized && (
                              <Badge variant="outline" className="text-blue-600 border-blue-200">
                                Ready
                              </Badge>
                            )}
                            {providers.active_provider === key && (
                              <Badge className="bg-green-600 text-white">
                                Active
                              </Badge>
                            )}
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-3">
                          <div className="flex gap-2">
                            <div className="relative flex-1">
                              <Input
                                type={showKeys[key] ? "text" : "password"}
                                placeholder={info.placeholder}
                                value={apiKeys[key] || ''}
                                onChange={(e) => setApiKeys(prev => ({ ...prev, [key]: e.target.value }))}
                                className={provider?.has_api_key ? "border-green-200" : ""}
                              />
                              {key !== 'ollama' && (
                                <Button
                                  type="button"
                                  variant="ghost"
                                  size="sm"
                                  className="absolute right-0 top-0 h-full px-3"
                                  onClick={() => toggleKeyVisibility(key)}
                                >
                                  {showKeys[key] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                </Button>
                              )}
                            </div>
                            <Button
                              onClick={() => setAPIKey(key, apiKeys[key] || '')}
                              disabled={!apiKeys[key]?.trim() && key !== 'ollama'}
                              className="shrink-0"
                            >
                              Set Key
                            </Button>
                          </div>
                          
                          <div className="text-sm text-muted-foreground">
                            Model: {provider?.model || 'Not configured'} • 
                            Base URL: {provider?.base_url || 'Not configured'}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </TabsContent>

            <TabsContent value="test" className="space-y-4">
              <div className="grid gap-4">
                {Object.entries(providers.providers).map(([key, provider]) => (
                  <Card key={key}>
                    <CardContent className="pt-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className="text-xl">{PROVIDER_INFO[key as keyof typeof PROVIDER_INFO]?.icon}</span>
                          <div>
                            <h3 className="font-semibold">{PROVIDER_INFO[key as keyof typeof PROVIDER_INFO]?.name}</h3>
                            <p className="text-sm text-muted-foreground">
                              {provider.is_initialized ? 'Ready to use' : 'Not initialized'}
                            </p>
                          </div>
                        </div>
                        
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => testProvider(key)}
                            disabled={!provider.has_api_key}
                            className="flex items-center gap-2"
                          >
                            <TestTube className="h-4 w-4" />
                            Test
                          </Button>
                          
                          <Button
                            size="sm"
                            onClick={() => switchProvider(key)}
                            disabled={!provider.is_initialized || providers.active_provider === key}
                          >
                            {providers.active_provider === key ? 'Active' : 'Switch'}
                          </Button>
                        </div>
                      </div>
                      
                      {testResults[key] && (
                        <div className={`mt-3 p-3 rounded-md ${
                          testResults[key].status === 'success' 
                            ? 'bg-green-50 text-green-800 border border-green-200' 
                            : 'bg-red-50 text-red-800 border border-red-200'
                        }`}>
                          <p className="text-sm font-medium">{testResults[key].message}</p>
                          {testResults[key].test_response && (
                            <p className="text-sm mt-1 opacity-80">{testResults[key].test_response}</p>
                          )}
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="status" className="space-y-4">
              <div className="grid gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Provider Status Overview</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {Object.entries(providers.providers).map(([key, provider]) => (
                        <div key={key} className="flex items-center justify-between p-3 border rounded-lg">
                          <div className="flex items-center gap-3">
                            <span className="text-lg">{PROVIDER_INFO[key as keyof typeof PROVIDER_INFO]?.icon}</span>
                            <div>
                              <p className="font-medium">{PROVIDER_INFO[key as keyof typeof PROVIDER_INFO]?.name}</p>
                              <p className="text-sm text-muted-foreground">{provider.model}</p>
                            </div>
                          </div>
                          
                          <div className="flex items-center gap-2">
                            <Badge variant={provider.has_api_key ? "default" : "secondary"}>
                              {provider.has_api_key ? "Configured" : "No Key"}
                            </Badge>
                            <Badge variant={provider.is_initialized ? "default" : "secondary"}>
                              {provider.is_initialized ? "Ready" : "Not Ready"}
                            </Badge>
                            {providers.active_provider === key && (
                              <Badge className="bg-green-600 text-white">Active</Badge>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
