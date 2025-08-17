'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Settings, 
  Brain, 
  Zap, 
  Check, 
  ChevronDown,
  Loader2,
  AlertCircle,
  Info
} from 'lucide-react';
import { toast } from 'sonner';

interface LLMProvider {
  name: string;
  type: string;
  models: string[];
  is_active: boolean;
  config: any;
}

interface Agent {
  id: string;
  name: string;
  type: string;
  model_name?: string;
  provider?: string;
  is_available: boolean;
}

interface AgentModelSelectorProps {
  agent: Agent;
  onModelUpdate?: (agentId: string, modelName: string, provider?: string) => void;
  className?: string;
}

export default function AgentModelSelector({ 
  agent, 
  onModelUpdate,
  className = '' 
}: AgentModelSelectorProps) {
  const [providers, setProviders] = useState<LLMProvider[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>(agent?.provider || '');
  const [selectedModel, setSelectedModel] = useState<string>(agent?.model_name || agent?.model || '');
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Fetch available providers and models
  useEffect(() => {
    fetchProviders();
  }, []);

  const fetchProviders = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/llm-providers');
      if (response.ok) {
        const data = await response.json();
        setProviders(data.providers || []);
        
        // Set default provider if none selected
        if (!selectedProvider && data.providers.length > 0) {
          const activeProvider = data.providers.find((p: LLMProvider) => p.is_active) || data.providers[0];
          setSelectedProvider(activeProvider.name);
          
          // Set default model
          if (!selectedModel && activeProvider.models.length > 0) {
            setSelectedModel(activeProvider.models[0]);
          }
        }
      }
    } catch (error) {
      console.error('Error fetching providers:', error);
      toast.error('Failed to load LLM providers');
    } finally {
      setIsLoading(false);
    }
  };

  const handleModelUpdate = async () => {
    if (!selectedModel || !selectedProvider) {
      toast.error('Please select both provider and model');
      return;
    }

    setIsSaving(true);
    try {
      const response = await fetch(`/api/agents/${agent.id}/model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          agent_id: agent.id,
          model_name: selectedModel,
          provider: selectedProvider
        })
      });

      if (response.ok) {
        toast.success(`Updated ${agent.name} to use ${selectedModel}`);
        onModelUpdate?.(agent.id, selectedModel, selectedProvider);
        setIsOpen(false);
      } else {
        const error = await response.json();
        toast.error(error.detail || 'Failed to update agent model');
      }
    } catch (error) {
      console.error('Error updating agent model:', error);
      toast.error('Failed to update agent model');
    } finally {
      setIsSaving(false);
    }
  };

  const currentProvider = providers.find(p => p.name === selectedProvider);
  const availableModels = currentProvider?.models || [];

  const getProviderIcon = (providerType: string) => {
    switch (providerType.toLowerCase()) {
      case 'openai':
        return '🤖';
      case 'anthropic':
        return '🧠';
      case 'google':
        return '🔍';
      case 'local':
        return '💻';
      default:
        return '⚡';
    }
  };

  const getModelDisplayName = (modelName: string) => {
    // Clean up model names for display
    return modelName
      .replace(/^(gpt-|claude-|gemini-|llama-)/i, '')
      .replace(/-/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase());
  };

  if (agent?.type === 'human') {
    return (
      <div className={`flex items-center space-x-2 ${className}`}>
        <div className="flex items-center space-x-1 text-sm text-gray-500">
          <Info className="w-4 h-4" />
          <span>Human Agent</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      {/* Current Model Display */}
      <motion.button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          flex items-center justify-between w-full px-3 py-2 text-sm
          border rounded-lg transition-all duration-200
          ${agent?.is_available 
            ? 'border-green-200 bg-green-50 hover:bg-green-100' 
            : 'border-gray-200 bg-gray-50 hover:bg-gray-100'
          }
          ${isOpen ? 'ring-2 ring-blue-500 ring-opacity-20' : ''}
        `}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className="flex items-center space-x-2">
          <Brain className="w-4 h-4 text-gray-600" />
          <div className="text-left">
            <div className="font-medium">
              {selectedModel ? getModelDisplayName(selectedModel) : 'No Model'}
            </div>
            {selectedProvider && (
              <div className="text-xs text-gray-500 flex items-center space-x-1">
                <span>{getProviderIcon(currentProvider?.type || '')}</span>
                <span>{selectedProvider}</span>
              </div>
            )}
          </div>
        </div>
        <ChevronDown 
          className={`w-4 h-4 transition-transform duration-200 ${
            isOpen ? 'rotate-180' : ''
          }`} 
        />
      </motion.button>

      {/* Dropdown Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="absolute top-full left-0 right-0 mt-2 bg-white border border-gray-200 rounded-lg shadow-lg z-50"
          >
            <div className="p-4 space-y-4">
              {/* Provider Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Provider
                </label>
                <div className="space-y-1">
                  {isLoading ? (
                    <div className="flex items-center justify-center py-4">
                      <Loader2 className="w-5 h-5 animate-spin text-gray-400" />
                    </div>
                  ) : providers.length === 0 ? (
                    <div className="flex items-center space-x-2 text-sm text-gray-500 py-2">
                      <AlertCircle className="w-4 h-4" />
                      <span>No providers available</span>
                    </div>
                  ) : (
                    providers.map((provider) => (
                      <motion.button
                        key={provider.name}
                        onClick={() => {
                          setSelectedProvider(provider.name);
                          // Reset model selection when provider changes
                          if (provider.models.length > 0) {
                            setSelectedModel(provider.models[0]);
                          }
                        }}
                        className={`
                          w-full flex items-center justify-between px-3 py-2 text-sm
                          rounded-md transition-colors duration-150
                          ${selectedProvider === provider.name
                            ? 'bg-blue-100 text-blue-900 border border-blue-200'
                            : 'hover:bg-gray-100 text-gray-700'
                          }
                          ${!provider.is_active ? 'opacity-50' : ''}
                        `}
                        whileHover={{ scale: 1.01 }}
                        whileTap={{ scale: 0.99 }}
                        disabled={!provider.is_active}
                      >
                        <div className="flex items-center space-x-2">
                          <span>{getProviderIcon(provider.type)}</span>
                          <span className="font-medium">{provider.name}</span>
                          {!provider.is_active && (
                            <span className="text-xs text-red-500">(Inactive)</span>
                          )}
                        </div>
                        {selectedProvider === provider.name && (
                          <Check className="w-4 h-4 text-blue-600" />
                        )}
                      </motion.button>
                    ))
                  )}
                </div>
              </div>

              {/* Model Selection */}
              {selectedProvider && availableModels.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Model
                  </label>
                  <div className="space-y-1 max-h-40 overflow-y-auto">
                    {availableModels.map((model) => (
                      <motion.button
                        key={model}
                        onClick={() => setSelectedModel(model)}
                        className={`
                          w-full flex items-center justify-between px-3 py-2 text-sm
                          rounded-md transition-colors duration-150
                          ${selectedModel === model
                            ? 'bg-green-100 text-green-900 border border-green-200'
                            : 'hover:bg-gray-100 text-gray-700'
                          }
                        `}
                        whileHover={{ scale: 1.01 }}
                        whileTap={{ scale: 0.99 }}
                      >
                        <span className="font-medium">{getModelDisplayName(model)}</span>
                        {selectedModel === model && (
                          <Check className="w-4 h-4 text-green-600" />
                        )}
                      </motion.button>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex items-center justify-end space-x-2 pt-2 border-t border-gray-100">
                <button
                  onClick={() => setIsOpen(false)}
                  className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 transition-colors"
                >
                  Cancel
                </button>
                <motion.button
                  onClick={handleModelUpdate}
                  disabled={isSaving || !selectedModel || !selectedProvider}
                  className={`
                    px-4 py-1.5 text-sm font-medium rounded-md transition-all duration-200
                    ${isSaving || !selectedModel || !selectedProvider
                      ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                    }
                  `}
                  whileHover={!isSaving && selectedModel && selectedProvider ? { scale: 1.05 } : {}}
                  whileTap={!isSaving && selectedModel && selectedProvider ? { scale: 0.95 } : {}}
                >
                  {isSaving ? (
                    <div className="flex items-center space-x-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Updating...</span>
                    </div>
                  ) : (
                    'Update Model'
                  )}
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Bulk model selector for multiple agents
export function BulkAgentModelSelector({ 
  agents, 
  onBulkUpdate 
}: { 
  agents: Agent[]; 
  onBulkUpdate?: (updates: Array<{agentId: string, modelName: string, provider: string}>) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedProvider, setSelectedProvider] = useState<string>('');
  const [providers, setProviders] = useState<LLMProvider[]>([]);

  const agenticAgents = agents.filter(agent => agent.type === 'agentic');

  useEffect(() => {
    if (isOpen) {
      fetchProviders();
    }
  }, [isOpen]);

  const fetchProviders = async () => {
    try {
      const response = await fetch('/api/llm-providers');
      if (response.ok) {
        const data = await response.json();
        setProviders(data.providers || []);
      }
    } catch (error) {
      console.error('Error fetching providers:', error);
    }
  };

  const handleBulkUpdate = async () => {
    if (selectedAgents.length === 0 || !selectedModel || !selectedProvider) {
      toast.error('Please select agents, provider, and model');
      return;
    }

    const updates = selectedAgents.map(agentId => ({
      agentId,
      modelName: selectedModel,
      provider: selectedProvider
    }));

    try {
      // Update each agent
      await Promise.all(
        updates.map(update =>
          fetch(`/api/agents/${update.agentId}/model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              agent_id: update.agentId,
              model_name: update.modelName,
              provider: update.provider
            })
          })
        )
      );

      toast.success(`Updated ${selectedAgents.length} agents to use ${selectedModel}`);
      onBulkUpdate?.(updates);
      setIsOpen(false);
      setSelectedAgents([]);
    } catch (error) {
      console.error('Error bulk updating agents:', error);
      toast.error('Failed to update some agents');
    }
  };

  if (agenticAgents.length === 0) {
    return null;
  }

  return (
    <div className="relative">
      <motion.button
        onClick={() => setIsOpen(true)}
        className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <Settings className="w-4 h-4" />
        <span>Bulk Update Models</span>
      </motion.button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={() => setIsOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-lg p-6 max-w-md w-full mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <h3 className="text-lg font-semibold mb-4">Bulk Update Agent Models</h3>
              
              {/* Agent Selection */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Agents
                </label>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {agenticAgents.map(agent => (
                    <label key={agent.id} className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={selectedAgents.includes(agent.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedAgents([...selectedAgents, agent.id]);
                          } else {
                            setSelectedAgents(selectedAgents.filter(id => id !== agent.id));
                          }
                        }}
                        className="rounded border-gray-300"
                      />
                      <span className="text-sm">{agent.name}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Provider and Model Selection */}
              {/* Similar to individual selector but simplified */}
              
              <div className="flex justify-end space-x-2">
                <button
                  onClick={() => setIsOpen(false)}
                  className="px-4 py-2 text-gray-600 hover:text-gray-800"
                >
                  Cancel
                </button>
                <button
                  onClick={handleBulkUpdate}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Update Selected
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}