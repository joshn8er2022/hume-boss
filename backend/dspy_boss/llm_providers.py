
"""
LLM Provider Manager - Handles multiple LLM providers with API key management
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger

import dspy
from dspy.teleprompt import BootstrapFewShot


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GROK = "grok"
    OLLAMA = "ollama"
    GOOGLE = "google"
    OPENROUTER = "openrouter"


@dataclass
class LLMConfig:
    provider: LLMProvider
    api_key: Optional[str]
    base_url: Optional[str]
    model: str
    max_tokens: int = 4000
    temperature: float = 0.7
    is_active: bool = True


class LLMProviderManager:
    """Manages multiple LLM providers with API key configuration"""
    
    def __init__(self):
        self.providers: Dict[LLMProvider, LLMConfig] = {}
        self.active_provider: Optional[LLMProvider] = None
        self.initialized_models: Dict[LLMProvider, Any] = {}
        
        # Initialize with default configurations
        self._setup_default_configs()
        
    def _setup_default_configs(self):
        """Setup default configurations for all providers"""
        
        # OpenAI Configuration
        self.providers[LLMProvider.OPENAI] = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key=None,  # Will be set via API key management
            base_url="https://api.openai.com/v1",
            model="gpt-4-turbo-preview",
            max_tokens=4000,
            temperature=0.7
        )
        
        # Grok Configuration (using OpenAI-compatible API)
        self.providers[LLMProvider.GROK] = LLMConfig(
            provider=LLMProvider.GROK,
            api_key=None,
            base_url="https://api.x.ai/v1",  # Grok API endpoint
            model="grok-beta",
            max_tokens=4000,
            temperature=0.7
        )
        
        # Ollama Configuration
        self.providers[LLMProvider.OLLAMA] = LLMConfig(
            provider=LLMProvider.OLLAMA,
            api_key=None,  # Ollama typically doesn't need API keys for local
            base_url="http://localhost:11434",  # Default Ollama endpoint
            model="llama3.1",  # Default model
            max_tokens=4000,
            temperature=0.7
        )
        
        # Google AI Configuration
        self.providers[LLMProvider.GOOGLE] = LLMConfig(
            provider=LLMProvider.GOOGLE,
            api_key=None,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-1.5-pro",
            max_tokens=4000,
            temperature=0.7
        )
        
        # OpenRouter Configuration
        self.providers[LLMProvider.OPENROUTER] = LLMConfig(
            provider=LLMProvider.OPENROUTER,
            api_key=None,
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3-haiku",  # Default model
            max_tokens=4000,
            temperature=0.7
        )
        
    def set_api_key(self, provider: LLMProvider, api_key: str):
        """Set API key for a specific provider"""
        if provider in self.providers:
            self.providers[provider].api_key = api_key
            self.providers[provider].is_active = True
            logger.info(f"🔑 API key set for {provider.value}")
            
            # Try to initialize the provider
            asyncio.create_task(self._initialize_provider(provider))
        else:
            logger.error(f"❌ Unknown provider: {provider}")
            
    def set_provider_config(self, provider: LLMProvider, config_updates: Dict[str, Any]):
        """Update provider configuration"""
        if provider in self.providers:
            config = self.providers[provider]
            for key, value in config_updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            logger.info(f"⚙️ Updated configuration for {provider.value}")
        else:
            logger.error(f"❌ Unknown provider: {provider}")
            
    async def _initialize_provider(self, provider: LLMProvider):
        """Initialize a specific provider"""
        try:
            config = self.providers[provider]
            
            if not config.api_key and provider != LLMProvider.OLLAMA:
                logger.warning(f"⚠️ No API key set for {provider.value}")
                return False
                
            if provider == LLMProvider.OPENAI:
                model = dspy.LM(
                    model=f"openai/{config.model}",
                    api_key=config.api_key,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                
            elif provider == LLMProvider.GROK:
                # Grok uses OpenAI-compatible API
                model = dspy.LM(
                    model=f"openai/{config.model}",
                    api_key=config.api_key,
                    api_base=config.base_url,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                
            elif provider == LLMProvider.OLLAMA:
                # Ollama integration
                model = dspy.LM(
                    model=f"ollama/{config.model}",
                    api_base=config.base_url,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                
            elif provider == LLMProvider.GOOGLE:
                # Google AI integration
                model = dspy.LM(
                    model=f"google/{config.model}",
                    api_key=config.api_key,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                
            elif provider == LLMProvider.OPENROUTER:
                # OpenRouter uses OpenAI-compatible API
                model = dspy.LM(
                    model=f"openrouter/{config.model}",
                    api_key=config.api_key,
                    api_base=config.base_url,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                
            else:
                logger.error(f"❌ Unknown provider type: {provider}")
                return False
                
            self.initialized_models[provider] = model
            logger.info(f"✅ Successfully initialized {provider.value}")
            
            # Set as active provider if none is set
            if not self.active_provider:
                self.active_provider = provider
                dspy.configure(lm=model)
                logger.info(f"🎯 Set {provider.value} as active provider")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize {provider.value}: {e}")
            config.is_active = False
            return False
            
    def switch_provider(self, provider: LLMProvider):
        """Switch to a different provider"""
        if provider in self.initialized_models:
            self.active_provider = provider
            dspy.configure(lm=self.initialized_models[provider])
            logger.info(f"🔄 Switched to {provider.value}")
            return True
        else:
            logger.error(f"❌ Provider {provider.value} not initialized")
            return False
            
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {
            "active_provider": self.active_provider.value if self.active_provider else None,
            "providers": {}
        }
        
        for provider, config in self.providers.items():
            status["providers"][provider.value] = {
                "is_active": config.is_active,
                "has_api_key": bool(config.api_key),
                "model": config.model,
                "base_url": config.base_url,
                "is_initialized": provider in self.initialized_models,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            }
            
        return status
        
    def get_available_providers(self) -> List[str]:
        """Get list of available/initialized providers"""
        return [
            provider.value 
            for provider in self.initialized_models.keys()
        ]
        
    async def test_provider(self, provider: LLMProvider) -> Dict[str, Any]:
        """Test a provider connection"""
        try:
            if provider not in self.initialized_models:
                await self._initialize_provider(provider)
                
            if provider not in self.initialized_models:
                return {"status": "error", "message": "Provider not initialized"}
                
            # Simple test query
            model = self.initialized_models[provider]
            original_active = self.active_provider
            
            # Temporarily switch to test provider
            dspy.configure(lm=model)
            
            # Create a simple signature for testing
            class TestSignature(dspy.Signature):
                input_text = dspy.InputField()
                output = dspy.OutputField()
                
            test_module = dspy.ChainOfThought(TestSignature)
            result = test_module(input_text="Hello, are you working correctly?")
            
            # Restore original provider
            if original_active and original_active in self.initialized_models:
                dspy.configure(lm=self.initialized_models[original_active])
                self.active_provider = original_active
                
            return {
                "status": "success",
                "message": f"Provider {provider.value} is working correctly",
                "test_response": result.output[:100] + "..." if len(result.output) > 100 else result.output
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Provider test failed: {str(e)}"
            }
            
    async def initialize_all_providers(self):
        """Initialize all providers that have API keys"""
        logger.info("🔄 Initializing all available providers...")
        
        initialized_count = 0
        for provider in self.providers.keys():
            config = self.providers[provider]
            if config.api_key or provider == LLMProvider.OLLAMA:  # Ollama might not need API key
                success = await self._initialize_provider(provider)
                if success:
                    initialized_count += 1
                    
        logger.info(f"✅ Initialized {initialized_count} providers")
        return initialized_count
        
    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama-specific configuration for Docker setup"""
        ollama_config = self.providers[LLMProvider.OLLAMA]
        return {
            "base_url": ollama_config.base_url,
            "model": ollama_config.model,
            "max_tokens": ollama_config.max_tokens,
            "temperature": ollama_config.temperature,
            "docker_command": f"docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama",
            "pull_model_command": f"docker exec -it ollama ollama pull {ollama_config.model}",
            "status_check_url": f"{ollama_config.base_url}/api/tags"
        }
        
    def export_config(self) -> Dict[str, Any]:
        """Export provider configuration (without API keys for security)"""
        config = {}
        for provider, provider_config in self.providers.items():
            config[provider.value] = {
                "model": provider_config.model,
                "base_url": provider_config.base_url,
                "max_tokens": provider_config.max_tokens,
                "temperature": provider_config.temperature,
                "is_active": provider_config.is_active,
                "has_api_key": bool(provider_config.api_key)
            }
        return config
        
    def import_config(self, config: Dict[str, Any]):
        """Import provider configuration (API keys must be set separately)"""
        for provider_name, provider_config in config.items():
            try:
                provider = LLMProvider(provider_name)
                if provider in self.providers:
                    self.providers[provider].model = provider_config.get("model", self.providers[provider].model)
                    self.providers[provider].base_url = provider_config.get("base_url", self.providers[provider].base_url)
                    self.providers[provider].max_tokens = provider_config.get("max_tokens", self.providers[provider].max_tokens)
                    self.providers[provider].temperature = provider_config.get("temperature", self.providers[provider].temperature)
                    self.providers[provider].is_active = provider_config.get("is_active", True)
                    
            except ValueError:
                logger.error(f"❌ Unknown provider in config: {provider_name}")
                
        logger.info("📥 Imported provider configuration")
