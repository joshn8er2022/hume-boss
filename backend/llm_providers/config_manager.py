
"""
Configuration manager for LLM providers
Handles API keys, model configurations, and provider settings
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from loguru import logger
from pathlib import Path


class ProviderType(Enum):
    """Supported LLM provider types"""
    OPENAI = "openai"
    GROK = "grok"
    OLLAMA = "ollama"
    GOOGLE = "google"
    OPENROUTER = "openrouter"


class ProviderConfig(BaseModel):
    """Configuration for a specific LLM provider"""
    
    provider_type: ProviderType = Field(..., description="Type of provider")
    provider_name: str = Field(..., description="Human-readable provider name")
    
    # Authentication
    api_key: Optional[str] = Field(None, description="API key for authentication")
    api_base: Optional[str] = Field(None, description="Custom API base URL")
    
    # Connection settings
    timeout_seconds: int = Field(default=30, description="Request timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay_seconds: float = Field(default=1.0, description="Delay between retries")
    
    # Model configurations
    available_models: List[str] = Field(default_factory=list, description="Available models")
    default_model: Optional[str] = Field(None, description="Default model to use")
    
    # Generation parameters
    default_temperature: float = Field(default=0.7, description="Default temperature")
    default_max_tokens: Optional[int] = Field(default=None, description="Default max tokens")
    default_top_p: Optional[float] = Field(default=None, description="Default top-p")
    
    # Provider-specific settings
    provider_settings: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific settings")
    
    # Status and metadata
    enabled: bool = Field(default=True, description="Whether provider is enabled")
    priority: int = Field(default=3, description="Provider priority (1=highest)")
    last_used: Optional[datetime] = None
    error_count: int = 0
    success_count: int = 0
    
    # Docker settings (for Ollama)
    docker_config: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
    
    @validator('api_key')
    def validate_api_key(cls, v, values):
        """Validate API key requirements"""
        provider_type = values.get('provider_type')
        
        # Ollama doesn't require API key if running locally
        if provider_type == ProviderType.OLLAMA:
            return v
        
        # Other providers require API key
        if not v and provider_type in [ProviderType.OPENAI, ProviderType.GROK, ProviderType.GOOGLE, ProviderType.OPENROUTER]:
            raise ValueError(f"API key required for {provider_type.value}")
        
        return v
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.error_count
        if total == 0:
            return 1.0
        return self.success_count / total
    
    def record_success(self):
        """Record successful API call"""
        self.success_count += 1
        self.last_used = datetime.utcnow()
    
    def record_error(self):
        """Record failed API call"""
        self.error_count += 1


class LLMConfigManager:
    """
    Manages LLM provider configurations, API keys, and settings
    """
    
    def __init__(self, config_dir: str = "llm_config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.config_file = self.config_dir / "providers.json"
        self.secrets_file = self.config_dir / "secrets.json"
        
        # Provider configurations
        self.providers: Dict[str, ProviderConfig] = {}
        
        # Load existing configurations
        self._load_configurations()
        
        # Initialize default providers if none exist
        if not self.providers:
            self._initialize_default_providers()
        
        logger.info(f"LLMConfigManager initialized with {len(self.providers)} providers")
    
    def _load_configurations(self):
        """Load provider configurations from files"""
        
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                for provider_id, provider_data in config_data.items():
                    try:
                        # Convert provider_type string to enum
                        if isinstance(provider_data.get('provider_type'), str):
                            provider_data['provider_type'] = ProviderType(provider_data['provider_type'])
                        
                        self.providers[provider_id] = ProviderConfig(**provider_data)
                        
                    except Exception as e:
                        logger.warning(f"Error loading provider config {provider_id}: {e}")
            
            # Load API keys from environment variables and secrets file
            self._load_api_keys()
            
            logger.info(f"Loaded {len(self.providers)} provider configurations")
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
    
    def _load_api_keys(self):
        """Load API keys from environment variables and secrets file"""
        
        # Environment variable mappings
        env_key_mappings = {
            "openai": ["OPENAI_API_KEY", "OPENAI_KEY"],
            "grok": ["GROK_API_KEY", "XAI_API_KEY"],
            "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY", "OR_API_KEY"]
        }
        
        # Load from environment variables
        for provider_id, provider_config in self.providers.items():
            provider_type = provider_config.provider_type.value
            
            if provider_type in env_key_mappings:
                for env_var in env_key_mappings[provider_type]:
                    api_key = os.getenv(env_var)
                    if api_key:
                        provider_config.api_key = api_key
                        logger.debug(f"Loaded API key for {provider_id} from {env_var}")
                        break
        
        # Load from secrets file if it exists
        try:
            if self.secrets_file.exists():
                with open(self.secrets_file, 'r') as f:
                    secrets = json.load(f)
                
                for provider_id, provider_config in self.providers.items():
                    if provider_id in secrets and secrets[provider_id].get('api_key'):
                        provider_config.api_key = secrets[provider_id]['api_key']
                        logger.debug(f"Loaded API key for {provider_id} from secrets file")
        
        except Exception as e:
            logger.warning(f"Error loading secrets file: {e}")
    
    def _initialize_default_providers(self):
        """Initialize default provider configurations"""
        
        default_providers = {
            "openai": ProviderConfig(
                provider_type=ProviderType.OPENAI,
                provider_name="OpenAI",
                api_base="https://api.openai.com/v1",
                available_models=[
                    "gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview",
                    "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
                ],
                default_model="gpt-4",
                priority=1
            ),
            
            "grok": ProviderConfig(
                provider_type=ProviderType.GROK,
                provider_name="Grok (xAI)",
                api_base="https://api.x.ai/v1",
                available_models=["grok-beta", "grok-vision-beta"],
                default_model="grok-beta",
                priority=2
            ),
            
            "google": ProviderConfig(
                provider_type=ProviderType.GOOGLE,
                provider_name="Google Gemini",
                api_base="https://generativelanguage.googleapis.com/v1beta",
                available_models=[
                    "gemini-1.5-pro", "gemini-1.5-flash", 
                    "gemini-pro", "gemini-pro-vision"
                ],
                default_model="gemini-1.5-pro",
                priority=2
            ),
            
            "openrouter": ProviderConfig(
                provider_type=ProviderType.OPENROUTER,
                provider_name="OpenRouter",
                api_base="https://openrouter.ai/api/v1",
                available_models=[
                    "anthropic/claude-3-opus",
                    "anthropic/claude-3-sonnet", 
                    "meta-llama/llama-2-70b-chat",
                    "mistralai/mixtral-8x7b-instruct"
                ],
                default_model="anthropic/claude-3-sonnet",
                priority=3
            ),
            
            "ollama": ProviderConfig(
                provider_type=ProviderType.OLLAMA,
                provider_name="Ollama (Local)",
                api_base="http://localhost:11434/v1",
                available_models=[
                    "llama2", "llama2:13b", "llama2:70b",
                    "codellama", "mistral", "neural-chat"
                ],
                default_model="llama2",
                priority=4,
                docker_config={
                    "image": "ollama/ollama",
                    "port": 11434,
                    "volumes": ["/var/lib/ollama:/root/.ollama"],
                    "environment": {
                        "OLLAMA_HOST": "0.0.0.0",
                        "OLLAMA_ORIGINS": "*"
                    }
                }
            )
        }
        
        self.providers = default_providers
        self._save_configurations()
        
        logger.info("Initialized default provider configurations")
    
    def _save_configurations(self):
        """Save provider configurations to file"""
        
        try:
            config_data = {}
            
            for provider_id, provider_config in self.providers.items():
                config_dict = provider_config.dict()
                # Convert enum to string for JSON serialization
                config_dict['provider_type'] = provider_config.provider_type.value
                
                # Don't save API keys to main config file
                config_dict.pop('api_key', None)
                
                config_data[provider_id] = config_dict
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.debug("Provider configurations saved")
            
        except Exception as e:
            logger.error(f"Error saving configurations: {e}")
    
    def add_provider(
        self,
        provider_id: str,
        provider_type: ProviderType,
        provider_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Add a new provider configuration
        
        Args:
            provider_id: Unique provider identifier
            provider_type: Type of provider
            provider_name: Human-readable name
            api_key: API key for authentication
            **kwargs: Additional configuration parameters
        
        Returns:
            True if provider added successfully
        """
        
        try:
            if provider_id in self.providers:
                logger.warning(f"Provider {provider_id} already exists")
                return False
            
            config = ProviderConfig(
                provider_type=provider_type,
                provider_name=provider_name,
                api_key=api_key,
                **kwargs
            )
            
            self.providers[provider_id] = config
            self._save_configurations()
            
            # Save API key to secrets file if provided
            if api_key:
                self._save_api_key(provider_id, api_key)
            
            logger.info(f"Added provider: {provider_name} ({provider_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding provider {provider_id}: {e}")
            return False
    
    def update_provider(
        self,
        provider_id: str,
        **updates
    ) -> bool:
        """
        Update a provider configuration
        
        Args:
            provider_id: Provider to update
            **updates: Configuration updates
        
        Returns:
            True if update successful
        """
        
        try:
            if provider_id not in self.providers:
                logger.warning(f"Provider {provider_id} not found")
                return False
            
            provider = self.providers[provider_id]
            
            # Handle API key separately
            if 'api_key' in updates:
                api_key = updates.pop('api_key')
                provider.api_key = api_key
                if api_key:
                    self._save_api_key(provider_id, api_key)
            
            # Update other fields
            for field, value in updates.items():
                if hasattr(provider, field):
                    setattr(provider, field, value)
            
            self._save_configurations()
            
            logger.info(f"Updated provider: {provider_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating provider {provider_id}: {e}")
            return False
    
    def remove_provider(self, provider_id: str) -> bool:
        """
        Remove a provider configuration
        
        Args:
            provider_id: Provider to remove
        
        Returns:
            True if removal successful
        """
        
        try:
            if provider_id not in self.providers:
                return False
            
            del self.providers[provider_id]
            self._save_configurations()
            
            logger.info(f"Removed provider: {provider_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing provider {provider_id}: {e}")
            return False
    
    def get_provider(self, provider_id: str) -> Optional[ProviderConfig]:
        """Get provider configuration by ID"""
        return self.providers.get(provider_id)
    
    def get_enabled_providers(self) -> Dict[str, ProviderConfig]:
        """Get all enabled providers"""
        return {
            provider_id: config 
            for provider_id, config in self.providers.items() 
            if config.enabled
        }
    
    def get_providers_by_type(self, provider_type: ProviderType) -> Dict[str, ProviderConfig]:
        """Get providers by type"""
        return {
            provider_id: config 
            for provider_id, config in self.providers.items() 
            if config.provider_type == provider_type
        }
    
    def get_best_provider(
        self,
        exclude: List[str] = None,
        require_models: List[str] = None
    ) -> Optional[str]:
        """
        Get the best available provider based on priority and success rate
        
        Args:
            exclude: Provider IDs to exclude
            require_models: Models that must be available
        
        Returns:
            Best provider ID or None
        """
        
        exclude = exclude or []
        require_models = require_models or []
        
        candidates = []
        
        for provider_id, config in self.providers.items():
            # Skip disabled or excluded providers
            if not config.enabled or provider_id in exclude:
                continue
            
            # Skip if required models not available
            if require_models:
                if not all(model in config.available_models for model in require_models):
                    continue
            
            # Skip if no API key (except Ollama)
            if not config.api_key and config.provider_type != ProviderType.OLLAMA:
                continue
            
            # Calculate score based on priority and success rate
            success_rate = config.get_success_rate()
            priority_score = (6 - config.priority) / 5  # Convert to 0-1 scale
            combined_score = (success_rate * 0.7) + (priority_score * 0.3)
            
            candidates.append((provider_id, combined_score))
        
        if not candidates:
            return None
        
        # Return provider with highest score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _save_api_key(self, provider_id: str, api_key: str):
        """Save API key to secrets file"""
        
        try:
            secrets = {}
            
            if self.secrets_file.exists():
                with open(self.secrets_file, 'r') as f:
                    secrets = json.load(f)
            
            secrets[provider_id] = {"api_key": api_key}
            
            with open(self.secrets_file, 'w') as f:
                json.dump(secrets, f, indent=2)
            
            # Set restrictive permissions on secrets file
            self.secrets_file.chmod(0o600)
            
        except Exception as e:
            logger.error(f"Error saving API key for {provider_id}: {e}")
    
    def test_provider_connection(self, provider_id: str) -> Dict[str, Any]:
        """
        Test connection to a provider
        
        Args:
            provider_id: Provider to test
        
        Returns:
            Test results
        """
        
        provider = self.get_provider(provider_id)
        if not provider:
            return {"success": False, "error": "Provider not found"}
        
        if not provider.enabled:
            return {"success": False, "error": "Provider is disabled"}
        
        # Basic validation
        if provider.provider_type != ProviderType.OLLAMA and not provider.api_key:
            return {"success": False, "error": "API key required"}
        
        # This would typically make an actual API call
        # For now, return success if basic requirements are met
        return {
            "success": True,
            "provider_name": provider.provider_name,
            "provider_type": provider.provider_type.value,
            "available_models": provider.available_models,
            "default_model": provider.default_model
        }
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of all provider configurations"""
        
        summary = {
            "total_providers": len(self.providers),
            "enabled_providers": len(self.get_enabled_providers()),
            "providers_by_type": {},
            "providers": {}
        }
        
        # Count by type
        for provider_type in ProviderType:
            count = len(self.get_providers_by_type(provider_type))
            if count > 0:
                summary["providers_by_type"][provider_type.value] = count
        
        # Provider details
        for provider_id, config in self.providers.items():
            summary["providers"][provider_id] = {
                "name": config.provider_name,
                "type": config.provider_type.value,
                "enabled": config.enabled,
                "has_api_key": bool(config.api_key),
                "priority": config.priority,
                "success_rate": config.get_success_rate(),
                "models": len(config.available_models),
                "default_model": config.default_model
            }
        
        return summary
    
    def export_configuration(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export configuration for backup or sharing
        
        Args:
            include_secrets: Whether to include API keys
        
        Returns:
            Configuration data
        """
        
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "version": "1.0",
            "providers": {}
        }
        
        for provider_id, config in self.providers.items():
            config_data = config.dict()
            config_data['provider_type'] = config.provider_type.value
            
            if not include_secrets:
                config_data.pop('api_key', None)
            
            export_data["providers"][provider_id] = config_data
        
        return export_data
    
    def import_configuration(self, config_data: Dict[str, Any], merge: bool = True) -> bool:
        """
        Import configuration from exported data
        
        Args:
            config_data: Configuration data to import
            merge: Whether to merge with existing config or replace
        
        Returns:
            True if import successful
        """
        
        try:
            if not merge:
                self.providers = {}
            
            providers_data = config_data.get("providers", {})
            
            for provider_id, provider_data in providers_data.items():
                # Convert provider_type string to enum
                if isinstance(provider_data.get('provider_type'), str):
                    provider_data['provider_type'] = ProviderType(provider_data['provider_type'])
                
                self.providers[provider_id] = ProviderConfig(**provider_data)
            
            self._save_configurations()
            
            logger.info(f"Imported {len(providers_data)} provider configurations")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False
