
"""
LLM Provider Manager - Central coordinator for all LLM providers
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Type
from enum import Enum
from loguru import logger

import dspy

from .config_manager import LLMConfigManager, ProviderConfig, ProviderType
from .providers import (
    BaseLLMProvider, OpenAIProvider, GrokProvider, 
    GoogleProvider, OpenRouterProvider, OllamaProvider
)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for multiple providers"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_RANDOM = "weighted_random"
    LEAST_LOADED = "least_loaded"
    FASTEST_FIRST = "fastest_first"
    HIGHEST_SUCCESS_RATE = "highest_success_rate"


class LLMProviderManager:
    """
    Central manager for all LLM providers
    Handles provider lifecycle, load balancing, failover, and DSPY integration
    """
    
    def __init__(self, config_dir: str = "llm_config"):
        self.config_manager = LLMConfigManager(config_dir)
        
        # Provider instances
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.provider_classes: Dict[ProviderType, Type[BaseLLMProvider]] = {
            ProviderType.OPENAI: OpenAIProvider,
            ProviderType.GROK: GrokProvider,
            ProviderType.GOOGLE: GoogleProvider,
            ProviderType.OPENROUTER: OpenRouterProvider,
            ProviderType.OLLAMA: OllamaProvider
        }
        
        # System state
        self.is_initialized = False
        self.active_providers: List[str] = []
        self.failed_providers: List[str] = []
        
        # Load balancing
        self.load_balancing_strategy = LoadBalancingStrategy.HIGHEST_SUCCESS_RATE
        self.round_robin_index = 0
        
        # Performance tracking
        self.provider_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "requests_by_provider": {},
            "errors_by_provider": {}
        }
        
        # Failover settings
        self.enable_failover = True
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.health_check_interval = 300  # 5 minutes
        
        logger.info("LLMProviderManager initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize all configured providers
        
        Returns:
            True if at least one provider was initialized successfully
        """
        
        if self.is_initialized:
            return True
        
        try:
            logger.info("Initializing LLM providers")
            
            # Initialize providers from configuration
            enabled_providers = self.config_manager.get_enabled_providers()
            
            if not enabled_providers:
                logger.warning("No enabled providers found in configuration")
                return False
            
            initialization_results = []
            
            for provider_id, config in enabled_providers.items():
                try:
                    success = await self._initialize_provider(provider_id, config)
                    initialization_results.append((provider_id, success))
                    
                    if success:
                        self.active_providers.append(provider_id)
                    else:
                        self.failed_providers.append(provider_id)
                        
                except Exception as e:
                    logger.error(f"Error initializing provider {provider_id}: {e}")
                    self.failed_providers.append(provider_id)
                    initialization_results.append((provider_id, False))
            
            # Check if at least one provider was successful
            successful_count = sum(1 for _, success in initialization_results if success)
            
            if successful_count == 0:
                logger.error("No providers could be initialized successfully")
                return False
            
            # Start health check background task
            await self._start_health_monitoring()
            
            self.is_initialized = True
            
            logger.info(f"LLM Provider initialization completed: {successful_count}/{len(enabled_providers)} providers active")
            
            # Log provider status
            for provider_id, success in initialization_results:
                status = "✓" if success else "✗"
                logger.info(f"  {status} {provider_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM providers: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown all providers and cleanup"""
        
        logger.info("Shutting down LLM providers")
        
        # Stop health monitoring
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all providers
        for provider_id, provider in self.providers.items():
            try:
                await provider.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting provider {provider_id}: {e}")
        
        self.providers.clear()
        self.active_providers.clear()
        self.failed_providers.clear()
        self.is_initialized = False
        
        logger.info("LLM providers shutdown completed")
    
    async def _initialize_provider(self, provider_id: str, config: ProviderConfig) -> bool:
        """Initialize a single provider"""
        
        try:
            provider_class = self.provider_classes.get(config.provider_type)
            if not provider_class:
                logger.error(f"Unknown provider type: {config.provider_type}")
                return False
            
            # Create provider instance
            provider = provider_class(config)
            
            # Attempt connection
            connected = await provider.connect()
            
            if connected:
                self.providers[provider_id] = provider
                self.provider_metrics["requests_by_provider"][provider_id] = 0
                self.provider_metrics["errors_by_provider"][provider_id] = 0
                
                logger.info(f"Provider {provider_id} ({config.provider_name}) initialized successfully")
                return True
            else:
                logger.warning(f"Provider {provider_id} failed to connect")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing provider {provider_id}: {e}")
            return False
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        preferred_provider: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using the best available provider with failover
        
        Args:
            prompt: Text prompt to generate from
            model: Specific model to use
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            preferred_provider: Preferred provider to try first
            **kwargs: Additional generation parameters
        
        Returns:
            Generation result
        """
        
        if not self.is_initialized:
            return {
                "success": False,
                "error": "LLM providers not initialized"
            }
        
        if not self.active_providers:
            return {
                "success": False,
                "error": "No active providers available"
            }
        
        start_time = datetime.utcnow()
        
        try:
            # Determine provider order to try
            provider_order = self._get_provider_order(preferred_provider)
            
            last_error = None
            attempts = 0
            
            for provider_id in provider_order:
                if attempts >= self.max_retries:
                    break
                
                attempts += 1
                
                try:
                    provider = self.providers.get(provider_id)
                    if not provider or provider_id not in self.active_providers:
                        continue
                    
                    logger.debug(f"Attempting generation with provider {provider_id}")
                    
                    result = await provider.generate(
                        prompt=prompt,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                    
                    if result.get("success"):
                        # Update metrics
                        self.provider_metrics["total_requests"] += 1
                        self.provider_metrics["successful_requests"] += 1
                        self.provider_metrics["requests_by_provider"][provider_id] += 1
                        
                        response_time = (datetime.utcnow() - start_time).total_seconds()
                        self._update_average_response_time(response_time)
                        
                        result["provider_used"] = provider_id
                        result["response_time"] = response_time
                        result["attempts"] = attempts
                        
                        logger.debug(f"Generation successful with {provider_id} in {response_time:.2f}s")
                        return result
                    else:
                        last_error = result.get("error", "Unknown error")
                        self.provider_metrics["errors_by_provider"][provider_id] += 1
                        
                        # Mark provider as temporarily failed if too many errors
                        await self._handle_provider_error(provider_id)
                
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Provider {provider_id} generated exception: {e}")
                    
                    if provider_id in self.providers:
                        self.provider_metrics["errors_by_provider"][provider_id] += 1
                        await self._handle_provider_error(provider_id)
                
                # Wait before retry
                if attempts < self.max_retries and self.retry_delay > 0:
                    await asyncio.sleep(self.retry_delay)
            
            # All providers failed
            self.provider_metrics["total_requests"] += 1
            self.provider_metrics["failed_requests"] += 1
            
            return {
                "success": False,
                "error": f"All providers failed. Last error: {last_error}",
                "attempts": attempts,
                "providers_tried": provider_order[:attempts]
            }
            
        except Exception as e:
            logger.error(f"Error in generate_text: {e}")
            
            self.provider_metrics["total_requests"] += 1
            self.provider_metrics["failed_requests"] += 1
            
            return {
                "success": False,
                "error": f"Generation failed: {str(e)}"
            }
    
    def get_dspy_lm(
        self,
        provider_id: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Optional[dspy.LM]:
        """
        Get a DSPY language model from the best available provider
        
        Args:
            provider_id: Specific provider to use
            model: Model to use
            **kwargs: Additional model parameters
        
        Returns:
            DSPY language model or None if not available
        """
        
        try:
            if not self.is_initialized or not self.active_providers:
                logger.error("No active providers available for DSPY LM")
                return None
            
            # Use specific provider if requested
            if provider_id:
                if provider_id in self.providers and provider_id in self.active_providers:
                    provider = self.providers[provider_id]
                    return provider.get_dspy_lm(model=model, **kwargs)
                else:
                    logger.warning(f"Requested provider {provider_id} not available")
                    return None
            
            # Use best available provider
            best_provider_id = self._get_best_provider()
            if not best_provider_id:
                return None
            
            provider = self.providers[best_provider_id]
            return provider.get_dspy_lm(model=model, **kwargs)
            
        except Exception as e:
            logger.error(f"Error getting DSPY LM: {e}")
            return None
    
    def _get_provider_order(self, preferred_provider: Optional[str] = None) -> List[str]:
        """Get ordered list of providers to try based on load balancing strategy"""
        
        available_providers = [p for p in self.active_providers if p in self.providers]
        
        if not available_providers:
            return []
        
        # Put preferred provider first if specified and available
        if preferred_provider and preferred_provider in available_providers:
            provider_order = [preferred_provider]
            remaining = [p for p in available_providers if p != preferred_provider]
        else:
            provider_order = []
            remaining = available_providers.copy()
        
        # Apply load balancing strategy to remaining providers
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Simple round-robin
            if remaining:
                index = self.round_robin_index % len(remaining)
                provider_order.extend(remaining[index:] + remaining[:index])
                self.round_robin_index += 1
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.HIGHEST_SUCCESS_RATE:
            # Sort by success rate
            def get_success_rate(provider_id):
                provider = self.providers[provider_id]
                return provider.config.get_success_rate()
            
            remaining.sort(key=get_success_rate, reverse=True)
            provider_order.extend(remaining)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED:
            # Sort by request count (ascending)
            def get_request_count(provider_id):
                return self.provider_metrics["requests_by_provider"].get(provider_id, 0)
            
            remaining.sort(key=get_request_count)
            provider_order.extend(remaining)
        
        else:
            # Default: use current order
            provider_order.extend(remaining)
        
        return provider_order
    
    def _get_best_provider(self) -> Optional[str]:
        """Get the best provider for DSPY LM"""
        
        provider_order = self._get_provider_order()
        return provider_order[0] if provider_order else None
    
    def _update_average_response_time(self, response_time: float):
        """Update average response time metric"""
        
        current_avg = self.provider_metrics["average_response_time"]
        total_requests = self.provider_metrics["successful_requests"]
        
        if total_requests == 1:
            self.provider_metrics["average_response_time"] = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.provider_metrics["average_response_time"] = (
                alpha * response_time + (1 - alpha) * current_avg
            )
    
    async def _handle_provider_error(self, provider_id: str):
        """Handle provider error and potentially mark as failed"""
        
        if provider_id not in self.providers:
            return
        
        provider = self.providers[provider_id]
        error_count = self.provider_metrics["errors_by_provider"].get(provider_id, 0)
        
        # If too many consecutive errors, temporarily disable provider
        if error_count >= 5 and provider_id in self.active_providers:
            self.active_providers.remove(provider_id)
            self.failed_providers.append(provider_id)
            
            logger.warning(f"Provider {provider_id} temporarily disabled due to errors")
    
    async def _start_health_monitoring(self):
        """Start background health monitoring task"""
        
        self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info("Started LLM provider health monitoring")
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        
        try:
            while True:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
                
        except asyncio.CancelledError:
            logger.info("Health monitoring cancelled")
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all providers"""
        
        logger.debug("Performing provider health checks")
        
        # Check active providers
        for provider_id in self.active_providers.copy():
            try:
                provider = self.providers[provider_id]
                health_result = await provider.test_connection()
                
                if not health_result.get("success"):
                    logger.warning(f"Provider {provider_id} health check failed: {health_result.get('error')}")
                    self.active_providers.remove(provider_id)
                    if provider_id not in self.failed_providers:
                        self.failed_providers.append(provider_id)
                
            except Exception as e:
                logger.warning(f"Health check error for {provider_id}: {e}")
        
        # Try to recover failed providers
        for provider_id in self.failed_providers.copy():
            try:
                provider = self.providers[provider_id]
                health_result = await provider.test_connection()
                
                if health_result.get("success"):
                    logger.info(f"Provider {provider_id} recovered and reactivated")
                    self.failed_providers.remove(provider_id)
                    self.active_providers.append(provider_id)
                    
                    # Reset error count
                    self.provider_metrics["errors_by_provider"][provider_id] = 0
                
            except Exception as e:
                logger.debug(f"Recovery check failed for {provider_id}: {e}")
    
    # Provider management methods
    
    async def add_provider(
        self,
        provider_id: str,
        provider_type: ProviderType,
        provider_name: str,
        api_key: Optional[str] = None,
        **config_kwargs
    ) -> bool:
        """
        Add a new provider at runtime
        
        Args:
            provider_id: Unique provider identifier
            provider_type: Type of provider
            provider_name: Human-readable name
            api_key: API key
            **config_kwargs: Additional configuration
        
        Returns:
            True if provider added successfully
        """
        
        try:
            # Add to configuration
            success = self.config_manager.add_provider(
                provider_id=provider_id,
                provider_type=provider_type,
                provider_name=provider_name,
                api_key=api_key,
                **config_kwargs
            )
            
            if not success:
                return False
            
            # Initialize the provider if system is running
            if self.is_initialized:
                config = self.config_manager.get_provider(provider_id)
                if config and config.enabled:
                    success = await self._initialize_provider(provider_id, config)
                    if success:
                        self.active_providers.append(provider_id)
                    else:
                        self.failed_providers.append(provider_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding provider {provider_id}: {e}")
            return False
    
    async def remove_provider(self, provider_id: str) -> bool:
        """
        Remove a provider
        
        Args:
            provider_id: Provider to remove
        
        Returns:
            True if removed successfully
        """
        
        try:
            # Disconnect provider if active
            if provider_id in self.providers:
                await self.providers[provider_id].disconnect()
                del self.providers[provider_id]
            
            # Remove from active/failed lists
            if provider_id in self.active_providers:
                self.active_providers.remove(provider_id)
            if provider_id in self.failed_providers:
                self.failed_providers.remove(provider_id)
            
            # Remove from configuration
            self.config_manager.remove_provider(provider_id)
            
            # Clean up metrics
            self.provider_metrics["requests_by_provider"].pop(provider_id, None)
            self.provider_metrics["errors_by_provider"].pop(provider_id, None)
            
            logger.info(f"Provider {provider_id} removed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error removing provider {provider_id}: {e}")
            return False
    
    async def test_provider(self, provider_id: str) -> Dict[str, Any]:
        """
        Test a specific provider
        
        Args:
            provider_id: Provider to test
        
        Returns:
            Test results
        """
        
        try:
            if provider_id not in self.providers:
                return {
                    "success": False,
                    "error": f"Provider {provider_id} not found"
                }
            
            provider = self.providers[provider_id]
            result = await provider.test_connection()
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Test failed: {str(e)}"
            }
    
    def update_api_key(self, provider_id: str, api_key: str) -> bool:
        """
        Update API key for a provider
        
        Args:
            provider_id: Provider to update
            api_key: New API key
        
        Returns:
            True if updated successfully
        """
        
        try:
            success = self.config_manager.update_provider(provider_id, api_key=api_key)
            
            if success and provider_id in self.providers:
                # Update the provider instance
                self.providers[provider_id].config.api_key = api_key
                logger.info(f"API key updated for provider {provider_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating API key for {provider_id}: {e}")
            return False
    
    def set_load_balancing_strategy(self, strategy: LoadBalancingStrategy):
        """Set load balancing strategy"""
        
        self.load_balancing_strategy = strategy
        logger.info(f"Load balancing strategy updated to: {strategy.value}")
    
    def enable_failover(self, enabled: bool = True):
        """Enable or disable failover"""
        
        self.enable_failover = enabled
        logger.info(f"Failover {'enabled' if enabled else 'disabled'}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "initialized": self.is_initialized,
            "active_providers": len(self.active_providers),
            "failed_providers": len(self.failed_providers),
            "total_providers": len(self.providers),
            "load_balancing_strategy": self.load_balancing_strategy.value,
            "failover_enabled": self.enable_failover,
            "metrics": self.provider_metrics.copy(),
            "provider_status": {
                provider_id: {
                    "active": provider_id in self.active_providers,
                    "provider_name": provider.config.provider_name,
                    "provider_type": provider.config.provider_type.value,
                    "success_rate": provider.config.get_success_rate(),
                    "last_used": provider.config.last_used.isoformat() if provider.config.last_used else None,
                    "error_count": provider.config.error_count
                }
                for provider_id, provider in self.providers.items()
            }
        }
    
    def get_provider_details(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific provider"""
        
        if provider_id not in self.providers:
            return None
        
        provider = self.providers[provider_id]
        config = provider.config
        
        return {
            "provider_id": provider_id,
            "provider_name": config.provider_name,
            "provider_type": config.provider_type.value,
            "enabled": config.enabled,
            "active": provider_id in self.active_providers,
            "has_api_key": bool(config.api_key),
            "available_models": config.available_models,
            "default_model": config.default_model,
            "priority": config.priority,
            "success_rate": config.get_success_rate(),
            "success_count": config.success_count,
            "error_count": config.error_count,
            "last_used": config.last_used.isoformat() if config.last_used else None,
            "last_error": provider.last_error,
            "is_connected": provider.is_connected,
            "configuration": {
                "api_base": config.api_base,
                "timeout_seconds": config.timeout_seconds,
                "max_retries": config.max_retries,
                "default_temperature": config.default_temperature,
                "default_max_tokens": config.default_max_tokens
            },
            "metrics": {
                "requests": self.provider_metrics["requests_by_provider"].get(provider_id, 0),
                "errors": self.provider_metrics["errors_by_provider"].get(provider_id, 0)
            }
        }
    
    async def get_available_models(self, provider_id: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get available models from providers
        
        Args:
            provider_id: Specific provider (None for all)
        
        Returns:
            Dictionary of provider_id -> model list
        """
        
        result = {}
        
        if provider_id:
            # Get models from specific provider
            if provider_id in self.providers:
                provider = self.providers[provider_id]
                
                # For Ollama, try to get live model list
                if isinstance(provider, OllamaProvider):
                    try:
                        models_result = await provider.list_models()
                        if models_result.get("success"):
                            result[provider_id] = models_result["models"]
                        else:
                            result[provider_id] = provider.config.available_models
                    except:
                        result[provider_id] = provider.config.available_models
                else:
                    result[provider_id] = provider.config.available_models
        else:
            # Get models from all providers
            for provider_id, provider in self.providers.items():
                if isinstance(provider, OllamaProvider):
                    try:
                        models_result = await provider.list_models()
                        if models_result.get("success"):
                            result[provider_id] = models_result["models"]
                        else:
                            result[provider_id] = provider.config.available_models
                    except:
                        result[provider_id] = provider.config.available_models
                else:
                    result[provider_id] = provider.config.available_models
        
        return result
