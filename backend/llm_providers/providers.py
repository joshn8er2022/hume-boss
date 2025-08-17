
"""
Individual LLM provider implementations
"""

import asyncio
import aiohttp
import docker
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime
from loguru import logger
import json

import dspy

from .config_manager import ProviderConfig, ProviderType


class BaseLLMProvider(ABC):
    """Base class for all LLM providers"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None
        self.is_connected = False
        self.last_error: Optional[str] = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the provider"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the provider"""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using the provider"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to the provider"""
        pass
    
    def get_dspy_lm(self, model: Optional[str] = None, **kwargs) -> dspy.LM:
        """Get DSPY language model instance"""
        raise NotImplementedError("Subclasses must implement get_dspy_lm")
    
    def record_success(self):
        """Record successful API call"""
        self.config.record_success()
        self.last_error = None
    
    def record_error(self, error: str):
        """Record failed API call"""
        self.config.record_error()
        self.last_error = error


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation"""
    
    async def connect(self) -> bool:
        """Connect to OpenAI API"""
        try:
            if not self.config.api_key:
                raise ValueError("OpenAI API key is required")
            
            # Initialize OpenAI client
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base,
                timeout=self.config.timeout_seconds
            )
            
            # Test connection with a simple API call
            models = await self.client.models.list()
            self.is_connected = True
            
            logger.info("Connected to OpenAI successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            self.record_error(str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from OpenAI"""
        self.is_connected = False
        self.client = None
        logger.info("Disconnected from OpenAI")
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using OpenAI"""
        
        try:
            if not self.is_connected or not self.client:
                await self.connect()
            
            model = model or self.config.default_model
            temperature = temperature if temperature is not None else self.config.default_temperature
            max_tokens = max_tokens or self.config.default_max_tokens
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            result = {
                "success": True,
                "text": response.choices[0].message.content,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "provider": "openai"
            }
            
            self.record_success()
            return result
            
        except Exception as e:
            error_msg = f"OpenAI generation failed: {e}"
            logger.error(error_msg)
            self.record_error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "provider": "openai"
            }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test OpenAI connection"""
        
        try:
            result = await self.generate(
                "Hello, this is a test message. Please respond with 'Connection successful.'",
                max_tokens=10
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "OpenAI connection test successful",
                    "response": result["text"][:100]  # First 100 chars
                }
            else:
                return result
                
        except Exception as e:
            return {
                "success": False,
                "error": f"OpenAI connection test failed: {e}"
            }
    
    def get_dspy_lm(self, model: Optional[str] = None, **kwargs) -> dspy.LM:
        """Get DSPY OpenAI language model"""
        
        model = model or self.config.default_model
        
        return dspy.OpenAI(
            model=model,
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            temperature=kwargs.get('temperature', self.config.default_temperature),
            max_tokens=kwargs.get('max_tokens', self.config.default_max_tokens),
            **kwargs
        )


class GrokProvider(BaseLLMProvider):
    """Grok (xAI) provider implementation"""
    
    async def connect(self) -> bool:
        """Connect to Grok API"""
        try:
            if not self.config.api_key:
                raise ValueError("Grok API key is required")
            
            # Grok uses OpenAI-compatible API
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base or "https://api.x.ai/v1",
                timeout=self.config.timeout_seconds
            )
            
            self.is_connected = True
            logger.info("Connected to Grok successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Grok: {e}")
            self.record_error(str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from Grok"""
        self.is_connected = False
        self.client = None
        logger.info("Disconnected from Grok")
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using Grok"""
        
        try:
            if not self.is_connected or not self.client:
                await self.connect()
            
            model = model or self.config.default_model or "grok-beta"
            temperature = temperature if temperature is not None else self.config.default_temperature
            max_tokens = max_tokens or self.config.default_max_tokens
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            result = {
                "success": True,
                "text": response.choices[0].message.content,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                "provider": "grok"
            }
            
            self.record_success()
            return result
            
        except Exception as e:
            error_msg = f"Grok generation failed: {e}"
            logger.error(error_msg)
            self.record_error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "provider": "grok"
            }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Grok connection"""
        
        try:
            result = await self.generate(
                "Hello, this is a test message. Please respond with 'Connection successful.'",
                max_tokens=10
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "Grok connection test successful",
                    "response": result["text"][:100]
                }
            else:
                return result
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Grok connection test failed: {e}"
            }
    
    def get_dspy_lm(self, model: Optional[str] = None, **kwargs) -> dspy.LM:
        """Get DSPY Grok language model"""
        
        model = model or self.config.default_model or "grok-beta"
        
        # Use OpenAI adapter for Grok
        return dspy.OpenAI(
            model=model,
            api_key=self.config.api_key,
            base_url=self.config.api_base or "https://api.x.ai/v1",
            temperature=kwargs.get('temperature', self.config.default_temperature),
            max_tokens=kwargs.get('max_tokens', self.config.default_max_tokens),
            **kwargs
        )


class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider implementation"""
    
    async def connect(self) -> bool:
        """Connect to Google Gemini API"""
        try:
            if not self.config.api_key:
                raise ValueError("Google API key is required")
            
            # Initialize Google Generative AI client
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.api_key)
                
                # Test connection by listing models
                models = genai.list_models()
                self.client = genai
                self.is_connected = True
                
                logger.info("Connected to Google Gemini successfully")
                return True
                
            except ImportError:
                raise ValueError("google-generativeai package is required for Google provider")
            
        except Exception as e:
            logger.error(f"Failed to connect to Google Gemini: {e}")
            self.record_error(str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from Google Gemini"""
        self.is_connected = False
        self.client = None
        logger.info("Disconnected from Google Gemini")
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using Google Gemini"""
        
        try:
            if not self.is_connected or not self.client:
                await self.connect()
            
            model = model or self.config.default_model or "gemini-1.5-pro"
            temperature = temperature if temperature is not None else self.config.default_temperature
            
            # Create model instance
            genai_model = self.client.GenerativeModel(model)
            
            # Configure generation
            generation_config = {
                "temperature": temperature,
            }
            
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            
            # Generate content
            response = genai_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            result = {
                "success": True,
                "text": response.text,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                    "total_tokens": response.usage_metadata.total_token_count if response.usage_metadata else 0
                },
                "provider": "google"
            }
            
            self.record_success()
            return result
            
        except Exception as e:
            error_msg = f"Google Gemini generation failed: {e}"
            logger.error(error_msg)
            self.record_error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "provider": "google"
            }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Google Gemini connection"""
        
        try:
            result = await self.generate(
                "Hello, this is a test message. Please respond with 'Connection successful.'",
                max_tokens=10
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "Google Gemini connection test successful",
                    "response": result["text"][:100]
                }
            else:
                return result
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Google Gemini connection test failed: {e}"
            }
    
    def get_dspy_lm(self, model: Optional[str] = None, **kwargs) -> dspy.LM:
        """Get DSPY Google language model"""
        
        model = model or self.config.default_model or "gemini-1.5-pro"
        
        # For DSPY compatibility, we might need to use a wrapper
        # This is a simplified implementation
        return dspy.GooglePaLM(
            model=model,
            api_key=self.config.api_key,
            temperature=kwargs.get('temperature', self.config.default_temperature),
            **kwargs
        )


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter provider implementation"""
    
    async def connect(self) -> bool:
        """Connect to OpenRouter API"""
        try:
            if not self.config.api_key:
                raise ValueError("OpenRouter API key is required")
            
            # OpenRouter uses OpenAI-compatible API
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_base or "https://openrouter.ai/api/v1",
                timeout=self.config.timeout_seconds
            )
            
            self.is_connected = True
            logger.info("Connected to OpenRouter successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to OpenRouter: {e}")
            self.record_error(str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from OpenRouter"""
        self.is_connected = False
        self.client = None
        logger.info("Disconnected from OpenRouter")
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using OpenRouter"""
        
        try:
            if not self.is_connected or not self.client:
                await self.connect()
            
            model = model or self.config.default_model or "anthropic/claude-3-sonnet"
            temperature = temperature if temperature is not None else self.config.default_temperature
            max_tokens = max_tokens or self.config.default_max_tokens
            
            # Add OpenRouter-specific headers
            extra_headers = {
                "HTTP-Referer": "https://dspy-boss.local",
                "X-Title": "DSPY Boss System"
            }
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=extra_headers,
                **kwargs
            )
            
            result = {
                "success": True,
                "text": response.choices[0].message.content,
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                "provider": "openrouter"
            }
            
            self.record_success()
            return result
            
        except Exception as e:
            error_msg = f"OpenRouter generation failed: {e}"
            logger.error(error_msg)
            self.record_error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "provider": "openrouter"
            }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test OpenRouter connection"""
        
        try:
            result = await self.generate(
                "Hello, this is a test message. Please respond with 'Connection successful.'",
                max_tokens=10
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "OpenRouter connection test successful",
                    "response": result["text"][:100]
                }
            else:
                return result
                
        except Exception as e:
            return {
                "success": False,
                "error": f"OpenRouter connection test failed: {e}"
            }
    
    def get_dspy_lm(self, model: Optional[str] = None, **kwargs) -> dspy.LM:
        """Get DSPY OpenRouter language model"""
        
        model = model or self.config.default_model or "anthropic/claude-3-sonnet"
        
        return dspy.OpenAI(
            model=model,
            api_key=self.config.api_key,
            base_url=self.config.api_base or "https://openrouter.ai/api/v1",
            temperature=kwargs.get('temperature', self.config.default_temperature),
            max_tokens=kwargs.get('max_tokens', self.config.default_max_tokens),
            **kwargs
        )


class OllamaProvider(BaseLLMProvider):
    """Ollama local provider implementation with Docker support"""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.docker_client = None
        self.container = None
        self.container_name = "dspy-boss-ollama"
    
    async def connect(self) -> bool:
        """Connect to Ollama (with Docker setup if needed)"""
        try:
            # First, try to connect to existing Ollama instance
            if await self._test_ollama_connection():
                self.is_connected = True
                logger.info("Connected to existing Ollama instance")
                return True
            
            # If not available, try to start Docker container
            if self.config.docker_config:
                success = await self._start_ollama_docker()
                if success:
                    self.is_connected = True
                    logger.info("Started Ollama Docker container and connected")
                    return True
            
            logger.error("Could not connect to Ollama instance")
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            self.record_error(str(e))
            return False
    
    async def disconnect(self):
        """Disconnect from Ollama"""
        self.is_connected = False
        
        # Optionally stop Docker container
        if self.container and self.config.docker_config:
            try:
                self.container.stop()
                logger.info("Stopped Ollama Docker container")
            except Exception as e:
                logger.warning(f"Error stopping Ollama container: {e}")
        
        logger.info("Disconnected from Ollama")
    
    async def _test_ollama_connection(self) -> bool:
        """Test connection to Ollama API"""
        try:
            api_base = self.config.api_base or "http://localhost:11434"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{api_base}/api/version") as response:
                    if response.status == 200:
                        return True
            
            return False
            
        except Exception:
            return False
    
    async def _start_ollama_docker(self) -> bool:
        """Start Ollama using Docker"""
        try:
            import docker
            
            self.docker_client = docker.from_env()
            docker_config = self.config.docker_config
            
            # Check if container already exists
            try:
                existing_container = self.docker_client.containers.get(self.container_name)
                if existing_container.status != "running":
                    existing_container.start()
                self.container = existing_container
                
                # Wait for service to be ready
                await asyncio.sleep(5)
                return await self._test_ollama_connection()
                
            except docker.errors.NotFound:
                pass
            
            # Create new container
            self.container = self.docker_client.containers.run(
                docker_config["image"],
                name=self.container_name,
                ports={f"{docker_config['port']}/tcp": docker_config["port"]},
                volumes=docker_config.get("volumes", []),
                environment=docker_config.get("environment", {}),
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            # Wait for service to be ready
            logger.info("Waiting for Ollama container to be ready...")
            for _ in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                if await self._test_ollama_connection():
                    return True
            
            return False
            
        except ImportError:
            logger.error("docker package is required for Ollama Docker integration")
            return False
        except Exception as e:
            logger.error(f"Error starting Ollama Docker container: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using Ollama"""
        
        try:
            if not self.is_connected:
                await self.connect()
            
            model = model or self.config.default_model or "llama2"
            temperature = temperature if temperature is not None else self.config.default_temperature
            api_base = self.config.api_base or "http://localhost:11434"
            
            # Ollama API request
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False
            }
            
            if max_tokens:
                payload["options"] = {"num_predict": max_tokens}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{api_base}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    
                    if response.status != 200:
                        raise Exception(f"Ollama API error: {response.status}")
                    
                    data = await response.json()
                    
                    result = {
                        "success": True,
                        "text": data.get("response", ""),
                        "model": model,
                        "usage": {
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                        },
                        "provider": "ollama"
                    }
                    
                    self.record_success()
                    return result
            
        except Exception as e:
            error_msg = f"Ollama generation failed: {e}"
            logger.error(error_msg)
            self.record_error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "provider": "ollama"
            }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Ollama connection"""
        
        try:
            if not await self._test_ollama_connection():
                return {
                    "success": False,
                    "error": "Cannot connect to Ollama API"
                }
            
            result = await self.generate(
                "Hello, this is a test message. Please respond with 'Connection successful.'",
                max_tokens=20
            )
            
            if result["success"]:
                return {
                    "success": True,
                    "message": "Ollama connection test successful",
                    "response": result["text"][:100]
                }
            else:
                return result
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Ollama connection test failed: {e}"
            }
    
    def get_dspy_lm(self, model: Optional[str] = None, **kwargs) -> dspy.LM:
        """Get DSPY Ollama language model"""
        
        model = model or self.config.default_model or "llama2"
        api_base = self.config.api_base or "http://localhost:11434"
        
        # Use Ollama adapter for DSPY
        return dspy.Ollama(
            model=model,
            base_url=api_base,
            temperature=kwargs.get('temperature', self.config.default_temperature),
            **kwargs
        )
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model in Ollama"""
        
        try:
            api_base = self.config.api_base or "http://localhost:11434"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{api_base}/api/pull",
                    json={"name": model_name},
                    timeout=aiohttp.ClientTimeout(total=600)  # 10 minute timeout
                ) as response:
                    
                    if response.status == 200:
                        return {
                            "success": True,
                            "message": f"Model {model_name} pulled successfully"
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Failed to pull model: {response.status}"
                        }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error pulling model: {e}"
            }
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models in Ollama"""
        
        try:
            api_base = self.config.api_base or "http://localhost:11434"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{api_base}/api/tags") as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        models = [model["name"] for model in data.get("models", [])]
                        
                        return {
                            "success": True,
                            "models": models
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Failed to list models: {response.status}"
                        }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error listing models: {e}"
            }
