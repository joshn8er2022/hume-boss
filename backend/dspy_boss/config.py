
"""
Configuration management for DSPY Boss system
"""

import json
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic import BaseModel, Field
from loguru import logger

from .models import MCPServerConfig, AgentConfig, PromptSignature


class DSPYBossConfig(BaseModel):
    """Main configuration for DSPY Boss system"""
    
    # System settings
    system_name: str = Field(default="DSPY Boss System")
    version: str = Field(default="1.0.0")
    debug_mode: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # Boss settings
    max_concurrent_tasks: int = Field(default=10)
    task_timeout_default: int = Field(default=300)  # seconds
    health_check_interval: int = Field(default=60)  # seconds
    reflection_interval: int = Field(default=3600)  # seconds
    
    # Agent management
    max_agentic_agents: int = Field(default=5)
    agent_spawn_threshold: int = Field(default=8)  # spawn new agent when workload exceeds this
    agent_idle_timeout: int = Field(default=1800)  # seconds
    
    # Task queue settings
    task_queue_workers: int = Field(default=3)
    task_queue_mode: str = Field(default="infinite")  # "finite" or "infinite"
    task_retry_attempts: int = Field(default=3)
    task_retry_delay: int = Field(default=5)  # seconds
    
    # MCP settings
    mcp_connection_timeout: int = Field(default=30)
    mcp_retry_attempts: int = Field(default=3)
    mcp_health_check_interval: int = Field(default=300)  # seconds
    
    # DSPY settings
    dspy_model: str = Field(default="gpt-3.5-turbo")
    dspy_max_tokens: int = Field(default=1000)
    dspy_temperature: float = Field(default=0.7)
    
    # File paths
    config_dir: Path = Field(default=Path("configs"))
    logs_dir: Path = Field(default=Path("logs"))
    data_dir: Path = Field(default=Path("data"))
    
    # Loaded configurations
    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    prompt_signatures: Dict[str, PromptSignature] = Field(default_factory=dict)


class ConfigLoader:
    """Handles loading configurations from various sources"""
    
    def __init__(self, config_dir: Path = Path("configs")):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def load_mcp_servers(self, filename: str = "mcp_servers.json") -> Dict[str, MCPServerConfig]:
        """Load MCP server configurations from JSON file"""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            logger.warning(f"MCP servers config file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            servers = {}
            for name, config in data.items():
                try:
                    servers[name] = MCPServerConfig(**config)
                    logger.info(f"Loaded MCP server config: {name}")
                except Exception as e:
                    logger.error(f"Error loading MCP server config {name}: {e}")
            
            return servers
            
        except Exception as e:
            logger.error(f"Error loading MCP servers config: {e}")
            return {}
    
    def load_agents(self, filename: str = "agents.yaml") -> Dict[str, AgentConfig]:
        """Load agent configurations from YAML file"""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Agents config file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            agents = {}
            for name, config in data.items():
                try:
                    agents[name] = AgentConfig(**config)
                    logger.info(f"Loaded agent config: {name}")
                except Exception as e:
                    logger.error(f"Error loading agent config {name}: {e}")
            
            return agents
            
        except Exception as e:
            logger.error(f"Error loading agents config: {e}")
            return {}
    
    def load_prompt_signatures(self, filename: str = "prompts.yaml") -> Dict[str, PromptSignature]:
        """Load prompt signatures from YAML file"""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Prompts config file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            prompts = {}
            for name, config in data.items():
                try:
                    prompts[name] = PromptSignature(**config)
                    logger.info(f"Loaded prompt signature: {name}")
                except Exception as e:
                    logger.error(f"Error loading prompt signature {name}: {e}")
            
            return prompts
            
        except Exception as e:
            logger.error(f"Error loading prompt signatures config: {e}")
            return {}
    
    def save_mcp_servers(self, servers: Dict[str, MCPServerConfig], filename: str = "mcp_servers.json"):
        """Save MCP server configurations to JSON file"""
        file_path = self.config_dir / filename
        
        try:
            data = {name: server.model_dump() for name, server in servers.items()}
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved MCP servers config to {file_path}")
        except Exception as e:
            logger.error(f"Error saving MCP servers config: {e}")
    
    def save_agents(self, agents: Dict[str, AgentConfig], filename: str = "agents.yaml"):
        """Save agent configurations to YAML file"""
        file_path = self.config_dir / filename
        
        try:
            data = {name: agent.model_dump() for name, agent in agents.items()}
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            logger.info(f"Saved agents config to {file_path}")
        except Exception as e:
            logger.error(f"Error saving agents config: {e}")
    
    def save_prompt_signatures(self, prompts: Dict[str, PromptSignature], filename: str = "prompts.yaml"):
        """Save prompt signatures to YAML file"""
        file_path = self.config_dir / filename
        
        try:
            data = {name: prompt.model_dump() for name, prompt in prompts.items()}
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            logger.info(f"Saved prompt signatures to {file_path}")
        except Exception as e:
            logger.error(f"Error saving prompt signatures: {e}")
    
    def create_sample_configs(self):
        """Create sample configuration files"""
        
        # Sample MCP servers
        sample_mcp = {
            "close_crm": {
                "name": "Close CRM",
                "url": "https://close-mcp-server.klavis.ai/mcp/",
                "instance_id": "7a69be3d-40df-4bfe-8cdc-ca8d15d6d6f0",
                "api_key": "E33hYGfyM2LN5KRkuswV7G765Ga/yoZVctUevP4mG98=",
                "description": "Close CRM integration for customer management",
                "capabilities": ["crm", "contacts", "deals", "communication"]
            },
            "slack": {
                "name": "Slack",
                "url": "https://slack-mcp-server.klavis.ai/mcp/",
                "instance_id": "d251a37c-8e99-4e1d-90a2-f10456723a52",
                "api_key": "E33hYGfyM2LN5KRkuswV7G765Ga/yoZVctUevP4mG98=",
                "description": "Slack integration for team communication",
                "capabilities": ["messaging", "channels", "notifications"]
            },
            "linkedin": {
                "name": "LinkedIn",
                "url": "https://linkedin-mcp-server.klavis.ai/mcp/",
                "instance_id": "e7ac77f1-b634-4c13-8335-a7dc028fbb18",
                "api_key": "E33hYGfyM2LN5KRkuswV7G765Ga/yoZVctUevP4mG98=",
                "description": "LinkedIn integration for professional networking",
                "capabilities": ["networking", "posts", "connections"]
            },
            "deep_research": {
                "name": "Deep Research",
                "url": "https://firecrawl-deep-research-mcp-server.klavis.ai/mcp/",
                "instance_id": "4a3119d4-7759-4b50-b237-4be89ed60651",
                "api_key": "E33hYGfyM2LN5KRkuswV7G765Ga/yoZVctUevP4mG98=",
                "description": "Deep research capabilities using Firecrawl",
                "capabilities": ["research", "web_scraping", "analysis"]
            },
            "web_crawl": {
                "name": "Web Crawl",
                "url": "https://firecrawl-websearch-mcp-server.klavis.ai/mcp/",
                "instance_id": "20e341b2-68cf-4121-862e-24287741128f",
                "api_key": "E33hYGfyM2LN5KRkuswV7G765Ga/yoZVctUevP4mG98=",
                "description": "Web search and crawling using Firecrawl",
                "capabilities": ["web_search", "crawling", "data_extraction"]
            }
        }
        
        # Sample agents
        sample_agents = {
            "research_specialist": {
                "name": "Research Specialist",
                "type": "human",
                "description": "Human expert in research and analysis",
                "capabilities": ["research", "analysis", "reporting"],
                "contact_method": "slack",
                "contact_details": {"channel": "#research-team", "user": "@researcher"}
            },
            "sales_manager": {
                "name": "Sales Manager",
                "type": "human",
                "description": "Human sales manager for deal management",
                "capabilities": ["sales", "crm", "customer_relations"],
                "contact_method": "close_crm",
                "contact_details": {"user_id": "sales_manager_001"}
            },
            "data_analyst": {
                "name": "Data Analyst Agent",
                "type": "agentic",
                "description": "Autonomous agent for data analysis tasks",
                "capabilities": ["data_analysis", "visualization", "reporting"],
                "model_name": "gpt-4",
                "prompt_signature": "data_analysis_react"
            }
        }
        
        # Sample prompt signatures
        sample_prompts = {
            "data_analysis_react": {
                "name": "Data Analysis React Agent",
                "signature": "question -> analysis",
                "description": "React agent for data analysis tasks",
                "input_fields": ["question", "data"],
                "output_fields": ["analysis", "insights", "recommendations"],
                "is_react_agent": True,
                "react_steps": 5,
                "react_tools": ["python", "pandas", "matplotlib"],
                "examples": [
                    {
                        "question": "Analyze sales trends",
                        "analysis": "Based on the data, sales have increased 15% over the last quarter..."
                    }
                ]
            },
            "task_assignment": {
                "name": "Task Assignment",
                "signature": "task, available_agents -> assignment",
                "description": "Assigns tasks to appropriate agents",
                "input_fields": ["task", "available_agents", "priorities"],
                "output_fields": ["assignment", "reasoning"],
                "examples": [
                    {
                        "task": "Research market trends",
                        "assignment": "Assign to research_specialist due to expertise in market analysis"
                    }
                ]
            }
        }
        
        # Save sample configs
        self.save_mcp_servers({name: MCPServerConfig(**config) for name, config in sample_mcp.items()})
        self.save_agents({name: AgentConfig(**config) for name, config in sample_agents.items()})
        self.save_prompt_signatures({name: PromptSignature(**config) for name, config in sample_prompts.items()})
        
        logger.info("Created sample configuration files")


def load_full_config(config_dir: str = "configs") -> DSPYBossConfig:
    """Load complete configuration including all sub-configurations"""
    
    config_path = Path(config_dir)
    loader = ConfigLoader(config_path)
    
    # Create sample configs if they don't exist
    if not any((config_path / f).exists() for f in ["mcp_servers.json", "agents.yaml", "prompts.yaml"]):
        logger.info("Creating sample configuration files...")
        loader.create_sample_configs()
    
    # Load all configurations
    config = DSPYBossConfig(config_dir=config_path)
    config.mcp_servers = loader.load_mcp_servers()
    config.agents = loader.load_agents()
    config.prompt_signatures = loader.load_prompt_signatures()
    
    logger.info(f"Loaded configuration with {len(config.mcp_servers)} MCP servers, "
               f"{len(config.agents)} agents, and {len(config.prompt_signatures)} prompt signatures")
    
    return config
