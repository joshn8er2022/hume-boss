
"""
Integration layer between the new autonomous system and the existing Next.js frontend
"""

import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger

# Core system imports
from .state.state_holder import StateHolder
from .state.historical_manager import HistoricalStateManager
from .state.forecasting_engine import StateForecaster
from .autonomous.execution_cycle import ExecutionCycle, CycleConfig
from .autonomous.autonomous_executor import AutonomousExecutor, ExecutionConfig
from .agents.agent_manager import AgentManager, TaskDistributionStrategy
from .agents.agent_hierarchy import AgentRole
from .agents.agent_spawner import SpawningStrategy
from .llm_providers.provider_manager import LLMProviderManager, LoadBalancingStrategy


class DSPYBossIntegration:
    """
    Main integration class that coordinates all backend systems
    Provides API for the Next.js frontend
    """
    
    def __init__(self):
        # Core components
        self.state_holder = StateHolder()
        self.historical_manager = HistoricalStateManager()
        self.forecaster = StateForecaster(self.historical_manager)
        
        # LLM providers
        self.llm_provider_manager = LLMProviderManager()
        
        # Agent system
        self.agent_manager = AgentManager(
            state_holder=self.state_holder,
            spawning_strategy=SpawningStrategy.BALANCED,
            task_distribution_strategy=TaskDistributionStrategy.INTELLIGENT
        )
        
        # Autonomous execution system
        self.execution_cycle = ExecutionCycle(
            state_holder=self.state_holder,
            historical_manager=self.historical_manager,
            forecaster=self.forecaster,
            cycle_config=CycleConfig(),
            execution_config=ExecutionConfig()
        )
        
        # System state
        self.is_initialized = False
        self.is_running = False
        
        logger.info("DSPY Boss Integration initialized")
    
    async def initialize_system(self) -> Dict[str, Any]:
        """
        Initialize the complete autonomous system
        
        Returns:
            Initialization results
        """
        
        try:
            logger.info("Initializing DSPY Boss autonomous system")
            
            # Initialize LLM providers
            llm_success = await self.llm_provider_manager.initialize()
            if not llm_success:
                logger.warning("LLM providers failed to initialize - continuing with limited functionality")
            
            # Initialize agent management system
            agent_success = await self.agent_manager.initialize_system()
            if not agent_success:
                return {
                    "success": False,
                    "error": "Failed to initialize agent management system"
                }
            
            # Initialize execution cycle
            cycle_success = await self.execution_cycle.start_complete_system()
            if not cycle_success:
                return {
                    "success": False,
                    "error": "Failed to start execution cycle"
                }
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("DSPY Boss autonomous system initialized successfully")
            
            return {
                "success": True,
                "components": {
                    "llm_providers": llm_success,
                    "agent_system": agent_success,
                    "execution_cycle": cycle_success
                },
                "status": "System fully operational and autonomous"
            }
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def shutdown_system(self) -> Dict[str, Any]:
        """
        Shutdown the autonomous system
        
        Returns:
            Shutdown results
        """
        
        try:
            logger.info("Shutting down DSPY Boss autonomous system")
            
            # Stop execution cycle
            await self.execution_cycle.stop_complete_system()
            
            # Stop agent management
            await self.agent_manager.stop_autonomous_management()
            
            # Shutdown LLM providers
            await self.llm_provider_manager.shutdown()
            
            self.is_running = False
            self.is_initialized = False
            
            logger.info("System shutdown completed")
            
            return {"success": True, "message": "System shutdown completed"}
            
        except Exception as e:
            logger.error(f"System shutdown error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview for dashboard"""
        
        try:
            current_state = await self.state_holder.get_current_state()
            health_summary = await self.state_holder.get_system_health_summary()
            
            # Get component statuses
            llm_status = self.llm_provider_manager.get_system_status()
            agent_status = self.agent_manager.get_system_status()
            execution_status = await self.execution_cycle.get_complete_system_status()
            
            return {
                "system_info": {
                    "initialized": self.is_initialized,
                    "running": self.is_running,
                    "current_iteration": current_state.iteration_number if current_state else 0,
                    "boss_state": current_state.boss_state if current_state else "UNKNOWN",
                    "system_phase": current_state.system_phase if current_state else "UNKNOWN"
                },
                "health": health_summary,
                "components": {
                    "llm_providers": llm_status,
                    "agents": agent_status,
                    "execution": execution_status
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system overview: {e}")
            return {"error": str(e)}
    
    async def get_agent_hierarchy_display(self) -> Dict[str, Any]:
        """Get agent hierarchy formatted for UI display"""
        
        try:
            hierarchy_summary = self.agent_manager.agent_hierarchy.get_hierarchy_summary()
            performance_summary = self.agent_manager.agent_hierarchy.get_agent_performance_summary()
            
            return {
                "hierarchy": hierarchy_summary,
                "performance": performance_summary
            }
            
        except Exception as e:
            logger.error(f"Error getting agent hierarchy: {e}")
            return {"error": str(e)}
    
    async def spawn_agent(
        self,
        role: str = "generalist",
        capabilities: List[str] = None,
        specialization: str = None
    ) -> Dict[str, Any]:
        """Spawn a new agent"""
        
        try:
            agent_role = AgentRole(role) if role in [r.value for r in AgentRole] else AgentRole.GENERALIST
            
            result = await self.agent_manager.force_spawn_agent(
                role=agent_role,
                capabilities=capabilities,
                specialization=specialization
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error spawning agent: {e}")
            return {"success": False, "error": str(e)}
    
    async def delegate_task(
        self,
        task_id: str,
        task_description: str,
        requirements: Dict[str, Any] = None,
        priority: int = 3
    ) -> Dict[str, Any]:
        """Delegate a task to an agent"""
        
        try:
            task_requirements = {
                "description": task_description,
                "capabilities": requirements.get("capabilities", []) if requirements else [],
                "estimated_duration": requirements.get("estimated_duration", 60) if requirements else 60,
                **(requirements or {})
            }
            
            result = await self.agent_manager.delegate_task_intelligently(
                task_id=task_id,
                task_requirements=task_requirements,
                task_priority=priority
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error delegating task: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_llm_provider_status(self) -> Dict[str, Any]:
        """Get LLM provider status"""
        
        try:
            return self.llm_provider_manager.get_system_status()
            
        except Exception as e:
            return {"error": str(e)}
    
    async def add_llm_provider(
        self,
        provider_type: str,
        provider_name: str,
        api_key: str,
        **config
    ) -> Dict[str, Any]:
        """Add a new LLM provider"""
        
        try:
            from .llm_providers.config_manager import ProviderType
            
            provider_type_enum = ProviderType(provider_type)
            provider_id = f"{provider_type}_{int(datetime.utcnow().timestamp())}"
            
            success = await self.llm_provider_manager.add_provider(
                provider_id=provider_id,
                provider_type=provider_type_enum,
                provider_name=provider_name,
                api_key=api_key,
                **config
            )
            
            return {
                "success": success,
                "provider_id": provider_id if success else None,
                "message": f"Provider {provider_name} {'added' if success else 'failed to add'}"
            }
            
        except Exception as e:
            logger.error(f"Error adding LLM provider: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for dashboard"""
        
        try:
            current_state = await self.state_holder.get_current_state()
            recent_states = await self.state_holder.get_recent_states(20)
            performance_summary = self.agent_manager.autonomous_executor.get_performance_summary() if hasattr(self.agent_manager, 'autonomous_executor') else {}
            
            return {
                "current_iteration": current_state.iteration_number if current_state else 0,
                "recent_states_count": len(recent_states),
                "execution_performance": performance_summary,
                "system_metrics": current_state.metrics if current_state else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting execution metrics: {e}")
            return {"error": str(e)}
    
    async def get_forecasting_data(self) -> Dict[str, Any]:
        """Get forecasting data for dashboard"""
        
        try:
            current_state = await self.state_holder.get_current_state()
            if not current_state:
                return {"error": "No current state available"}
            
            # Generate forecast
            forecast = await self.forecaster.forecast_next_states(current_state, "short-term")
            
            # Get trend predictions
            trend_predictions = await self.forecaster.predict_performance_trends(current_state)
            
            return {
                "forecast": forecast.dict(),
                "trends": trend_predictions
            }
            
        except Exception as e:
            logger.error(f"Error getting forecasting data: {e}")
            return {"error": str(e)}


# Global instance for the application
dspy_boss = DSPYBossIntegration()


# API endpoints for Next.js integration
async def api_initialize_system():
    """API endpoint to initialize the system"""
    return await dspy_boss.initialize_system()


async def api_get_system_overview():
    """API endpoint for system overview"""
    return await dspy_boss.get_system_overview()


async def api_get_agent_hierarchy():
    """API endpoint for agent hierarchy"""
    return await dspy_boss.get_agent_hierarchy_display()


async def api_spawn_agent(request_data: Dict[str, Any]):
    """API endpoint to spawn agent"""
    return await dspy_boss.spawn_agent(
        role=request_data.get("role", "generalist"),
        capabilities=request_data.get("capabilities"),
        specialization=request_data.get("specialization")
    )


async def api_delegate_task(request_data: Dict[str, Any]):
    """API endpoint to delegate task"""
    return await dspy_boss.delegate_task(
        task_id=request_data.get("task_id"),
        task_description=request_data.get("description"),
        requirements=request_data.get("requirements"),
        priority=request_data.get("priority", 3)
    )


async def api_get_llm_providers():
    """API endpoint for LLM provider status"""
    return await dspy_boss.get_llm_provider_status()


async def api_add_llm_provider(request_data: Dict[str, Any]):
    """API endpoint to add LLM provider"""
    return await dspy_boss.add_llm_provider(
        provider_type=request_data.get("provider_type"),
        provider_name=request_data.get("provider_name"),
        api_key=request_data.get("api_key"),
        **request_data.get("config", {})
    )


async def api_get_execution_metrics():
    """API endpoint for execution metrics"""
    return await dspy_boss.get_execution_metrics()


async def api_get_forecasting_data():
    """API endpoint for forecasting data"""
    return await dspy_boss.get_forecasting_data()
