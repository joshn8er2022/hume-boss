
"""
Main DSPY Boss class - orchestrates the entire system with autonomous DSPY-driven intelligence
"""

import asyncio
import signal
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger

import dspy

from .config import DSPYBossConfig, load_full_config
from .models import (
    TaskDefinition, TaskPriority, AgentConfig, SystemMetrics, ReportEntry,
    SystemState, AutonomousConfig, LLMProviderConfig
)
from .state_machine import StateMachineManager, BossState
from .mcp import MCPManager
from .task_manager import TaskManager
from .agents import AgentManager
from .self_diagnosis import SelfDiagnosisSystem

# New autonomous components
from .autonomous_engine import AutonomousEngine
from .state_holder import StateHolder
from .agent_hierarchy import AgentHierarchy
from .llm_providers import LLMProviderManager, LLMProvider


class DSPYBoss:
    """Main DSPY Boss system orchestrator - Now fully autonomous and DSPY-driven"""
    
    def __init__(self, config_dir: str = "configs"):
        # Load configuration
        self.config = load_full_config(config_dir)
        
        # Initialize core components
        self.state_manager = StateMachineManager()
        self.mcp_manager = MCPManager(self.config.mcp_servers)
        self.task_manager = TaskManager(
            workers=self.config.task_queue_workers
        )
        self.agent_manager = AgentManager(
            self.task_manager,
            self.mcp_manager
        )
        self.diagnosis_system = SelfDiagnosisSystem(self.state_manager)
        
        # NEW: Autonomous DSPY-driven components
        self.state_holder = StateHolder(max_recent_states=100)
        self.llm_manager = LLMProviderManager()
        self.autonomous_engine = AutonomousEngine(
            state_holder=self.state_holder,
            llm_manager=self.llm_manager,
            mcp_manager=self.mcp_manager
        )
        
        # System state
        self.is_running = False
        self.is_autonomous = False  # NEW: Track autonomous mode
        self.start_time: Optional[datetime] = None
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Setup logging
        self._setup_logging()
        
        # Setup DSPY (will be replaced by LLM manager)
        self._setup_dspy()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Initialize autonomous configuration
        self.autonomous_config = AutonomousConfig()
        
        logger.info(f"DSPY Boss initialized with {len(self.config.mcp_servers)} MCP servers, "
                   f"{len(self.config.agents)} agents, and {len(self.config.prompt_signatures)} prompt signatures")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logger.remove()  # Remove default handler
        
        # Add console handler
        logger.add(
            lambda msg: print(msg, end=""),
            level=self.config.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # Add file handler
        log_file = self.config.logs_dir / "dspy_boss.log"
        self.config.logs_dir.mkdir(exist_ok=True)
        
        logger.add(
            str(log_file),
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )
    
    def _setup_dspy(self):
        """Setup DSPY configuration"""
        try:
            # Configure DSPY with default LLM
            lm = dspy.LM(
                model=self.config.dspy_model,
                max_tokens=self.config.dspy_max_tokens,
                temperature=self.config.dspy_temperature
            )
            dspy.configure(lm=lm)
            
            logger.info(f"DSPY configured with model: {self.config.dspy_model}")
            
        except Exception as e:
            logger.warning(f"Error configuring DSPY: {e}. Using default configuration.")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self):
        """Start the DSPY Boss system"""
        if self.is_running:
            logger.warning("DSPY Boss is already running")
            return
        
        logger.info("Starting DSPY Boss system...")
        self.start_time = datetime.utcnow()
        
        try:
            # Initialize state machine
            self.state_manager.setup_default_callbacks()
            self.state_manager.transition.transition_to(BossState.AWAKE, "System startup")
            
            # Initialize MCP connections
            await self.mcp_manager.initialize()
            
            # Start task manager
            await self.task_manager.start()
            
            # Register sample task functions
            self._register_task_functions()
            
            # Load and start agents
            self.agent_manager.load_agents(self.config.agents, self.config.prompt_signatures)
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_running = True
            
            # Transition to thinking state to assess initial workload
            self.state_manager.transition.transition_to(BossState.THINKING, "Initial workload assessment")
            
            logger.info("DSPY Boss system started successfully")
            
        except Exception as e:
            logger.error(f"Error starting DSPY Boss system: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Shutdown the DSPY Boss system"""
        if not self.is_running:
            return
        
        logger.info("Shutting down DSPY Boss system...")
        self.is_running = False
        
        # Transition to stop state
        self.state_manager.transition.transition_to(BossState.STOP, "System shutdown")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Stop agents
        self.agent_manager.stop_all_agents()
        
        # Stop task manager
        await self.task_manager.stop()
        
        # Shutdown MCP connections
        await self.mcp_manager.shutdown()
        
        logger.info("DSPY Boss system shutdown complete")
    
    def _register_task_functions(self):
        """Register available task functions"""
        # Register built-in task functions
        async def sample_task(query: str = "test", **kwargs):
            """Sample task function for testing"""
            return f"Completed task with query: {query}"
        
        self.task_manager.register_task_function("sample_task", sample_task)
        self.task_manager.register_task_function("test_function", sample_task)
        
        logger.info("Registered task functions")
    
    async def _start_background_tasks(self):
        """Start background monitoring and management tasks"""
        
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.background_tasks.append(health_task)
        
        # Workload management task
        workload_task = asyncio.create_task(self._workload_management_loop())
        self.background_tasks.append(workload_task)
        
        # Reflection task
        reflection_task = asyncio.create_task(self._reflection_loop())
        self.background_tasks.append(reflection_task)
        
        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.background_tasks.append(metrics_task)
        
        logger.info("Started background tasks")
    
    async def _health_monitoring_loop(self):
        """Background health monitoring"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Run health check
                health_result = await self.diagnosis_system.run_health_check()
                
                # Handle critical health issues
                if health_result.status == "critical":
                    logger.critical(f"Critical health issue detected: {health_result.summary}")
                    self.state_manager.handle_error(f"Critical health: {health_result.summary}")
                elif health_result.status == "warning":
                    logger.warning(f"Health warning: {health_result.summary}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _workload_management_loop(self):
        """Background workload management"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Get current workload
                pending_tasks = len(self.state_manager.transition.state_data.pending_tasks)
                
                # Update workload in state manager
                self.state_manager.update_workload(pending_tasks)
                
                # Check if we need to spawn new agents
                if pending_tasks >= self.config.agent_spawn_threshold:
                    available_agents = len([
                        a for a in self.agent_manager.agents.values() 
                        if a.config.is_available and len(a.current_tasks) < a.config.max_concurrent_tasks
                    ])
                    
                    if available_agents == 0:
                        logger.info(f"High workload ({pending_tasks} tasks), spawning new agent")
                        self.agent_manager.spawn_agentic_agent()
                
                # Remove idle agents
                if pending_tasks < 3:  # Low workload
                    removed_count = self.agent_manager.remove_idle_agents(self.config.agent_idle_timeout)
                    if removed_count > 0:
                        logger.info(f"Removed {removed_count} idle agents")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in workload management loop: {e}")
                await asyncio.sleep(30)
    
    async def _reflection_loop(self):
        """Background reflection and optimization"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.reflection_interval)
                
                # Transition to reflecting state
                if self.state_manager.transition.can_transition_to(BossState.REFLECTING):
                    self.state_manager.transition.transition_to(BossState.REFLECTING, "Scheduled reflection")
                    
                    # Run performance analysis
                    perf_result = await self.diagnosis_system.analyze_performance()
                    
                    # Update reflection data
                    self.state_manager.transition.state_data.last_reflection = datetime.utcnow()
                    self.state_manager.transition.state_data.reflection_notes = perf_result.summary
                    
                    # Extract improvement actions from recommendations
                    if perf_result.recommendations:
                        self.state_manager.transition.state_data.improvement_actions.extend(
                            perf_result.recommendations
                        )
                    
                    logger.info(f"Reflection completed: {perf_result.summary}")
                    
                    # Transition back to appropriate state
                    if len(self.state_manager.transition.state_data.pending_tasks) > 0:
                        self.state_manager.transition.transition_to(BossState.AWAKE, "Tasks pending after reflection")
                    else:
                        self.state_manager.transition.transition_to(BossState.IDLE, "No tasks after reflection")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in reflection loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _metrics_collection_loop(self):
        """Background metrics collection"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute
                
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                
                # Add to diagnosis system
                self.diagnosis_system.add_system_metrics(metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            import psutil
            
            # Task metrics
            task_stats = self.task_manager.get_stats()
            
            # Agent metrics
            agent_stats = self.agent_manager.get_agent_stats()
            
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # MCP metrics
            mcp_stats = self.mcp_manager.get_server_stats()
            active_mcp_connections = len([s for s in mcp_stats.values() if s.get('is_connected', False)])
            
            metrics = SystemMetrics(
                tasks_per_minute=task_stats.get('tasks_per_minute', 0.0),
                average_task_completion_time=task_stats.get('average_duration', 0.0),
                task_success_rate=((task_stats.get('completed_tasks', 0) / 
                                  max(task_stats.get('total_tasks', 1), 1)) * 100),
                active_agents_count=agent_stats.get('active_agents', 0),
                agent_utilization=((agent_stats.get('active_agents', 0) / 
                                  max(agent_stats.get('total_agents', 1), 1)) * 100),
                memory_usage_mb=memory.used / (1024 * 1024),
                cpu_usage_percent=cpu_percent,
                active_mcp_connections=active_mcp_connections,
                mcp_response_time_avg=0.0  # Would calculate from MCP stats
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics()  # Return default metrics
    
    # Public API methods
    
    async def add_task(self, name: str, description: str, function_name: str, 
                      parameters: Dict[str, Any] = None, priority: TaskPriority = TaskPriority.MEDIUM,
                      timeout: Optional[int] = None, requires_human: bool = False) -> str:
        """Add a new task to the system"""
        
        task = TaskDefinition(
            name=name,
            description=description,
            function_name=function_name,
            parameters=parameters or {},
            priority=priority,
            timeout=timeout or self.config.task_timeout_default,
            requires_human=requires_human
        )
        
        # Add to task manager
        success = await self.task_manager.add_task(task)
        
        if success:
            # Try to assign to an agent immediately
            assigned_agent = self.agent_manager.assign_task_to_best_agent(task)
            if assigned_agent:
                logger.info(f"Task {task.name} assigned to agent {assigned_agent.config.name}")
            else:
                logger.info(f"Task {task.name} queued - no suitable agent available")
            
            return task.id
        else:
            raise RuntimeError("Failed to add task to system")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "system": {
                "is_running": self.is_running,
                "uptime_seconds": uptime,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "version": self.config.version
            },
            "state_machine": self.state_manager.get_status(),
            "task_manager": self.task_manager.get_stats(),
            "agent_manager": self.agent_manager.get_agent_stats(),
            "mcp_servers": self.mcp_manager.get_server_stats(),
            "health": self.diagnosis_system.get_system_health_summary(),
            "metrics": self.diagnosis_system.get_metrics_summary()
        }
    
    async def run_diagnosis(self, diagnosis_type: str = "comprehensive") -> Any:
        """Run system diagnosis"""
        if diagnosis_type == "health":
            return await self.diagnosis_system.run_health_check()
        elif diagnosis_type == "performance":
            return await self.diagnosis_system.analyze_performance()
        elif diagnosis_type == "errors":
            return await self.diagnosis_system.investigate_errors()
        elif diagnosis_type == "comprehensive":
            return await self.diagnosis_system.run_comprehensive_diagnosis()
        else:
            raise ValueError(f"Unknown diagnosis type: {diagnosis_type}")
    
    async def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        return await self.diagnosis_system.get_current_metrics()
    
    async def run_forever(self):
        """Run the system indefinitely"""
        await self.start()
        
        try:
            while self.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            await self.shutdown()


# Convenience function for running the system
async def run_dspy_boss(config_dir: str = "configs", dry_run: bool = False):
    """Run DSPY Boss system"""
    boss = DSPYBoss(config_dir)
    
    if dry_run:
        logger.info("Dry run mode - testing initialization only")
        await boss.start()
        
        # Add a sample task
        task_id = await boss.add_task(
            name="Sample Research Task",
            description="Research the latest trends in AI",
            function_name="research",
            parameters={"query": "AI trends 2024", "depth": "basic"},
            priority=TaskPriority.MEDIUM
        )
        
        logger.info(f"Added sample task: {task_id}")
        
        # Wait a bit to see system in action
        await asyncio.sleep(10)
        
        # Show status
        status = boss.get_system_status()
        logger.info(f"System status: {status}")
        
        await boss.shutdown()
    else:
        await boss.run_forever()


if __name__ == "__main__":
    import sys
    
    dry_run = "--dry-run" in sys.argv
    config_dir = "configs"
    
    # Check for config directory argument
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_dir = sys.argv[i + 1]
    
    asyncio.run(run_dspy_boss(config_dir, dry_run))

    # ============= NEW AUTONOMOUS METHODS =============
    
    async def start_autonomous_mode(self):
        """Start fully autonomous DSPY-driven operation"""
        if self.is_autonomous:
            logger.warning("Already in autonomous mode")
            return
            
        logger.info("🚀 Starting AUTONOMOUS DSPY-driven mode...")
        
        # Initialize LLM providers first
        provider_count = await self.llm_manager.initialize_all_providers()
        if provider_count == 0:
            logger.error("❌ No LLM providers initialized. Cannot start autonomous mode.")
            return False
            
        # Start autonomous engine
        await self.autonomous_engine.start_autonomous_operation()
        self.is_autonomous = True
        
        # Update system state
        self.state_holder.update_current_state({
            "autonomous_mode": True,
            "start_time": datetime.utcnow().isoformat(),
            "active_providers": self.llm_manager.get_available_providers()
        })
        
        logger.info("✅ Autonomous mode ACTIVE - DSPY signatures are now in control")
        return True
        
    async def stop_autonomous_mode(self):
        """Stop autonomous operation"""
        if not self.is_autonomous:
            logger.warning("Not in autonomous mode")
            return
            
        logger.info("⏹️ Stopping autonomous mode...")
        
        await self.autonomous_engine.stop_autonomous_operation()
        self.is_autonomous = False
        
        # Update system state
        self.state_holder.update_current_state({
            "autonomous_mode": False,
            "stop_time": datetime.utcnow().isoformat()
        })
        
        logger.info("✅ Autonomous mode stopped")
        
    def set_llm_api_key(self, provider: str, api_key: str) -> bool:
        """Set API key for LLM provider"""
        try:
            from .llm_providers import LLMProvider
            provider_enum = LLMProvider(provider.lower())
            self.llm_manager.set_api_key(provider_enum, api_key)
            logger.info(f"🔑 API key set for {provider}")
            return True
        except ValueError:
            logger.error(f"❌ Unknown provider: {provider}")
            return False
            
    def get_llm_provider_status(self) -> Dict[str, Any]:
        """Get status of all LLM providers"""
        return self.llm_manager.get_provider_status()
        
    def get_agent_hierarchy_status(self) -> Dict[str, Any]:
        """Get agent hierarchy status with proper numbering"""
        return self.autonomous_engine.agent_hierarchy.get_all_statuses()
        
    def get_agent_display_info(self) -> List[Dict[str, Any]]:
        """Get agent info for UI display (Boss = Agent 0, others = Agent 1, 2, 3...)"""
        return self.autonomous_engine.agent_hierarchy.get_agent_display_info()
        
    def get_autonomous_status(self) -> Dict[str, Any]:
        """Get autonomous operation status"""
        current_context = self.autonomous_engine.current_context
        
        return {
            "is_autonomous": self.is_autonomous,
            "iteration_count": self.autonomous_engine.iteration_count,
            "current_phase": current_context.phase.value if current_context else None,
            "current_iteration_id": current_context.iteration_id if current_context else None,
            "system_state": self.state_holder.get_current_state(),
            "recent_decisions": self.state_holder.get_recent_states(5),
            "agent_hierarchy": self.get_agent_hierarchy_status(),
            "llm_providers": self.get_llm_provider_status()
        }
        
    def get_autonomous_config(self) -> Dict[str, Any]:
        """Get autonomous configuration"""
        return {
            "autonomous_config": self.autonomous_config.dict(),
            "llm_config": self.llm_manager.export_config(),
        }

