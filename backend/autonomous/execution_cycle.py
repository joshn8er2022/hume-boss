
"""
Execution cycle management for coordinating the autonomous system
Provides high-level orchestration of the autonomous execution
"""

import asyncio
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger

from .iteration_engine import IterationEngine
from .autonomous_executor import AutonomousExecutor, ExecutionConfig, ExecutorState
from ..state.state_holder import StateHolder
from ..state.historical_manager import HistoricalStateManager
from ..state.forecasting_engine import StateForecaster


class CyclePhase(Enum):
    """Phases in the execution cycle"""
    INITIALIZATION = "initialization"
    AUTONOMOUS_EXECUTION = "autonomous_execution"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class CycleConfig(BaseModel):
    """Configuration for execution cycle"""
    
    # Cycle timing
    monitoring_interval_seconds: int = Field(default=60, description="Monitoring check interval")
    maintenance_interval_hours: int = Field(default=6, description="Maintenance cycle interval")
    
    # Health checks
    health_check_interval_seconds: int = Field(default=120, description="Health check interval")
    performance_review_interval_hours: int = Field(default=1, description="Performance review interval")
    
    # Maintenance operations
    auto_compression_enabled: bool = Field(default=True, description="Enable automatic state compression")
    auto_archival_enabled: bool = Field(default=True, description="Enable automatic state archival")
    pattern_analysis_interval_hours: int = Field(default=4, description="Pattern analysis interval")
    
    # Safety and recovery
    auto_recovery_enabled: bool = Field(default=True, description="Enable automatic error recovery")
    graceful_shutdown_timeout_seconds: int = Field(default=300, description="Graceful shutdown timeout")


class ExecutionCycle:
    """
    High-level execution cycle coordinator
    Manages the complete autonomous system lifecycle
    """
    
    def __init__(
        self,
        state_holder: StateHolder,
        historical_manager: HistoricalStateManager,
        forecaster: StateForecaster,
        cycle_config: CycleConfig = None,
        execution_config: ExecutionConfig = None
    ):
        self.state_holder = state_holder
        self.historical_manager = historical_manager
        self.forecaster = forecaster
        
        self.cycle_config = cycle_config or CycleConfig()
        self.execution_config = execution_config or ExecutionConfig()
        
        # Initialize core components
        self.iteration_engine = IterationEngine(
            state_holder=state_holder,
            historical_manager=historical_manager,
            forecaster=forecaster
        )
        
        self.autonomous_executor = AutonomousExecutor(
            iteration_engine=self.iteration_engine,
            config=self.execution_config
        )
        
        # Cycle state
        self.current_phase = CyclePhase.INITIALIZATION
        self.is_running = False
        self.shutdown_requested = False
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.maintenance_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Metrics and monitoring
        self.cycle_start_time: Optional[datetime] = None
        self.last_maintenance: Optional[datetime] = None
        self.last_health_check: Optional[datetime] = None
        self.last_performance_review: Optional[datetime] = None
        
        # Callbacks
        self.phase_change_callbacks: List[Callable] = []
        self.health_issue_callbacks: List[Callable] = []
        self.maintenance_callbacks: List[Callable] = []
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Setup executor callbacks
        self._setup_executor_callbacks()
        
        logger.info("ExecutionCycle initialized")
    
    async def start_complete_system(self) -> bool:
        """
        Start the complete autonomous system
        
        Returns:
            True if started successfully
        """
        if self.is_running:
            logger.warning("System is already running")
            return False
        
        try:
            logger.info("Starting complete autonomous system")
            
            # Phase 1: Initialization
            await self._transition_to_phase(CyclePhase.INITIALIZATION)
            
            if not await self._initialize_system():
                logger.error("System initialization failed")
                return False
            
            # Phase 2: Start autonomous execution
            await self._transition_to_phase(CyclePhase.AUTONOMOUS_EXECUTION)
            
            if not await self.autonomous_executor.start_autonomous_execution():
                logger.error("Failed to start autonomous execution")
                return False
            
            # Phase 3: Start monitoring and maintenance
            await self._transition_to_phase(CyclePhase.MONITORING)
            
            await self._start_background_tasks()
            
            # Mark system as running
            self.is_running = True
            self.cycle_start_time = datetime.utcnow()
            self.shutdown_requested = False
            
            logger.info("Complete autonomous system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            await self._emergency_shutdown()
            return False
    
    async def stop_complete_system(self) -> bool:
        """
        Stop the complete autonomous system gracefully
        
        Returns:
            True if stopped successfully
        """
        if not self.is_running:
            logger.info("System is not running")
            return True
        
        try:
            logger.info("Stopping complete autonomous system")
            
            # Mark shutdown as requested
            self.shutdown_requested = True
            
            # Phase 1: Stop autonomous execution
            await self._transition_to_phase(CyclePhase.SHUTDOWN)
            
            if not await self.autonomous_executor.stop_autonomous_execution():
                logger.warning("Autonomous executor did not stop gracefully")
            
            # Phase 2: Stop background tasks
            await self._stop_background_tasks()
            
            # Phase 3: Final maintenance
            await self._final_maintenance()
            
            # Mark system as stopped
            self.is_running = False
            self.current_phase = CyclePhase.SHUTDOWN
            
            logger.info("Complete autonomous system stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during system stop: {e}")
            await self._emergency_shutdown()
            return False
    
    async def pause_system(self) -> bool:
        """Pause the autonomous system"""
        
        if not self.is_running:
            logger.warning("Cannot pause system that is not running")
            return False
        
        logger.info("Pausing autonomous system")
        
        return await self.autonomous_executor.pause_autonomous_execution()
    
    async def resume_system(self) -> bool:
        """Resume the paused autonomous system"""
        
        if not self.is_running:
            logger.warning("Cannot resume system that is not running")
            return False
        
        logger.info("Resuming autonomous system")
        
        return await self.autonomous_executor.resume_autonomous_execution()
    
    async def _initialize_system(self) -> bool:
        """Initialize all system components"""
        
        try:
            logger.info("Initializing system components")
            
            # Initialize state if needed
            current_state = await self.state_holder.get_current_state()
            if not current_state:
                logger.info("Initializing first system state")
                await self.state_holder.initialize_state({
                    "boss_state": "INITIALIZING",
                    "system_phase": "initialization"
                })
            
            # Run initial pattern analysis
            logger.info("Running initial pattern analysis")
            patterns = await self.historical_manager.analyze_patterns(lookback_days=7)
            logger.info(f"Identified {len(patterns)} historical patterns")
            
            # Initialize forecasting
            if current_state:
                logger.info("Generating initial forecast")
                forecast = await self.forecaster.forecast_next_states(current_state, "short-term")
                logger.info(f"Initial forecast confidence: {forecast.overall_confidence:.2f}")
            
            # Update state to indicate initialization complete
            await self.state_holder.update_state({
                "boss_state": "AWAKE",
                "system_phase": "ready"
            })
            
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        
        logger.info("Starting background tasks")
        
        # Monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Maintenance task
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        # Health check task
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Background tasks started")
    
    async def _stop_background_tasks(self):
        """Stop all background tasks"""
        
        logger.info("Stopping background tasks")
        
        tasks = [
            self.monitoring_task,
            self.maintenance_task,
            self.health_check_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        
        logger.info("Background tasks stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        logger.info("Monitoring loop started")
        
        try:
            while not self.shutdown_requested:
                await asyncio.sleep(self.cycle_config.monitoring_interval_seconds)
                
                if self.shutdown_requested:
                    break
                
                try:
                    # Monitor autonomous executor
                    executor_state = self.autonomous_executor.get_execution_state()
                    
                    if executor_state == ExecutorState.ERROR:
                        logger.warning("Autonomous executor is in error state")
                        await self._handle_executor_error()
                    
                    # Monitor system performance
                    performance = self.autonomous_executor.get_performance_summary()
                    
                    if performance.get("recent_success_rate", 1.0) < 0.5:
                        logger.warning(f"Low success rate detected: {performance['recent_success_rate']:.1%}")
                        await self._handle_low_performance()
                    
                    # Check if performance review is needed
                    if await self._should_run_performance_review():
                        await self._run_performance_review()
                    
                except Exception as e:
                    logger.warning(f"Monitoring loop error: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop fatal error: {e}")
        
        logger.info("Monitoring loop ended")
    
    async def _maintenance_loop(self):
        """Maintenance loop for system upkeep"""
        
        logger.info("Maintenance loop started")
        
        try:
            while not self.shutdown_requested:
                await asyncio.sleep(self.cycle_config.maintenance_interval_hours * 3600)
                
                if self.shutdown_requested:
                    break
                
                try:
                    await self._run_maintenance_cycle()
                    
                except Exception as e:
                    logger.warning(f"Maintenance cycle error: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Maintenance loop cancelled")
        except Exception as e:
            logger.error(f"Maintenance loop fatal error: {e}")
        
        logger.info("Maintenance loop ended")
    
    async def _health_check_loop(self):
        """Health check loop"""
        
        logger.info("Health check loop started")
        
        try:
            while not self.shutdown_requested:
                await asyncio.sleep(self.cycle_config.health_check_interval_seconds)
                
                if self.shutdown_requested:
                    break
                
                try:
                    health_status = await self._comprehensive_health_check()
                    
                    if health_status["status"] == "critical":
                        logger.error("Critical health issues detected")
                        await self._handle_health_crisis(health_status)
                    elif health_status["status"] == "warning":
                        logger.warning("Health warnings detected")
                        await self._handle_health_warnings(health_status)
                    
                    self.last_health_check = datetime.utcnow()
                    
                except Exception as e:
                    logger.warning(f"Health check error: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Health check loop cancelled")
        except Exception as e:
            logger.error(f"Health check loop fatal error: {e}")
        
        logger.info("Health check loop ended")
    
    async def _run_maintenance_cycle(self):
        """Run a complete maintenance cycle"""
        
        logger.info("Running maintenance cycle")
        
        try:
            maintenance_start = datetime.utcnow()
            
            # State compression
            if self.cycle_config.auto_compression_enabled:
                logger.info("Running state compression")
                compressed_count = await self.state_holder.historical_manager.compress_old_states()
                logger.info(f"Compressed {compressed_count} old states")
            
            # State archival
            if self.cycle_config.auto_archival_enabled:
                logger.info("Running state archival")
                archival_success = await self.state_holder.historical_manager.archive_old_states()
                logger.info(f"State archival {'successful' if archival_success else 'failed'}")
            
            # Pattern analysis
            if await self._should_run_pattern_analysis():
                logger.info("Running pattern analysis")
                patterns = await self.historical_manager.analyze_patterns()
                logger.info(f"Identified {len(patterns)} patterns")
            
            # Forecast cleanup
            await self.forecaster.cleanup_old_forecasts()
            
            # Update maintenance timestamp
            self.last_maintenance = datetime.utcnow()
            
            maintenance_duration = (datetime.utcnow() - maintenance_start).total_seconds()
            logger.info(f"Maintenance cycle completed in {maintenance_duration:.1f} seconds")
            
            # Notify callbacks
            for callback in self.maintenance_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(maintenance_duration)
                    else:
                        callback(maintenance_duration)
                except Exception as e:
                    logger.warning(f"Maintenance callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Maintenance cycle failed: {e}")
    
    async def _comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        
        health_status = {
            "timestamp": datetime.utcnow(),
            "status": "healthy",
            "issues": [],
            "warnings": [],
            "metrics": {}
        }
        
        try:
            # Check autonomous executor health
            executor_state = self.autonomous_executor.get_execution_state()
            performance = self.autonomous_executor.get_performance_summary()
            
            health_status["metrics"]["executor_state"] = executor_state.value
            health_status["metrics"]["success_rate"] = performance.get("overall_success_rate", 0)
            health_status["metrics"]["consecutive_errors"] = performance.get("consecutive_errors", 0)
            
            # Check for critical issues
            if executor_state == ExecutorState.ERROR:
                health_status["issues"].append("Autonomous executor in error state")
                health_status["status"] = "critical"
            
            if performance.get("consecutive_errors", 0) >= 3:
                health_status["issues"].append(f"High consecutive error count: {performance['consecutive_errors']}")
                health_status["status"] = "critical"
            
            # Check for warnings
            if performance.get("recent_success_rate", 1.0) < 0.7:
                health_status["warnings"].append(f"Low recent success rate: {performance['recent_success_rate']:.1%}")
                if health_status["status"] == "healthy":
                    health_status["status"] = "warning"
            
            # Check state holder health
            state_health = await self.state_holder.get_system_health_summary()
            health_status["metrics"]["state_health"] = state_health["status"]
            
            if state_health["status"] == "critical":
                health_status["issues"].append("State holder reports critical status")
                health_status["status"] = "critical"
            elif state_health["status"] == "warning":
                health_status["warnings"].append("State holder reports warning status")
                if health_status["status"] == "healthy":
                    health_status["status"] = "warning"
            
            # Check memory and performance
            if self.cycle_start_time:
                runtime_hours = (datetime.utcnow() - self.cycle_start_time).total_seconds() / 3600
                health_status["metrics"]["runtime_hours"] = runtime_hours
                
                if runtime_hours > 20:  # Long runtime
                    health_status["warnings"].append(f"Long runtime: {runtime_hours:.1f} hours")
                    if health_status["status"] == "healthy":
                        health_status["status"] = "warning"
            
        except Exception as e:
            health_status["issues"].append(f"Health check error: {str(e)}")
            health_status["status"] = "critical"
        
        return health_status
    
    async def _should_run_performance_review(self) -> bool:
        """Check if performance review should be run"""
        
        if not self.last_performance_review:
            return True
        
        time_since_review = datetime.utcnow() - self.last_performance_review
        return time_since_review.total_seconds() > (self.cycle_config.performance_review_interval_hours * 3600)
    
    async def _should_run_pattern_analysis(self) -> bool:
        """Check if pattern analysis should be run"""
        
        if not self.last_maintenance:
            return True
        
        time_since_analysis = datetime.utcnow() - self.last_maintenance
        return time_since_analysis.total_seconds() > (self.cycle_config.pattern_analysis_interval_hours * 3600)
    
    async def _run_performance_review(self):
        """Run comprehensive performance review"""
        
        logger.info("Running performance review")
        
        try:
            # Get current performance data
            performance = self.autonomous_executor.get_performance_summary()
            
            # Analyze trends
            current_state = await self.state_holder.get_current_state()
            if current_state:
                trend_predictions = await self.forecaster.predict_performance_trends(current_state)
                
                # Log performance insights
                logger.info(f"Performance Review Results:")
                logger.info(f"  - Overall success rate: {performance.get('overall_success_rate', 0):.1%}")
                logger.info(f"  - Recent success rate: {performance.get('recent_success_rate', 0):.1%}")
                logger.info(f"  - Iterations per hour: {performance.get('iterations_per_hour', 0):.1f}")
                logger.info(f"  - Consecutive errors: {performance.get('consecutive_errors', 0)}")
                
                # Check for performance improvement opportunities
                if performance.get("average_duration", 0) > 60:
                    logger.info("  - Opportunity: Iteration duration optimization needed")
                
                if performance.get("recent_success_rate", 1.0) < 0.8:
                    logger.info("  - Opportunity: Error reduction strategies needed")
            
            self.last_performance_review = datetime.utcnow()
            
        except Exception as e:
            logger.warning(f"Performance review failed: {e}")
    
    async def _handle_executor_error(self):
        """Handle autonomous executor errors"""
        
        logger.warning("Handling autonomous executor error")
        
        if self.cycle_config.auto_recovery_enabled:
            try:
                # Attempt to restart executor
                logger.info("Attempting automatic recovery of executor")
                
                await self.autonomous_executor.stop_autonomous_execution()
                await asyncio.sleep(5)  # Brief pause
                
                success = await self.autonomous_executor.start_autonomous_execution()
                
                if success:
                    logger.info("Automatic recovery successful")
                else:
                    logger.error("Automatic recovery failed")
                    
            except Exception as e:
                logger.error(f"Automatic recovery error: {e}")
    
    async def _handle_low_performance(self):
        """Handle low performance situations"""
        
        logger.info("Handling low performance situation")
        
        # Get current state for analysis
        current_state = await self.state_holder.get_current_state()
        if current_state:
            # Generate performance improvement recommendations
            forecast = await self.forecaster.forecast_next_states(current_state, "short-term")
            
            # Log recommendations
            for recommendation in forecast.strategic_recommendations:
                logger.info(f"Performance recommendation: {recommendation}")
    
    async def _handle_health_crisis(self, health_status: Dict[str, Any]):
        """Handle critical health issues"""
        
        logger.error("Handling health crisis")
        
        for issue in health_status["issues"]:
            logger.error(f"Critical issue: {issue}")
        
        # Notify callbacks
        for callback in self.health_issue_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(health_status)
                else:
                    callback(health_status)
            except Exception as e:
                logger.warning(f"Health issue callback failed: {e}")
        
        # Consider emergency shutdown if too many critical issues
        if len(health_status["issues"]) >= 3:
            logger.error("Too many critical issues, considering emergency shutdown")
            # Could implement emergency shutdown logic here
    
    async def _handle_health_warnings(self, health_status: Dict[str, Any]):
        """Handle health warnings"""
        
        for warning in health_status["warnings"]:
            logger.warning(f"Health warning: {warning}")
    
    async def _final_maintenance(self):
        """Run final maintenance before shutdown"""
        
        logger.info("Running final maintenance")
        
        try:
            # Force save current state
            current_state = await self.state_holder.get_current_state()
            if current_state:
                await self.historical_manager.store_state(current_state)
            
            # Final cleanup
            await self.forecaster.cleanup_old_forecasts(max_age_hours=0)  # Clean all forecasts
            
            logger.info("Final maintenance completed")
            
        except Exception as e:
            logger.warning(f"Final maintenance error: {e}")
    
    async def _emergency_shutdown(self):
        """Emergency shutdown procedure"""
        
        logger.error("Executing emergency shutdown")
        
        try:
            # Stop executor immediately
            if self.autonomous_executor:
                self.autonomous_executor.should_stop = True
            
            # Cancel all tasks
            await self._stop_background_tasks()
            
            # Mark as stopped
            self.is_running = False
            self.current_phase = CyclePhase.SHUTDOWN
            
        except Exception as e:
            logger.error(f"Emergency shutdown error: {e}")
    
    async def _transition_to_phase(self, new_phase: CyclePhase):
        """Transition to a new cycle phase"""
        
        logger.info(f"Transitioning from {self.current_phase.value} to {new_phase.value}")
        
        self.current_phase = new_phase
        
        # Notify callbacks
        for callback in self.phase_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(new_phase)
                else:
                    callback(new_phase)
            except Exception as e:
                logger.warning(f"Phase change callback failed: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.shutdown_requested = True
            
            # Create task to stop system
            asyncio.create_task(self.stop_complete_system())
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except ValueError:
            # Signal handlers can only be set from main thread
            logger.info("Could not set signal handlers (not in main thread)")
    
    def _setup_executor_callbacks(self):
        """Setup callbacks for autonomous executor"""
        
        async def on_state_change(state: ExecutorState, stats):
            logger.info(f"Executor state changed to: {state.value}")
        
        async def on_error(error: Exception, stats):
            logger.error(f"Executor error: {error}")
        
        self.autonomous_executor.add_state_change_callback(on_state_change)
        self.autonomous_executor.add_error_callback(on_error)
    
    # Public interface methods
    
    def add_phase_change_callback(self, callback: Callable):
        """Add callback for phase changes"""
        self.phase_change_callbacks.append(callback)
    
    def add_health_issue_callback(self, callback: Callable):
        """Add callback for health issues"""
        self.health_issue_callbacks.append(callback)
    
    def add_maintenance_callback(self, callback: Callable):
        """Add callback for maintenance events"""
        self.maintenance_callbacks.append(callback)
    
    def get_current_phase(self) -> CyclePhase:
        """Get current execution phase"""
        return self.current_phase
    
    def is_system_running(self) -> bool:
        """Check if system is running"""
        return self.is_running
    
    async def get_complete_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        try:
            executor_status = await self.autonomous_executor.get_system_status()
            health_status = await self._comprehensive_health_check()
            
            return {
                "cycle_phase": self.current_phase.value,
                "is_running": self.is_running,
                "cycle_start_time": self.cycle_start_time.isoformat() if self.cycle_start_time else None,
                "last_maintenance": self.last_maintenance.isoformat() if self.last_maintenance else None,
                "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
                "executor_status": executor_status,
                "health_status": health_status,
                "config": {
                    "cycle_config": self.cycle_config.dict(),
                    "execution_config": self.execution_config.dict()
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "cycle_phase": self.current_phase.value,
                "is_running": self.is_running
            }
