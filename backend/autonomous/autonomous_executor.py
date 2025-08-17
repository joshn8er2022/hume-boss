
"""
Autonomous executor that runs the continuous execution loop
Removes manual UI triggers and provides truly autonomous operation
"""

import asyncio
import signal
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger

from .iteration_engine import IterationEngine, IterationResult, IterationPhase
from ..state.state_holder import StateHolder


class ExecutorState(Enum):
    """States of the autonomous executor"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class ExecutionConfig(BaseModel):
    """Configuration for autonomous execution"""
    
    # Timing configuration
    iteration_interval_seconds: int = Field(default=30, description="Seconds between iterations")
    max_iteration_duration: int = Field(default=300, description="Maximum seconds per iteration")
    idle_check_interval: int = Field(default=10, description="Seconds between idle checks")
    
    # Error handling
    max_consecutive_errors: int = Field(default=3, description="Max consecutive errors before stopping")
    error_cooldown_seconds: int = Field(default=60, description="Cooldown after errors")
    
    # Performance limits
    max_iterations_per_hour: int = Field(default=120, description="Rate limiting")
    memory_check_interval: int = Field(default=300, description="Memory check interval")
    
    # Adaptive behavior
    adaptive_timing: bool = Field(default=True, description="Adjust timing based on performance")
    learning_mode: bool = Field(default=True, description="Enable learning from iterations")
    
    # Safety limits
    max_continuous_runtime_hours: int = Field(default=24, description="Max continuous runtime")
    emergency_stop_on_resource_exhaustion: bool = Field(default=True)


class ExecutionStats(BaseModel):
    """Statistics for autonomous execution"""
    
    started_at: Optional[datetime] = None
    total_runtime: float = 0.0
    total_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    consecutive_errors: int = 0
    last_iteration_at: Optional[datetime] = None
    last_error_at: Optional[datetime] = None
    average_iteration_duration: float = 0.0
    iterations_per_hour: float = 0.0
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class AutonomousExecutor:
    """
    Autonomous executor that runs continuous DSPY-driven iterations
    This replaces manual UI triggers with intelligent autonomous operation
    """
    
    def __init__(
        self,
        iteration_engine: IterationEngine,
        config: ExecutionConfig = None
    ):
        self.iteration_engine = iteration_engine
        self.config = config or ExecutionConfig()
        
        # State management
        self.state = ExecutorState.STOPPED
        self.stats = ExecutionStats()
        
        # Execution control
        self.should_stop = False
        self.should_pause = False
        self.execution_task: Optional[asyncio.Task] = None
        
        # Adaptive behavior
        self.iteration_timings: List[float] = []
        self.performance_history: List[Dict[str, Any]] = []
        
        # Callbacks
        self.state_change_callbacks: List[Callable] = []
        self.iteration_complete_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Safety monitoring
        self.resource_monitor_task: Optional[asyncio.Task] = None
        
        logger.info("AutonomousExecutor initialized")
    
    async def start_autonomous_execution(self) -> bool:
        """
        Start autonomous execution loop
        
        Returns:
            True if started successfully
        """
        if self.state != ExecutorState.STOPPED:
            logger.warning(f"Cannot start executor in state {self.state}")
            return False
        
        try:
            self.state = ExecutorState.STARTING
            await self._notify_state_change()
            
            logger.info("Starting autonomous execution")
            
            # Initialize stats
            self.stats.started_at = datetime.utcnow()
            self.should_stop = False
            self.should_pause = False
            
            # Start resource monitoring
            self.resource_monitor_task = asyncio.create_task(self._resource_monitor_loop())
            
            # Start main execution loop
            self.execution_task = asyncio.create_task(self._main_execution_loop())
            
            self.state = ExecutorState.RUNNING
            await self._notify_state_change()
            
            logger.info("Autonomous execution started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start autonomous execution: {e}")
            self.state = ExecutorState.ERROR
            await self._notify_state_change()
            return False
    
    async def stop_autonomous_execution(self) -> bool:
        """
        Stop autonomous execution gracefully
        
        Returns:
            True if stopped successfully
        """
        if self.state == ExecutorState.STOPPED:
            return True
        
        try:
            self.state = ExecutorState.STOPPING
            await self._notify_state_change()
            
            logger.info("Stopping autonomous execution")
            
            # Signal stop
            self.should_stop = True
            
            # Wait for execution task to complete
            if self.execution_task:
                try:
                    await asyncio.wait_for(self.execution_task, timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning("Execution task did not stop gracefully, canceling")
                    self.execution_task.cancel()
            
            # Stop resource monitoring
            if self.resource_monitor_task:
                self.resource_monitor_task.cancel()
                try:
                    await self.resource_monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Update stats
            if self.stats.started_at:
                self.stats.total_runtime = (datetime.utcnow() - self.stats.started_at).total_seconds()
            
            self.state = ExecutorState.STOPPED
            await self._notify_state_change()
            
            logger.info("Autonomous execution stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping autonomous execution: {e}")
            self.state = ExecutorState.ERROR
            await self._notify_state_change()
            return False
    
    async def pause_autonomous_execution(self) -> bool:
        """
        Pause autonomous execution
        
        Returns:
            True if paused successfully
        """
        if self.state != ExecutorState.RUNNING:
            logger.warning(f"Cannot pause executor in state {self.state}")
            return False
        
        self.state = ExecutorState.PAUSING
        self.should_pause = True
        await self._notify_state_change()
        
        logger.info("Pausing autonomous execution")
        
        # Wait for current iteration to complete
        await asyncio.sleep(1)
        
        self.state = ExecutorState.PAUSED
        await self._notify_state_change()
        
        logger.info("Autonomous execution paused")
        return True
    
    async def resume_autonomous_execution(self) -> bool:
        """
        Resume paused autonomous execution
        
        Returns:
            True if resumed successfully
        """
        if self.state != ExecutorState.PAUSED:
            logger.warning(f"Cannot resume executor in state {self.state}")
            return False
        
        self.should_pause = False
        self.state = ExecutorState.RUNNING
        await self._notify_state_change()
        
        logger.info("Autonomous execution resumed")
        return True
    
    async def _main_execution_loop(self):
        """
        Main autonomous execution loop
        This is the core of the autonomous system
        """
        logger.info("Main execution loop started")
        
        try:
            while not self.should_stop:
                loop_start_time = time.time()
                
                # Check if we should pause
                if self.should_pause:
                    await self._wait_for_resume()
                    continue
                
                # Check safety limits
                if not await self._check_safety_limits():
                    break
                
                # Check if we should execute an iteration
                if await self._should_execute_iteration():
                    await self._execute_single_iteration_with_monitoring()
                
                # Adaptive timing
                sleep_duration = await self._calculate_next_iteration_delay()
                
                # Sleep with interruption checking
                await self._interruptible_sleep(sleep_duration)
                
                # Update runtime stats
                loop_duration = time.time() - loop_start_time
                await self._update_runtime_stats(loop_duration)
                
        except Exception as e:
            logger.error(f"Main execution loop error: {e}")
            self.state = ExecutorState.ERROR
            await self._notify_state_change()
            await self._notify_error(e)
        
        logger.info("Main execution loop ended")
    
    async def _execute_single_iteration_with_monitoring(self):
        """
        Execute a single iteration with comprehensive monitoring
        """
        iteration_start_time = time.time()
        
        try:
            logger.debug("Executing autonomous iteration")
            
            # Check if iteration engine is ready
            if not await self.iteration_engine.should_continue_autonomous_execution():
                logger.info("Iteration engine indicates execution should not continue")
                self.should_stop = True
                return
            
            # Execute iteration
            result = await asyncio.wait_for(
                self.iteration_engine.execute_single_iteration(force_new_iteration=True),
                timeout=self.config.max_iteration_duration
            )
            
            # Update statistics
            iteration_duration = time.time() - iteration_start_time
            self.iteration_timings.append(iteration_duration)
            
            self.stats.total_iterations += 1
            self.stats.last_iteration_at = datetime.utcnow()
            
            if result.success:
                self.stats.successful_iterations += 1
                self.stats.consecutive_errors = 0
            else:
                self.stats.failed_iterations += 1
                self.stats.consecutive_errors += 1
                self.stats.last_error_at = datetime.utcnow()
            
            # Update average duration
            self._update_average_duration(iteration_duration)
            
            # Store performance data
            performance_data = {
                "timestamp": datetime.utcnow(),
                "duration": iteration_duration,
                "success": result.success,
                "iteration_number": result.final_state.iteration_number,
                "phase_durations": result.context.phase_durations,
                "actions_completed": len(result.context.actions_taken)
            }
            self.performance_history.append(performance_data)
            
            # Keep only last 100 performance records
            if len(self.performance_history) > 100:
                self.performance_history.pop(0)
            
            # Notify callbacks
            for callback in self.iteration_complete_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)
                    else:
                        callback(result)
                except Exception as e:
                    logger.warning(f"Iteration callback failed: {e}")
            
            # Check if we should stop due to consecutive errors
            if self.stats.consecutive_errors >= self.config.max_consecutive_errors:
                logger.error(f"Stopping due to {self.stats.consecutive_errors} consecutive errors")
                self.should_stop = True
                
                # Apply error cooldown
                await asyncio.sleep(self.config.error_cooldown_seconds)
            
            logger.debug(f"Iteration completed in {iteration_duration:.2f}s, success: {result.success}")
            
        except asyncio.TimeoutError:
            logger.error(f"Iteration timed out after {self.config.max_iteration_duration}s")
            self.stats.failed_iterations += 1
            self.stats.consecutive_errors += 1
            self.stats.last_error_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Iteration execution error: {e}")
            self.stats.failed_iterations += 1
            self.stats.consecutive_errors += 1
            self.stats.last_error_at = datetime.utcnow()
            await self._notify_error(e)
    
    async def _should_execute_iteration(self) -> bool:
        """
        Determine if we should execute an iteration
        Uses intelligent logic to decide when to act
        """
        
        # Check rate limiting
        if self.stats.started_at:
            runtime_hours = (datetime.utcnow() - self.stats.started_at).total_seconds() / 3600
            if runtime_hours > 0:
                current_rate = self.stats.total_iterations / runtime_hours
                if current_rate > self.config.max_iterations_per_hour:
                    logger.debug("Rate limit reached, skipping iteration")
                    return False
        
        # Check if enough time has passed since last iteration
        if self.stats.last_iteration_at:
            time_since_last = (datetime.utcnow() - self.stats.last_iteration_at).total_seconds()
            if time_since_last < self.config.iteration_interval_seconds:
                return False
        
        # Check system state to see if iteration is needed
        try:
            current_state = await self.iteration_engine.state_holder.get_current_state()
            if not current_state:
                return True  # Need initialization
            
            # Check if there are pending tasks or issues that need attention
            has_pending_tasks = len(current_state.active_tasks) > 0
            has_recent_errors = current_state.error_count > 0
            low_confidence = (current_state.confidence_score or 0) < 0.5
            
            # Execute if there's work to do or issues to address
            should_execute = has_pending_tasks or has_recent_errors or low_confidence
            
            if not should_execute:
                # Check for idle time - execute periodically even when idle
                if self.stats.last_iteration_at:
                    idle_time = (datetime.utcnow() - self.stats.last_iteration_at).total_seconds()
                    should_execute = idle_time > (self.config.iteration_interval_seconds * 3)  # 3x normal interval when idle
            
            return should_execute
            
        except Exception as e:
            logger.warning(f"Error checking if iteration should execute: {e}")
            return True  # Default to execute if we can't determine
    
    async def _calculate_next_iteration_delay(self) -> float:
        """
        Calculate how long to wait before next iteration
        Uses adaptive timing based on system performance
        """
        base_delay = self.config.iteration_interval_seconds
        
        if not self.config.adaptive_timing:
            return base_delay
        
        # Adjust based on recent performance
        if len(self.iteration_timings) >= 3:
            avg_duration = sum(self.iteration_timings[-3:]) / 3
            
            # If iterations are taking long, increase delay
            if avg_duration > 60:  # Over 1 minute
                base_delay *= 1.5
            elif avg_duration < 10:  # Very fast iterations
                base_delay *= 0.8
        
        # Adjust based on error rate
        if self.stats.total_iterations > 0:
            error_rate = self.stats.failed_iterations / self.stats.total_iterations
            if error_rate > 0.3:  # High error rate
                base_delay *= 2.0
            elif error_rate < 0.1:  # Low error rate
                base_delay *= 0.9
        
        # Adjust based on consecutive errors
        if self.stats.consecutive_errors > 0:
            base_delay *= (1 + self.stats.consecutive_errors * 0.5)
        
        # Keep within reasonable bounds
        return max(5.0, min(base_delay, 300.0))  # Between 5 seconds and 5 minutes
    
    async def _check_safety_limits(self) -> bool:
        """
        Check if safety limits are exceeded
        
        Returns:
            True if safe to continue, False if should stop
        """
        
        # Check maximum runtime
        if self.stats.started_at:
            runtime_hours = (datetime.utcnow() - self.stats.started_at).total_seconds() / 3600
            if runtime_hours > self.config.max_continuous_runtime_hours:
                logger.warning(f"Maximum runtime of {self.config.max_continuous_runtime_hours} hours exceeded")
                return False
        
        # Check consecutive errors
        if self.stats.consecutive_errors >= self.config.max_consecutive_errors:
            logger.warning(f"Maximum consecutive errors ({self.config.max_consecutive_errors}) exceeded")
            return False
        
        # Check system health through iteration engine
        try:
            if not await self.iteration_engine.should_continue_autonomous_execution():
                return False
        except Exception as e:
            logger.warning(f"Error checking iteration engine health: {e}")
        
        return True
    
    async def _wait_for_resume(self):
        """Wait while paused"""
        while self.should_pause and not self.should_stop:
            await asyncio.sleep(1)
    
    async def _interruptible_sleep(self, duration: float):
        """Sleep that can be interrupted by stop/pause signals"""
        
        sleep_start = time.time()
        
        while time.time() - sleep_start < duration:
            if self.should_stop or self.should_pause:
                break
            
            # Sleep in small increments to allow interruption
            await asyncio.sleep(min(1.0, duration - (time.time() - sleep_start)))
    
    async def _update_runtime_stats(self, loop_duration: float):
        """Update runtime statistics"""
        
        if self.stats.started_at:
            self.stats.total_runtime = (datetime.utcnow() - self.stats.started_at).total_seconds()
            
            # Calculate iterations per hour
            if self.stats.total_runtime > 0:
                self.stats.iterations_per_hour = (self.stats.total_iterations * 3600) / self.stats.total_runtime
    
    def _update_average_duration(self, duration: float):
        """Update average iteration duration"""
        
        if self.stats.total_iterations == 1:
            self.stats.average_iteration_duration = duration
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.average_iteration_duration = (
                alpha * duration + (1 - alpha) * self.stats.average_iteration_duration
            )
    
    async def _resource_monitor_loop(self):
        """Monitor system resources"""
        
        try:
            while not self.should_stop:
                if self.config.emergency_stop_on_resource_exhaustion:
                    # Monitor memory usage
                    try:
                        import psutil
                        memory_percent = psutil.virtual_memory().percent
                        
                        if memory_percent > 90:
                            logger.error(f"Memory usage critical: {memory_percent}%")
                            self.should_stop = True
                            break
                            
                    except ImportError:
                        # psutil not available, skip memory monitoring
                        pass
                
                await asyncio.sleep(self.config.memory_check_interval)
                
        except asyncio.CancelledError:
            logger.debug("Resource monitor cancelled")
        except Exception as e:
            logger.error(f"Resource monitor error: {e}")
    
    async def _notify_state_change(self):
        """Notify callbacks of state changes"""
        
        for callback in self.state_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.state, self.stats)
                else:
                    callback(self.state, self.stats)
            except Exception as e:
                logger.warning(f"State change callback failed: {e}")
    
    async def _notify_error(self, error: Exception):
        """Notify callbacks of errors"""
        
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error, self.stats)
                else:
                    callback(error, self.stats)
            except Exception as e:
                logger.warning(f"Error callback failed: {e}")
    
    # Public interface methods
    
    def add_state_change_callback(self, callback: Callable):
        """Add callback for state changes"""
        self.state_change_callbacks.append(callback)
    
    def add_iteration_complete_callback(self, callback: Callable):
        """Add callback for completed iterations"""
        self.iteration_complete_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add callback for errors"""
        self.error_callbacks.append(callback)
    
    def get_execution_state(self) -> ExecutorState:
        """Get current execution state"""
        return self.state
    
    def get_execution_stats(self) -> ExecutionStats:
        """Get current execution statistics"""
        return self.stats.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        if not self.performance_history:
            return {"status": "no_data"}
        
        recent_performance = self.performance_history[-10:]  # Last 10 iterations
        
        success_count = sum(1 for p in recent_performance if p["success"])
        avg_duration = sum(p["duration"] for p in recent_performance) / len(recent_performance)
        
        return {
            "recent_iterations": len(recent_performance),
            "recent_success_rate": success_count / len(recent_performance),
            "average_duration": avg_duration,
            "total_iterations": self.stats.total_iterations,
            "overall_success_rate": self.stats.successful_iterations / max(1, self.stats.total_iterations),
            "consecutive_errors": self.stats.consecutive_errors,
            "iterations_per_hour": self.stats.iterations_per_hour,
            "runtime_hours": self.stats.total_runtime / 3600 if self.stats.total_runtime else 0
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update execution configuration"""
        
        for key, value in new_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config {key} = {value}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        try:
            current_state = await self.iteration_engine.state_holder.get_current_state()
            
            return {
                "executor_state": self.state.value,
                "execution_stats": self.stats.dict(),
                "performance_summary": self.get_performance_summary(),
                "system_state": {
                    "boss_state": current_state.boss_state if current_state else None,
                    "agent_count": len(current_state.active_agents) if current_state else 0,
                    "task_count": len(current_state.active_tasks) if current_state else 0,
                    "error_count": current_state.error_count if current_state else 0
                },
                "config": self.config.dict(),
                "health_check": await self.iteration_engine.should_continue_autonomous_execution()
            }
            
        except Exception as e:
            return {"error": str(e), "executor_state": self.state.value}
