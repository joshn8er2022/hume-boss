
"""
Core state holder for autonomous DSPY Boss system
Manages current state and provides interface for historical access
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field
from loguru import logger


class AutonomousState(BaseModel):
    """
    Complete autonomous state representation
    Contains all information needed for decision making
    """
    
    # Core identification
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iteration_number: int = Field(..., description="Sequential iteration number")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # System state
    boss_state: str = Field(..., description="Current boss state")
    system_phase: str = Field(..., description="Current system phase (preprocessing, deciding, executing, etc.)")
    
    # Agent hierarchy (Boss = Agent 0, subordinates = Agent 1, 2, 3...)
    agents: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="All agents by ID")
    agent_hierarchy: List[str] = Field(default_factory=list, description="Agent IDs in hierarchy order")
    active_agents: List[str] = Field(default_factory=list, description="Currently active agent IDs")
    
    # Task management
    tasks: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="All tasks by ID")
    task_queue: List[str] = Field(default_factory=list, description="Task IDs in queue")
    active_tasks: List[str] = Field(default_factory=list, description="Currently executing task IDs")
    
    # Decision context
    last_decision: Optional[Dict[str, Any]] = None
    decision_reasoning: Optional[str] = None
    confidence_score: Optional[float] = None
    
    # Performance metrics
    metrics: Dict[str, Any] = Field(default_factory=dict)
    success_indicators: Dict[str, float] = Field(default_factory=dict)
    error_count: int = 0
    warnings: List[str] = Field(default_factory=list)
    
    # MCP and external connections
    mcp_servers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    external_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Forecasting and planning
    predicted_next_state: Optional[str] = None
    forecast_confidence: Optional[float] = None
    planned_actions: List[str] = Field(default_factory=list)
    
    # Learning and reflection
    insights_learned: List[str] = Field(default_factory=list)
    patterns_identified: List[str] = Field(default_factory=list)
    improvement_actions: List[str] = Field(default_factory=list)
    
    # Execution results
    execution_success: Optional[bool] = None
    execution_errors: List[str] = Field(default_factory=list)
    execution_duration: Optional[float] = None
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class StateSnapshot(BaseModel):
    """
    Lightweight snapshot of state for quick access
    """
    state_id: str
    iteration_number: int
    timestamp: datetime
    boss_state: str
    system_phase: str
    agent_count: int
    active_agent_count: int
    task_count: int
    active_task_count: int
    success_rate: Optional[float] = None
    error_count: int = 0
    key_metrics: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class StateHolder:
    """
    Core state holder managing current state and providing historical access
    """
    
    def __init__(self, max_recent_states: int = 100):
        self.max_recent_states = max_recent_states
        
        # Current state
        self.current_state: Optional[AutonomousState] = None
        
        # Recent states for quick access (last 100 by default)
        self.recent_states: List[AutonomousState] = []
        self.state_snapshots: List[StateSnapshot] = []
        
        # State index for fast lookup
        self.state_index: Dict[str, int] = {}
        self.iteration_index: Dict[int, str] = {}
        
        # Locks for thread safety
        self._state_lock = asyncio.Lock()
        
        # Observers for state changes
        self._observers: List[callable] = []
        
        logger.info(f"StateHolder initialized with max_recent_states={max_recent_states}")
    
    async def initialize_state(self, initial_config: Dict[str, Any] = None) -> AutonomousState:
        """Initialize the first autonomous state"""
        async with self._state_lock:
            initial_state = AutonomousState(
                iteration_number=0,
                boss_state="INITIALIZING",
                system_phase="initialization",
                **(initial_config or {})
            )
            
            self.current_state = initial_state
            await self._add_to_recent_states(initial_state)
            await self._notify_observers("state_initialized", initial_state)
            
            logger.info(f"State initialized: {initial_state.state_id}")
            return initial_state
    
    async def update_state(self, updates: Dict[str, Any]) -> AutonomousState:
        """Update current state with new information"""
        async with self._state_lock:
            if not self.current_state:
                raise ValueError("No current state to update. Call initialize_state() first.")
            
            # Create new state based on current state with updates
            current_dict = self.current_state.dict()
            
            # Deep merge updates
            for key, value in updates.items():
                if isinstance(value, dict) and key in current_dict and isinstance(current_dict[key], dict):
                    current_dict[key].update(value)
                else:
                    current_dict[key] = value
            
            # Increment iteration number if this is a new iteration
            if updates.get("new_iteration", False):
                current_dict["iteration_number"] += 1
                current_dict["timestamp"] = datetime.utcnow()
                current_dict["state_id"] = str(uuid.uuid4())
            
            # Create new state
            new_state = AutonomousState(**current_dict)
            self.current_state = new_state
            
            await self._add_to_recent_states(new_state)
            await self._notify_observers("state_updated", new_state)
            
            logger.debug(f"State updated: {new_state.state_id}")
            return new_state
    
    async def new_iteration(self, phase: str = "preprocessing") -> AutonomousState:
        """Start a new iteration with a fresh state"""
        if not self.current_state:
            return await self.initialize_state()
        
        return await self.update_state({
            "new_iteration": True,
            "system_phase": phase,
            "execution_success": None,
            "execution_errors": [],
            "execution_duration": None,
            "last_decision": None,
            "decision_reasoning": None,
            "confidence_score": None
        })
    
    async def get_current_state(self) -> Optional[AutonomousState]:
        """Get current state"""
        return self.current_state
    
    async def get_recent_states(self, count: int = None) -> List[AutonomousState]:
        """Get recent states (up to count, or all recent states)"""
        if count is None:
            return self.recent_states.copy()
        return self.recent_states[-count:] if count > 0 else []
    
    async def get_state_by_id(self, state_id: str) -> Optional[AutonomousState]:
        """Get state by ID from recent states"""
        for state in self.recent_states:
            if state.state_id == state_id:
                return state
        return None
    
    async def get_state_by_iteration(self, iteration_number: int) -> Optional[AutonomousState]:
        """Get state by iteration number from recent states"""
        for state in self.recent_states:
            if state.iteration_number == iteration_number:
                return state
        return None
    
    async def get_states_by_phase(self, phase: str, count: int = 10) -> List[AutonomousState]:
        """Get states filtered by system phase"""
        matching_states = [
            state for state in self.recent_states 
            if state.system_phase == phase
        ]
        return matching_states[-count:] if count > 0 else matching_states
    
    async def get_state_snapshots(self, count: int = None) -> List[StateSnapshot]:
        """Get lightweight state snapshots"""
        if count is None:
            return self.state_snapshots.copy()
        return self.state_snapshots[-count:] if count > 0 else []
    
    async def get_performance_history(self, metric: str, count: int = 20) -> List[tuple]:
        """Get history of a specific performance metric"""
        history = []
        for state in self.recent_states[-count:]:
            if metric in state.metrics:
                history.append((state.timestamp, state.metrics[metric]))
            elif metric in state.success_indicators:
                history.append((state.timestamp, state.success_indicators[metric]))
        return history
    
    async def add_observer(self, callback: callable):
        """Add observer for state changes"""
        self._observers.append(callback)
    
    async def remove_observer(self, callback: callable):
        """Remove observer"""
        if callback in self._observers:
            self._observers.remove(callback)
    
    async def _add_to_recent_states(self, state: AutonomousState):
        """Add state to recent states with size management"""
        self.recent_states.append(state)
        
        # Create snapshot
        snapshot = StateSnapshot(
            state_id=state.state_id,
            iteration_number=state.iteration_number,
            timestamp=state.timestamp,
            boss_state=state.boss_state,
            system_phase=state.system_phase,
            agent_count=len(state.agents),
            active_agent_count=len(state.active_agents),
            task_count=len(state.tasks),
            active_task_count=len(state.active_tasks),
            success_rate=state.success_indicators.get("success_rate"),
            error_count=state.error_count,
            key_metrics={k: v for k, v in state.metrics.items() if isinstance(v, (int, float))}
        )
        self.state_snapshots.append(snapshot)
        
        # Update indexes
        self.state_index[state.state_id] = len(self.recent_states) - 1
        self.iteration_index[state.iteration_number] = state.state_id
        
        # Maintain size limit
        if len(self.recent_states) > self.max_recent_states:
            removed_state = self.recent_states.pop(0)
            self.state_snapshots.pop(0)
            
            # Update indexes
            del self.state_index[removed_state.state_id]
            if removed_state.iteration_number in self.iteration_index:
                del self.iteration_index[removed_state.iteration_number]
            
            # Adjust remaining indexes
            for state_id in self.state_index:
                self.state_index[state_id] -= 1
    
    async def _notify_observers(self, event: str, state: AutonomousState):
        """Notify all observers of state changes"""
        for observer in self._observers:
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer(event, state)
                else:
                    observer(event, state)
            except Exception as e:
                logger.warning(f"Observer notification failed: {e}")
    
    async def export_state_history(self, format: str = "json") -> str:
        """Export state history in specified format"""
        if format == "json":
            return json.dumps([state.dict() for state in self.recent_states], default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health based on recent states"""
        if not self.recent_states:
            return {"status": "no_data", "message": "No states available"}
        
        recent_states = self.recent_states[-10:]  # Last 10 states
        
        # Calculate metrics
        error_count = sum(len(state.execution_errors) for state in recent_states)
        success_count = sum(1 for state in recent_states if state.execution_success is True)
        total_decisions = len([s for s in recent_states if s.last_decision is not None])
        
        avg_confidence = None
        if total_decisions > 0:
            confidence_scores = [s.confidence_score for s in recent_states if s.confidence_score is not None]
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            "status": "healthy" if error_count < 3 else "warning" if error_count < 6 else "critical",
            "recent_error_count": error_count,
            "recent_success_count": success_count, 
            "total_iterations": self.current_state.iteration_number if self.current_state else 0,
            "average_confidence": avg_confidence,
            "current_phase": self.current_state.system_phase if self.current_state else None,
            "agent_count": len(self.current_state.agents) if self.current_state else 0,
            "active_task_count": len(self.current_state.active_tasks) if self.current_state else 0
        }
