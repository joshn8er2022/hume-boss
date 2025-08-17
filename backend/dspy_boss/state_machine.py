
"""
State machine implementation for the DSPY Boss
"""

from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from loguru import logger
from .models import BossStateData, ReportEntry


class BossState(str, Enum):
    """Boss state machine states"""
    IDLE = "idle"
    AWAKE = "awake"
    RESTART = "restart"
    STOP = "stop"
    EXECUTING = "executing"
    RESEARCHING = "researching"
    THINKING = "thinking"
    RETHINK = "rethink"
    REFLECTING = "reflecting"


class StateTransition:
    """Manages state transitions and validation"""
    
    # Valid state transitions
    VALID_TRANSITIONS: Dict[BossState, List[BossState]] = {
        BossState.IDLE: [BossState.AWAKE, BossState.STOP, BossState.RESTART],
        BossState.AWAKE: [BossState.THINKING, BossState.EXECUTING, BossState.RESEARCHING, BossState.IDLE, BossState.STOP],
        BossState.THINKING: [BossState.EXECUTING, BossState.RESEARCHING, BossState.RETHINK, BossState.REFLECTING, BossState.IDLE],
        BossState.RETHINK: [BossState.THINKING, BossState.EXECUTING, BossState.REFLECTING, BossState.IDLE],
        BossState.EXECUTING: [BossState.THINKING, BossState.REFLECTING, BossState.IDLE, BossState.AWAKE],
        BossState.RESEARCHING: [BossState.THINKING, BossState.EXECUTING, BossState.REFLECTING, BossState.IDLE],
        BossState.REFLECTING: [BossState.THINKING, BossState.IDLE, BossState.AWAKE],
        BossState.RESTART: [BossState.IDLE, BossState.AWAKE],
        BossState.STOP: []  # Terminal state
    }
    
    def __init__(self):
        self.current_state: BossState = BossState.IDLE
        self.previous_state: Optional[BossState] = None
        self.state_history: List[tuple[BossState, datetime]] = []
        self.state_data: BossStateData = BossStateData()
        self.transition_callbacks: Dict[BossState, List[Callable]] = {}
        
    def can_transition_to(self, new_state: BossState) -> bool:
        """Check if transition to new state is valid"""
        return new_state in self.VALID_TRANSITIONS.get(self.current_state, [])
    
    def transition_to(self, new_state: BossState, reason: Optional[str] = None) -> bool:
        """Attempt to transition to new state"""
        if not self.can_transition_to(new_state):
            logger.warning(f"Invalid state transition from {self.current_state} to {new_state}")
            return False
        
        # Record transition
        self.previous_state = self.current_state
        self.state_history.append((self.current_state, datetime.utcnow()))
        
        logger.info(f"State transition: {self.current_state} -> {new_state}" + 
                   (f" (reason: {reason})" if reason else ""))
        
        self.current_state = new_state
        
        # Execute callbacks for new state
        self._execute_callbacks(new_state)
        
        return True
    
    def add_transition_callback(self, state: BossState, callback: Callable):
        """Add callback to execute when entering a state"""
        if state not in self.transition_callbacks:
            self.transition_callbacks[state] = []
        self.transition_callbacks[state].append(callback)
    
    def _execute_callbacks(self, state: BossState):
        """Execute callbacks for the given state"""
        callbacks = self.transition_callbacks.get(state, [])
        for callback in callbacks:
            try:
                callback(state, self.state_data)
            except Exception as e:
                logger.error(f"Error executing callback for state {state}: {e}")
    
    def get_state_duration(self) -> float:
        """Get duration in current state (seconds)"""
        if not self.state_history:
            return 0.0
        
        last_transition = self.state_history[-1][1]
        return (datetime.utcnow() - last_transition).total_seconds()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state"""
        return {
            "current_state": self.current_state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "duration_seconds": self.get_state_duration(),
            "transition_count": len(self.state_history),
            "workload": self.state_data.current_workload,
            "active_agents": len(self.state_data.active_agents),
            "pending_tasks": len(self.state_data.pending_tasks)
        }
    
    def should_transition_based_on_workload(self) -> Optional[BossState]:
        """Suggest state transition based on current workload"""
        workload = self.state_data.current_workload
        pending_tasks = len(self.state_data.pending_tasks)
        
        if self.current_state == BossState.IDLE:
            if pending_tasks > 0:
                return BossState.AWAKE
        
        elif self.current_state == BossState.AWAKE:
            if pending_tasks == 0:
                return BossState.IDLE
            elif workload > 10:  # High workload threshold
                return BossState.THINKING
            else:
                return BossState.EXECUTING
        
        elif self.current_state == BossState.EXECUTING:
            if workload == 0:
                return BossState.REFLECTING
            elif workload > 15:  # Very high workload
                return BossState.THINKING
        
        elif self.current_state == BossState.THINKING:
            if workload < 5:  # Low workload
                return BossState.EXECUTING
            elif len(self.state_data.system_errors) > 0:
                return BossState.RESEARCHING
        
        elif self.current_state == BossState.RESEARCHING:
            if len(self.state_data.system_errors) == 0:
                return BossState.THINKING
        
        elif self.current_state == BossState.REFLECTING:
            # After reflection, decide next action
            if pending_tasks > 0:
                return BossState.AWAKE
            else:
                return BossState.IDLE
        
        return None
    
    def force_transition(self, new_state: BossState, reason: str = "Forced transition"):
        """Force transition regardless of validity (use with caution)"""
        logger.warning(f"Forcing state transition from {self.current_state} to {new_state}: {reason}")
        
        self.previous_state = self.current_state
        self.state_history.append((self.current_state, datetime.utcnow()))
        self.current_state = new_state
        
        self._execute_callbacks(new_state)
    
    def reset_to_idle(self):
        """Reset state machine to idle state"""
        logger.info("Resetting state machine to IDLE")
        self.force_transition(BossState.IDLE, "System reset")
        
        # Clear state data
        self.state_data = BossStateData()
    
    def is_terminal_state(self) -> bool:
        """Check if current state is terminal (STOP)"""
        return self.current_state == BossState.STOP
    
    def get_next_valid_states(self) -> List[BossState]:
        """Get list of valid next states from current state"""
        return self.VALID_TRANSITIONS.get(self.current_state, [])


class StateMachineManager:
    """High-level manager for the state machine"""
    
    def __init__(self):
        self.transition = StateTransition()
        self.auto_transition_enabled = True
        self.transition_interval = 5.0  # seconds
    
    @property
    def current_state(self) -> BossState:
        """Get current state"""
        return self.transition.current_state
    
    def get_state_data(self) -> Dict[str, Any]:
        """Get current state data"""
        return self.transition.state_data.model_dump()
        
    def setup_default_callbacks(self):
        """Setup default callbacks for state transitions"""
        
        def on_awake(state: BossState, data: BossStateData):
            logger.info("Boss is now awake - assessing workload")
            
        def on_thinking(state: BossState, data: BossStateData):
            logger.info("Boss is thinking - analyzing tasks and resources")
            
        def on_executing(state: BossState, data: BossStateData):
            logger.info("Boss is executing - managing task assignments")
            
        def on_researching(state: BossState, data: BossStateData):
            logger.info("Boss is researching - investigating issues")
            
        def on_reflecting(state: BossState, data: BossStateData):
            logger.info("Boss is reflecting - analyzing performance")
            data.last_reflection = datetime.utcnow()
            
        def on_idle(state: BossState, data: BossStateData):
            logger.info("Boss is idle - waiting for tasks")
            
        # Register callbacks
        self.transition.add_transition_callback(BossState.AWAKE, on_awake)
        self.transition.add_transition_callback(BossState.THINKING, on_thinking)
        self.transition.add_transition_callback(BossState.EXECUTING, on_executing)
        self.transition.add_transition_callback(BossState.RESEARCHING, on_researching)
        self.transition.add_transition_callback(BossState.REFLECTING, on_reflecting)
        self.transition.add_transition_callback(BossState.IDLE, on_idle)
    
    def update_workload(self, workload: int):
        """Update current workload and trigger auto-transition if enabled"""
        self.transition.state_data.current_workload = workload
        
        if self.auto_transition_enabled:
            suggested_state = self.transition.should_transition_based_on_workload()
            if suggested_state and suggested_state != self.transition.current_state:
                self.transition.transition_to(suggested_state, "Workload-based auto-transition")
    
    def handle_error(self, error_message: str):
        """Handle system error and update state accordingly"""
        self.transition.state_data.system_errors.append(error_message)
        
        # If not already researching or in a critical state, transition to researching
        if self.transition.current_state not in [BossState.RESEARCHING, BossState.STOP, BossState.RESTART]:
            if self.transition.can_transition_to(BossState.RESEARCHING):
                self.transition.transition_to(BossState.RESEARCHING, f"Error occurred: {error_message}")
    
    def clear_errors(self):
        """Clear system errors"""
        self.transition.state_data.system_errors.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the state machine"""
        return {
            "state_summary": self.transition.get_state_summary(),
            "auto_transition_enabled": self.auto_transition_enabled,
            "valid_next_states": [s.value for s in self.transition.get_next_valid_states()],
            "is_terminal": self.transition.is_terminal_state(),
            "state_data": self.transition.state_data.model_dump()
        }
