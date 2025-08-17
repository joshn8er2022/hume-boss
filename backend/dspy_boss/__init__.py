
"""
DSPY Boss System - Autonomous Agent Management Framework
"""

__version__ = "1.0.0"
__author__ = "DSPY Boss System"

from .models import *
from .state_machine import BossState, StateTransition
from .config import DSPYBossConfig
from .boss import DSPYBoss

__all__ = [
    "DSPYBoss",
    "DSPYBossConfig", 
    "BossState",
    "StateTransition",
    "AgentConfig",
    "TaskDefinition",
    "MCPServerConfig",
    "ReportEntry",
    "FailureEntry"
]
