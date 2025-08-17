
"""
DSPY Signature classes for autonomous decision-making engine
"""

from .decision_engine import AutonomousDecisionSignature, IterationPreprocessingSignature, NextIterationSignature
from .state_forecasting import StateForecastingSignature, HistoricalAnalysisSignature
from .agent_management import AgentSpawningSignature, TaskDelegationSignature
from .system_reflection import SystemReflectionSignature, ImprovementSignature

__all__ = [
    "AutonomousDecisionSignature",
    "IterationPreprocessingSignature", 
    "NextIterationSignature",
    "StateForecastingSignature",
    "HistoricalAnalysisSignature",
    "AgentSpawningSignature",
    "TaskDelegationSignature",
    "SystemReflectionSignature",
    "ImprovementSignature"
]
