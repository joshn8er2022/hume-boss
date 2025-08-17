
"""
Core DSPY Signatures for autonomous decision-making engine
"""

import dspy
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SystemContext(BaseModel):
    """Context for system decision making"""
    current_state: str
    historical_states: List[Dict[str, Any]]
    active_agents: List[Dict[str, Any]]
    pending_tasks: List[Dict[str, Any]]
    system_metrics: Dict[str, Any]
    mcp_servers_status: Dict[str, Any]
    recent_errors: List[str]
    last_iteration_outcome: Optional[str] = None


class DecisionOutput(BaseModel):
    """Output structure for autonomous decisions"""
    decision_type: str = Field(..., description="Type of decision made")
    action_plan: List[str] = Field(..., description="Sequence of actions to execute")
    reasoning: str = Field(..., description="Detailed reasoning behind the decision")
    priority_level: int = Field(..., description="Priority level (1-5, 1 being highest)")
    expected_outcome: str = Field(..., description="Expected result of this decision")
    risk_assessment: str = Field(..., description="Assessment of risks and mitigation strategies")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Required resources")
    success_metrics: List[str] = Field(default_factory=list, description="How to measure success")
    estimated_duration: Optional[str] = None
    agent_assignments: List[Dict[str, str]] = Field(default_factory=list)


class AutonomousDecisionSignature(dspy.Signature):
    """
    Core DSPY signature for autonomous decision-making.
    This is the brain of the system that makes all strategic decisions.
    """
    
    context: SystemContext = dspy.InputField(description="Complete system context including state, agents, tasks, and metrics")
    historical_pattern_analysis: str = dspy.InputField(description="Analysis of historical patterns and trends")
    current_objectives: List[str] = dspy.InputField(description="Current system objectives and goals")
    
    decision: DecisionOutput = dspy.OutputField(description="The autonomous decision with full plan and reasoning")
    next_state_prediction: str = dspy.OutputField(description="Predicted next system state after decision execution")
    confidence_score: float = dspy.OutputField(description="Confidence in this decision (0.0-1.0)")


class PreprocessingContext(BaseModel):
    """Context for iteration preprocessing"""
    previous_iteration_results: Optional[Dict[str, Any]] = None
    system_state_changes: List[str] = Field(default_factory=list)
    new_data_available: List[str] = Field(default_factory=list)
    external_triggers: List[str] = Field(default_factory=list)
    error_reports: List[str] = Field(default_factory=list)


class PreprocessingOutput(BaseModel):
    """Output of preprocessing for next iteration"""
    processed_context: Dict[str, Any] = Field(..., description="Enhanced context for decision making")
    priority_focus_areas: List[str] = Field(..., description="Areas requiring immediate attention")
    data_insights: List[str] = Field(..., description="Key insights from data analysis")
    recommended_approach: str = Field(..., description="Recommended approach for this iteration")
    preparation_actions: List[str] = Field(default_factory=list)


class IterationPreprocessingSignature(dspy.Signature):
    """
    DSPY signature for preprocessing before each iteration.
    Analyzes previous iteration results and prepares context for decision making.
    """
    
    preprocessing_context: PreprocessingContext = dspy.InputField(description="Context from previous iteration and new data")
    system_state: Dict[str, Any] = dspy.InputField(description="Current system state")
    retrieval_results: List[str] = dspy.InputField(description="Results from DSPY retriever")
    
    preprocessing_output: PreprocessingOutput = dspy.OutputField(description="Processed and enhanced context for decision making")
    readiness_status: str = dspy.OutputField(description="System readiness for next iteration")


class NextIterationContext(BaseModel):
    """Context for planning next iteration"""
    current_iteration_results: Dict[str, Any]
    execution_success: bool
    error_details: Optional[List[str]] = None
    performance_metrics: Dict[str, Any]
    time_elapsed: float


class NextIterationOutput(BaseModel):
    """Output for next iteration planning"""
    next_iteration_plan: Dict[str, Any] = Field(..., description="Detailed plan for next iteration")
    adjustments_needed: List[str] = Field(..., description="Adjustments based on current results")
    learning_insights: List[str] = Field(..., description="Key learnings from this iteration")
    optimization_suggestions: List[str] = Field(default_factory=list)
    continuation_strategy: str = Field(..., description="Strategy for continuing the autonomous process")


class NextIterationSignature(dspy.Signature):
    """
    DSPY signature for planning the next iteration based on current results.
    This runs in the 'finally' block and prepares for the next cycle.
    """
    
    iteration_context: NextIterationContext = dspy.InputField(description="Results and context from current iteration")
    system_feedback: List[str] = dspy.InputField(description="Feedback from system components")
    
    next_iteration_output: NextIterationOutput = dspy.OutputField(description="Plan and preparations for next iteration")
    continuation_decision: str = dspy.OutputField(description="Decision on whether and how to continue")
