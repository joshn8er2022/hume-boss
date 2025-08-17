
"""
DSPY Signatures for system reflection and continuous improvement
"""

import dspy
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ReflectionContext(BaseModel):
    """Context for system reflection"""
    recent_performance: Dict[str, Any]
    completed_tasks: List[Dict[str, Any]]
    failed_tasks: List[Dict[str, Any]]
    agent_performance: Dict[str, Dict[str, Any]]
    system_metrics: Dict[str, Any]
    user_feedback: List[str]


class ReflectionInsight(BaseModel):
    """Individual reflection insight"""
    insight_type: str = Field(..., description="Type of insight (performance, efficiency, strategy, etc.)")
    description: str = Field(..., description="Description of the insight")
    evidence: List[str] = Field(..., description="Evidence supporting this insight")
    impact_level: str = Field(..., description="Impact level (high/medium/low)")
    actionability: str = Field(..., description="How actionable this insight is")
    related_components: List[str] = Field(default_factory=list)


class SystemReflectionOutput(BaseModel):
    """Output of system reflection"""
    key_insights: List[ReflectionInsight] = Field(..., description="Key insights from reflection")
    performance_assessment: str = Field(..., description="Overall performance assessment")
    strength_areas: List[str] = Field(..., description="Areas where system performed well")
    improvement_areas: List[str] = Field(..., description="Areas needing improvement")
    pattern_recognition: List[str] = Field(..., description="Patterns recognized in performance")
    learning_summary: str = Field(..., description="Summary of key learnings")
    confidence_rating: float = Field(..., description="Confidence in reflection accuracy (0.0-1.0)")


class SystemReflectionSignature(dspy.Signature):
    """
    DSPY signature for deep system reflection and learning.
    Analyzes recent performance and extracts insights for improvement.
    """
    
    reflection_context: ReflectionContext = dspy.InputField(description="Context for system reflection")
    reflection_focus: List[str] = dspy.InputField(description="Specific areas to focus reflection on")
    historical_baselines: Dict[str, Any] = dspy.InputField(description="Historical performance baselines")
    
    reflection_output: SystemReflectionOutput = dspy.OutputField(description="Comprehensive reflection results")
    meta_insights: List[str] = dspy.OutputField(description="Meta-insights about the reflection process itself")


class ImprovementAction(BaseModel):
    """Specific improvement action"""
    action_type: str = Field(..., description="Type of improvement action")
    description: str = Field(..., description="Detailed description of the action")
    priority: int = Field(..., description="Priority level (1-5, 1 being highest)")
    estimated_effort: str = Field(..., description="Estimated effort required")
    expected_impact: str = Field(..., description="Expected impact on system performance")
    implementation_steps: List[str] = Field(..., description="Steps to implement this action")
    success_metrics: List[str] = Field(..., description="How to measure success")
    timeline: str = Field(..., description="Expected implementation timeline")
    dependencies: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)


class ImprovementOutput(BaseModel):
    """Output of improvement planning"""
    improvement_actions: List[ImprovementAction] = Field(..., description="Prioritized improvement actions")
    implementation_roadmap: str = Field(..., description="Overall roadmap for improvements")
    resource_requirements: Dict[str, Any] = Field(..., description="Resources needed for improvements")
    expected_outcomes: Dict[str, str] = Field(..., description="Expected outcomes from improvements")
    monitoring_framework: str = Field(..., description="Framework for monitoring improvement progress")
    rollback_plans: List[str] = Field(default_factory=list)


class ImprovementSignature(dspy.Signature):
    """
    DSPY signature for generating system improvement plans.
    Converts reflection insights into actionable improvement strategies.
    """
    
    reflection_insights: List[ReflectionInsight] = dspy.InputField(description="Insights from system reflection")
    system_constraints: Dict[str, Any] = dspy.InputField(description="System constraints and limitations")
    improvement_goals: List[str] = dspy.InputField(description="Specific improvement goals")
    
    improvement_plan: ImprovementOutput = dspy.OutputField(description="Comprehensive improvement plan")
    implementation_readiness: str = dspy.OutputField(description="Assessment of readiness to implement changes")
