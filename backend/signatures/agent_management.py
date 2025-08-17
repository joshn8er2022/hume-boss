
"""
DSPY Signatures for autonomous agent management and spawning
"""

import dspy
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class AgentSpawningContext(BaseModel):
    """Context for agent spawning decisions"""
    current_workload: Dict[str, Any]
    existing_agents: List[Dict[str, Any]]
    task_queue_analysis: Dict[str, Any]
    system_capacity: Dict[str, Any]
    performance_requirements: Dict[str, Any]


class NewAgentSpec(BaseModel):
    """Specification for a new agent"""
    agent_name: str = Field(..., description="Human-readable name for the agent")
    agent_type: str = Field(..., description="Type/role of the agent")
    capabilities: List[str] = Field(..., description="Capabilities this agent should have")
    specialization: str = Field(..., description="Primary specialization area")
    model_config: Dict[str, Any] = Field(..., description="LLM model configuration")
    max_concurrent_tasks: int = Field(default=3, description="Maximum concurrent tasks")
    priority_level: int = Field(default=3, description="Agent priority level (1-5)")
    resource_allocation: Dict[str, Any] = Field(default_factory=dict)


class AgentSpawningOutput(BaseModel):
    """Output of agent spawning decision"""
    should_spawn: bool = Field(..., description="Whether to spawn new agents")
    agent_specifications: List[NewAgentSpec] = Field(default_factory=list)
    spawning_reasoning: str = Field(..., description="Reasoning behind spawning decision")
    expected_benefits: List[str] = Field(..., description="Expected benefits from new agents")
    resource_impact: str = Field(..., description="Impact on system resources")
    implementation_plan: List[str] = Field(default_factory=list)


class AgentSpawningSignature(dspy.Signature):
    """
    DSPY signature for autonomous agent spawning decisions.
    The Boss agent (Agent 0) uses this to decide when and what kind of subordinate agents to spawn.
    """
    
    spawning_context: AgentSpawningContext = dspy.InputField(description="Context for agent spawning decision")
    system_requirements: Dict[str, Any] = dspy.InputField(description="Current system requirements")
    boss_strategy: str = dspy.InputField(description="Current Boss strategic direction")
    
    spawning_decision: AgentSpawningOutput = dspy.OutputField(description="Decision on agent spawning with specifications")
    hierarchy_impact: str = dspy.OutputField(description="Impact on agent hierarchy and numbering")


class TaskDelegationContext(BaseModel):
    """Context for task delegation"""
    available_tasks: List[Dict[str, Any]]
    agent_capabilities: Dict[str, List[str]]
    agent_current_load: Dict[str, int]
    task_priorities: Dict[str, int]
    deadline_constraints: Dict[str, str]


class TaskAssignment(BaseModel):
    """Individual task assignment"""
    task_id: str = Field(..., description="Task identifier")
    assigned_agent_id: str = Field(..., description="Agent assigned to this task")
    assignment_reasoning: str = Field(..., description="Why this agent was chosen")
    expected_completion_time: str = Field(..., description="Expected completion timeframe")
    success_criteria: List[str] = Field(..., description="Criteria for successful completion")
    monitoring_checkpoints: List[str] = Field(default_factory=list)


class TaskDelegationOutput(BaseModel):
    """Output of task delegation process"""
    task_assignments: List[TaskAssignment] = Field(..., description="All task assignments")
    delegation_strategy: str = Field(..., description="Overall delegation strategy used")
    load_balancing_rationale: str = Field(..., description="How load balancing was achieved")
    coordination_requirements: List[str] = Field(..., description="Inter-agent coordination needed")
    monitoring_plan: str = Field(..., description="Plan for monitoring task execution")
    escalation_triggers: List[str] = Field(default_factory=list)


class TaskDelegationSignature(dspy.Signature):
    """
    DSPY signature for intelligent task delegation to subordinate agents.
    Boss agent uses this to optimally distribute work among Agent 1, 2, 3, etc.
    """
    
    delegation_context: TaskDelegationContext = dspy.InputField(description="Context for task delegation")
    optimization_goals: List[str] = dspy.InputField(description="Goals for optimal delegation")
    
    delegation_output: TaskDelegationOutput = dspy.OutputField(description="Optimal task delegation plan")
    performance_prediction: Dict[str, float] = dspy.OutputField(description="Predicted performance metrics")
