
"""
Agent hierarchy system with Boss as Agent 0 and subordinates as Agent 1, 2, 3...
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger

import dspy

from ..signatures.agent_management import AgentSpawningSignature, TaskDelegationSignature
from ..state.state_holder import StateHolder


class AgentRole(Enum):
    """Agent roles in the hierarchy"""
    BOSS = "boss"                    # Agent 0 - The decision maker
    SPECIALIST = "specialist"        # Agents 1+ - Specialized workers
    GENERALIST = "generalist"        # Agents 1+ - General purpose workers
    COORDINATOR = "coordinator"      # Agents 1+ - Task coordinators
    RESEARCHER = "researcher"        # Agents 1+ - Research specialists
    EXECUTOR = "executor"           # Agents 1+ - Task executors


class AgentStatus(Enum):
    """Status of an agent"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy" 
    IDLE = "idle"
    ERROR = "error"
    PAUSED = "paused"
    TERMINATED = "terminated"


class AgentCapability(BaseModel):
    """Capability definition for an agent"""
    
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    skill_level: float = Field(default=0.5, description="Skill level 0.0-1.0")
    requirements: List[str] = Field(default_factory=list, description="Requirements for this capability")
    enabled: bool = Field(default=True, description="Whether capability is enabled")


class AgentMetrics(BaseModel):
    """Performance metrics for an agent"""
    
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_duration: float = 0.0
    success_rate: float = 1.0
    last_active: Optional[datetime] = None
    total_runtime: float = 0.0
    error_count: int = 0
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class BaseAgent(BaseModel):
    """Base agent class"""
    
    # Identity
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_number: int = Field(..., description="Agent number (0 for Boss, 1+ for subordinates)")
    display_name: str = Field(..., description="Human-readable agent name")
    role: AgentRole = Field(..., description="Agent role")
    
    # Configuration
    capabilities: List[AgentCapability] = Field(default_factory=list)
    max_concurrent_tasks: int = Field(default=3, description="Maximum concurrent tasks")
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    
    # State
    status: AgentStatus = Field(default=AgentStatus.INITIALIZING)
    current_tasks: List[str] = Field(default_factory=list, description="Currently assigned task IDs")
    
    # Hierarchy relationships
    superior_agent_id: Optional[str] = None
    subordinate_agent_ids: List[str] = Field(default_factory=list)
    
    # Performance tracking
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
    
    def get_human_readable_name(self) -> str:
        """Get human-readable name for UI display"""
        if self.agent_number == 0:
            return "Boss Agent"
        else:
            return f"Agent {self.agent_number}"
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks"""
        return (
            self.status in [AgentStatus.ACTIVE, AgentStatus.IDLE] and
            len(self.current_tasks) < self.max_concurrent_tasks
        )
    
    def get_capability_names(self) -> List[str]:
        """Get list of capability names"""
        return [cap.name for cap in self.capabilities if cap.enabled]
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability"""
        return any(
            cap.name == capability_name and cap.enabled 
            for cap in self.capabilities
        )
    
    def get_load_percentage(self) -> float:
        """Get current load as percentage"""
        if self.max_concurrent_tasks == 0:
            return 0.0
        return (len(self.current_tasks) / self.max_concurrent_tasks) * 100


class BossAgent(BaseAgent):
    """
    Boss Agent (Agent 0) - The supreme decision maker
    Manages all subordinate agents and makes strategic decisions
    """
    
    def __init__(self, **data):
        super().__init__(
            agent_number=0,
            role=AgentRole.BOSS,
            display_name="Boss Agent",
            **data
        )
    
    # Strategic decision-making methods
    spawning_decisions_made: int = 0
    delegation_decisions_made: int = 0
    strategic_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    
    def record_spawning_decision(self, decision: Dict[str, Any]):
        """Record a spawning decision made by the Boss"""
        self.spawning_decisions_made += 1
        self.strategic_decisions.append({
            "type": "spawning",
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision
        })
        
        # Keep only last 50 decisions
        if len(self.strategic_decisions) > 50:
            self.strategic_decisions.pop(0)
    
    def record_delegation_decision(self, decision: Dict[str, Any]):
        """Record a delegation decision made by the Boss"""
        self.delegation_decisions_made += 1
        self.strategic_decisions.append({
            "type": "delegation", 
            "timestamp": datetime.utcnow().isoformat(),
            "decision": decision
        })
        
        # Keep only last 50 decisions
        if len(self.strategic_decisions) > 50:
            self.strategic_decisions.pop(0)


class SubordinateAgent(BaseAgent):
    """
    Subordinate Agent (Agent 1, 2, 3...) - Specialized workers
    Report to Boss Agent and execute specific tasks
    """
    
    def __init__(self, agent_number: int, **data):
        if agent_number < 1:
            raise ValueError("Subordinate agents must have number >= 1")
        
        super().__init__(
            agent_number=agent_number,
            display_name=f"Agent {agent_number}",
            superior_agent_id="boss_agent_0",  # Always report to Boss
            **data
        )
    
    # Task execution tracking
    specialization_area: Optional[str] = None
    learning_enabled: bool = True
    adaptation_rate: float = 0.1
    
    def set_specialization(self, area: str):
        """Set agent specialization area"""
        self.specialization_area = area
        logger.info(f"Agent {self.agent_number} specialized in {area}")
    
    def adapt_capabilities(self, task_feedback: Dict[str, Any]):
        """Adapt capabilities based on task feedback"""
        if not self.learning_enabled:
            return
        
        # Simple capability adaptation based on success
        if task_feedback.get("success", False):
            # Boost capabilities used in successful tasks
            for cap_name in task_feedback.get("capabilities_used", []):
                for cap in self.capabilities:
                    if cap.name == cap_name:
                        cap.skill_level = min(1.0, cap.skill_level + self.adaptation_rate * 0.1)
        else:
            # Slightly decrease capabilities that failed
            for cap_name in task_feedback.get("capabilities_used", []):
                for cap in self.capabilities:
                    if cap.name == cap_name:
                        cap.skill_level = max(0.1, cap.skill_level - self.adaptation_rate * 0.05)


class AgentHierarchy:
    """
    Manages the complete agent hierarchy with Boss as Agent 0 and subordinates numbered sequentially
    """
    
    def __init__(self, state_holder: StateHolder):
        self.state_holder = state_holder
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.boss_agent: Optional[BossAgent] = None
        self.subordinate_agents: Dict[int, SubordinateAgent] = {}  # number -> agent
        
        # Agent numbering
        self.next_agent_number = 1  # Boss is 0, subordinates start at 1
        
        # Spawning tracking
        self.spawning_history: List[Dict[str, Any]] = []
        
        # Communication tracking
        self.active_communications: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("AgentHierarchy initialized")
    
    async def initialize_boss_agent(self, boss_config: Dict[str, Any] = None) -> BossAgent:
        """Initialize the Boss Agent (Agent 0)"""
        
        if self.boss_agent is not None:
            logger.warning("Boss agent already exists")
            return self.boss_agent
        
        boss_id = "boss_agent_0"
        
        # Default Boss capabilities
        default_capabilities = [
            AgentCapability(
                name="strategic_planning",
                description="Strategic planning and decision making",
                skill_level=0.9
            ),
            AgentCapability(
                name="agent_management", 
                description="Managing and spawning subordinate agents",
                skill_level=1.0
            ),
            AgentCapability(
                name="task_delegation",
                description="Delegating tasks to subordinate agents", 
                skill_level=0.95
            ),
            AgentCapability(
                name="system_oversight",
                description="Overall system monitoring and control",
                skill_level=0.9
            ),
            AgentCapability(
                name="decision_making",
                description="Autonomous decision making using DSPY",
                skill_level=1.0
            )
        ]
        
        boss_config = boss_config or {}
        boss_config.update({
            "agent_id": boss_id,
            "capabilities": boss_config.get("capabilities", default_capabilities),
            "max_concurrent_tasks": boss_config.get("max_concurrent_tasks", 10),
            "status": AgentStatus.ACTIVE
        })
        
        self.boss_agent = BossAgent(**boss_config)
        self.agents[boss_id] = self.boss_agent
        
        logger.info("Boss Agent (Agent 0) initialized")
        
        # Update state
        await self._update_state_with_agents()
        
        return self.boss_agent
    
    async def spawn_subordinate_agent(
        self,
        role: AgentRole = AgentRole.GENERALIST,
        capabilities: List[AgentCapability] = None,
        specialization: str = None,
        config: Dict[str, Any] = None
    ) -> SubordinateAgent:
        """
        Spawn a new subordinate agent with proper numbering
        
        Args:
            role: Role for the new agent
            capabilities: Specific capabilities for the agent
            specialization: Area of specialization
            config: Additional configuration
        
        Returns:
            The newly created subordinate agent
        """
        
        if not self.boss_agent:
            raise ValueError("Boss agent must be initialized before spawning subordinates")
        
        # Get next agent number
        agent_number = self.next_agent_number
        self.next_agent_number += 1
        
        agent_id = f"agent_{agent_number}"
        
        # Default capabilities based on role
        if capabilities is None:
            capabilities = self._get_default_capabilities_for_role(role)
        
        # Create agent configuration
        agent_config = {
            "agent_id": agent_id,
            "role": role,
            "capabilities": capabilities,
            "max_concurrent_tasks": 3,
            "status": AgentStatus.INITIALIZING,
            **(config or {})
        }
        
        # Create subordinate agent
        subordinate = SubordinateAgent(agent_number=agent_number, **agent_config)
        
        if specialization:
            subordinate.set_specialization(specialization)
        
        # Add to registries
        self.agents[agent_id] = subordinate
        self.subordinate_agents[agent_number] = subordinate
        
        # Update Boss's subordinate list
        self.boss_agent.subordinate_agent_ids.append(agent_id)
        
        # Record spawning decision
        spawning_record = {
            "agent_id": agent_id,
            "agent_number": agent_number,
            "role": role.value,
            "specialization": specialization,
            "timestamp": datetime.utcnow().isoformat(),
            "spawned_by": "boss_agent_0"
        }
        
        self.spawning_history.append(spawning_record)
        self.boss_agent.record_spawning_decision(spawning_record)
        
        # Set agent to active
        subordinate.status = AgentStatus.ACTIVE
        
        logger.info(f"Spawned {subordinate.get_human_readable_name()} ({role.value}) with ID {agent_id}")
        
        # Update state
        await self._update_state_with_agents()
        
        return subordinate
    
    async def terminate_subordinate_agent(self, agent_id: str) -> bool:
        """
        Terminate a subordinate agent
        
        Args:
            agent_id: ID of agent to terminate
        
        Returns:
            True if successfully terminated
        """
        
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return False
        
        agent = self.agents[agent_id]
        
        if isinstance(agent, BossAgent):
            logger.error("Cannot terminate Boss agent")
            return False
        
        # Mark as terminated
        agent.status = AgentStatus.TERMINATED
        
        # Remove from active registries
        if agent.agent_number in self.subordinate_agents:
            del self.subordinate_agents[agent.agent_number]
        
        # Remove from Boss's subordinate list
        if self.boss_agent and agent_id in self.boss_agent.subordinate_agent_ids:
            self.boss_agent.subordinate_agent_ids.remove(agent_id)
        
        # Keep in main registry for historical purposes but mark as terminated
        logger.info(f"Terminated {agent.get_human_readable_name()}")
        
        # Update state
        await self._update_state_with_agents()
        
        return True
    
    def get_agent_by_id(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agent_by_number(self, agent_number: int) -> Optional[BaseAgent]:
        """Get agent by number (0 for Boss, 1+ for subordinates)"""
        if agent_number == 0:
            return self.boss_agent
        return self.subordinate_agents.get(agent_number)
    
    def get_active_subordinates(self) -> List[SubordinateAgent]:
        """Get all active subordinate agents"""
        return [
            agent for agent in self.subordinate_agents.values()
            if agent.status in [AgentStatus.ACTIVE, AgentStatus.IDLE, AgentStatus.BUSY]
        ]
    
    def get_available_agents(self) -> List[BaseAgent]:
        """Get agents available for new tasks"""
        return [agent for agent in self.agents.values() if agent.is_available()]
    
    def get_agents_with_capability(self, capability_name: str) -> List[BaseAgent]:
        """Get agents that have a specific capability"""
        return [
            agent for agent in self.agents.values() 
            if agent.has_capability(capability_name) and agent.status != AgentStatus.TERMINATED
        ]
    
    def get_hierarchy_summary(self) -> Dict[str, Any]:
        """Get summary of the agent hierarchy for UI display"""
        
        active_subordinates = self.get_active_subordinates()
        
        return {
            "boss_agent": {
                "id": self.boss_agent.agent_id if self.boss_agent else None,
                "number": 0,
                "display_name": "Boss Agent",
                "status": self.boss_agent.status.value if self.boss_agent else None,
                "subordinate_count": len(self.boss_agent.subordinate_agent_ids) if self.boss_agent else 0,
                "decisions_made": self.boss_agent.spawning_decisions_made + self.boss_agent.delegation_decisions_made if self.boss_agent else 0
            },
            "subordinate_agents": [
                {
                    "id": agent.agent_id,
                    "number": agent.agent_number,
                    "display_name": agent.get_human_readable_name(),
                    "role": agent.role.value,
                    "status": agent.status.value,
                    "specialization": getattr(agent, 'specialization_area', None),
                    "current_tasks": len(agent.current_tasks),
                    "max_tasks": agent.max_concurrent_tasks,
                    "load_percentage": agent.get_load_percentage(),
                    "success_rate": agent.metrics.success_rate,
                    "capabilities": agent.get_capability_names()
                }
                for agent in active_subordinates
            ],
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.status in [AgentStatus.ACTIVE, AgentStatus.IDLE, AgentStatus.BUSY]]),
            "next_agent_number": self.next_agent_number
        }
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents"""
        
        summary = {
            "boss_performance": None,
            "subordinate_performance": [],
            "overall_metrics": {
                "total_tasks_completed": 0,
                "total_tasks_failed": 0,
                "overall_success_rate": 0.0,
                "average_load_percentage": 0.0
            }
        }
        
        if self.boss_agent:
            summary["boss_performance"] = {
                "agent_number": 0,
                "display_name": "Boss Agent",
                "decisions_made": self.boss_agent.spawning_decisions_made + self.boss_agent.delegation_decisions_made,
                "subordinates_managed": len(self.boss_agent.subordinate_agent_ids),
                "metrics": self.boss_agent.metrics.dict()
            }
        
        total_completed = 0
        total_failed = 0
        total_load = 0.0
        active_count = 0
        
        for agent in self.get_active_subordinates():
            agent_perf = {
                "agent_number": agent.agent_number,
                "display_name": agent.get_human_readable_name(),
                "role": agent.role.value,
                "specialization": getattr(agent, 'specialization_area', None),
                "metrics": agent.metrics.dict(),
                "load_percentage": agent.get_load_percentage(),
                "capabilities_count": len(agent.get_capability_names())
            }
            summary["subordinate_performance"].append(agent_perf)
            
            total_completed += agent.metrics.tasks_completed
            total_failed += agent.metrics.tasks_failed
            total_load += agent.get_load_percentage()
            active_count += 1
        
        # Calculate overall metrics
        total_tasks = total_completed + total_failed
        summary["overall_metrics"]["total_tasks_completed"] = total_completed
        summary["overall_metrics"]["total_tasks_failed"] = total_failed
        summary["overall_metrics"]["overall_success_rate"] = (total_completed / total_tasks) if total_tasks > 0 else 1.0
        summary["overall_metrics"]["average_load_percentage"] = (total_load / active_count) if active_count > 0 else 0.0
        
        return summary
    
    async def assign_task_to_agent(self, agent_id: str, task_id: str) -> bool:
        """Assign a task to a specific agent"""
        
        agent = self.get_agent_by_id(agent_id)
        if not agent or not agent.is_available():
            return False
        
        agent.current_tasks.append(task_id)
        agent.status = AgentStatus.BUSY
        agent.last_updated = datetime.utcnow()
        
        logger.debug(f"Assigned task {task_id} to {agent.get_human_readable_name()}")
        
        # Update state
        await self._update_state_with_agents()
        
        return True
    
    async def complete_task_for_agent(self, agent_id: str, task_id: str, success: bool = True, feedback: Dict[str, Any] = None) -> bool:
        """Mark a task as completed for an agent"""
        
        agent = self.get_agent_by_id(agent_id)
        if not agent or task_id not in agent.current_tasks:
            return False
        
        # Remove task from current tasks
        agent.current_tasks.remove(task_id)
        
        # Update metrics
        if success:
            agent.metrics.tasks_completed += 1
        else:
            agent.metrics.tasks_failed += 1
        
        # Recalculate success rate
        total_tasks = agent.metrics.tasks_completed + agent.metrics.tasks_failed
        agent.metrics.success_rate = agent.metrics.tasks_completed / total_tasks if total_tasks > 0 else 1.0
        
        # Update status
        agent.status = AgentStatus.IDLE if len(agent.current_tasks) == 0 else AgentStatus.BUSY
        agent.metrics.last_active = datetime.utcnow()
        agent.last_updated = datetime.utcnow()
        
        # Apply learning for subordinate agents
        if isinstance(agent, SubordinateAgent) and feedback:
            agent.adapt_capabilities(feedback)
        
        logger.debug(f"Task {task_id} completed by {agent.get_human_readable_name()} (success: {success})")
        
        # Update state
        await self._update_state_with_agents()
        
        return True
    
    def _get_default_capabilities_for_role(self, role: AgentRole) -> List[AgentCapability]:
        """Get default capabilities for a given role"""
        
        capability_sets = {
            AgentRole.SPECIALIST: [
                AgentCapability(name="specialized_analysis", description="Specialized domain analysis", skill_level=0.8),
                AgentCapability(name="detailed_research", description="In-depth research capabilities", skill_level=0.7),
                AgentCapability(name="expert_consultation", description="Provide expert advice", skill_level=0.9)
            ],
            AgentRole.GENERALIST: [
                AgentCapability(name="general_problem_solving", description="General problem solving", skill_level=0.6),
                AgentCapability(name="task_execution", description="Execute various types of tasks", skill_level=0.7),
                AgentCapability(name="adaptability", description="Adapt to different contexts", skill_level=0.8)
            ],
            AgentRole.COORDINATOR: [
                AgentCapability(name="task_coordination", description="Coordinate multiple tasks", skill_level=0.8),
                AgentCapability(name="communication", description="Inter-agent communication", skill_level=0.9),
                AgentCapability(name="workflow_management", description="Manage complex workflows", skill_level=0.7)
            ],
            AgentRole.RESEARCHER: [
                AgentCapability(name="research", description="Research and information gathering", skill_level=0.9),
                AgentCapability(name="data_analysis", description="Analyze and synthesize data", skill_level=0.8),
                AgentCapability(name="report_generation", description="Generate research reports", skill_level=0.7)
            ],
            AgentRole.EXECUTOR: [
                AgentCapability(name="task_execution", description="Execute tasks efficiently", skill_level=0.9),
                AgentCapability(name="tool_usage", description="Use various tools and systems", skill_level=0.8),
                AgentCapability(name="quality_assurance", description="Ensure quality of outputs", skill_level=0.7)
            ]
        }
        
        return capability_sets.get(role, capability_sets[AgentRole.GENERALIST])
    
    async def _update_state_with_agents(self):
        """Update the global state with current agent information"""
        
        try:
            # Prepare agent data for state
            agents_data = {}
            active_agents = []
            
            for agent_id, agent in self.agents.items():
                agents_data[agent_id] = agent.dict()
                
                if agent.status in [AgentStatus.ACTIVE, AgentStatus.IDLE, AgentStatus.BUSY]:
                    active_agents.append(agent_id)
            
            # Update state
            await self.state_holder.update_state({
                "agents": agents_data,
                "active_agents": active_agents,
                "agent_hierarchy": [agent.agent_number for agent in self.agents.values() if agent.status != AgentStatus.TERMINATED]
            })
            
        except Exception as e:
            logger.warning(f"Failed to update state with agent data: {e}")
    
    async def get_optimal_agent_for_task(self, task_requirements: Dict[str, Any]) -> Optional[BaseAgent]:
        """
        Find the optimal agent for a task based on requirements
        
        Args:
            task_requirements: Requirements including capabilities needed, priority, etc.
        
        Returns:
            Best agent for the task or None if no suitable agent found
        """
        
        required_capabilities = task_requirements.get("capabilities", [])
        task_priority = task_requirements.get("priority", 3)
        
        # Get available agents
        available_agents = self.get_available_agents()
        
        if not available_agents:
            return None
        
        # Score agents based on suitability
        agent_scores = []
        
        for agent in available_agents:
            if isinstance(agent, BossAgent):
                continue  # Boss doesn't execute regular tasks
            
            score = 0.0
            
            # Capability matching
            capability_match_count = 0
            capability_skill_sum = 0.0
            
            for req_cap in required_capabilities:
                for agent_cap in agent.capabilities:
                    if agent_cap.name == req_cap and agent_cap.enabled:
                        capability_match_count += 1
                        capability_skill_sum += agent_cap.skill_level
                        break
            
            if capability_match_count > 0:
                score += (capability_match_count / len(required_capabilities)) * 50  # Up to 50 points for capability match
                score += (capability_skill_sum / capability_match_count) * 30  # Up to 30 points for skill level
            
            # Load balancing - prefer agents with lower current load
            load_penalty = agent.get_load_percentage() * 0.2  # Up to 20 point penalty for high load
            score -= load_penalty
            
            # Success rate bonus
            score += agent.metrics.success_rate * 20  # Up to 20 points for high success rate
            
            # Role appropriateness (simplified)
            if agent.role == AgentRole.SPECIALIST and len(required_capabilities) > 2:
                score += 10  # Specialists preferred for complex tasks
            elif agent.role == AgentRole.GENERALIST and len(required_capabilities) <= 2:
                score += 5   # Generalists good for simple tasks
            
            agent_scores.append((agent, score))
        
        # Sort by score and return best agent
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        if agent_scores and agent_scores[0][1] > 0:
            return agent_scores[0][0]
        
        return None
    
    async def rebalance_agents(self) -> Dict[str, Any]:
        """
        Rebalance agent workloads and optimize the hierarchy
        
        Returns:
            Summary of rebalancing actions taken
        """
        
        rebalancing_summary = {
            "actions_taken": [],
            "agents_affected": [],
            "load_improvements": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            active_agents = self.get_active_subordinates()
            
            if len(active_agents) < 2:
                return rebalancing_summary
            
            # Calculate load distribution
            loads = [(agent, agent.get_load_percentage()) for agent in active_agents]
            loads.sort(key=lambda x: x[1], reverse=True)  # High to low load
            
            avg_load = sum(load for _, load in loads) / len(loads)
            
            # Identify overloaded and underloaded agents
            overloaded = [agent for agent, load in loads if load > avg_load + 20]  # 20% above average
            underloaded = [agent for agent, load in loads if load < avg_load - 20]  # 20% below average
            
            if not overloaded or not underloaded:
                rebalancing_summary["actions_taken"].append("No significant load imbalance detected")
                return rebalancing_summary
            
            # TODO: Implement task reassignment logic here
            # For now, just log the analysis
            
            rebalancing_summary["actions_taken"].append(f"Identified {len(overloaded)} overloaded and {len(underloaded)} underloaded agents")
            rebalancing_summary["agents_affected"] = [
                {"agent_id": agent.agent_id, "load": agent.get_load_percentage(), "status": "overloaded"}
                for agent in overloaded
            ] + [
                {"agent_id": agent.agent_id, "load": agent.get_load_percentage(), "status": "underloaded"}
                for agent in underloaded
            ]
            
            logger.info(f"Agent rebalancing completed: {len(rebalancing_summary['actions_taken'])} actions")
            
        except Exception as e:
            logger.error(f"Error during agent rebalancing: {e}")
            rebalancing_summary["actions_taken"].append(f"Error: {str(e)}")
        
        return rebalancing_summary
