
"""
Agent spawning system using DSPY signatures for intelligent agent creation
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger

import dspy

from .agent_hierarchy import AgentHierarchy, AgentRole, AgentCapability, SubordinateAgent
from ..signatures.agent_management import AgentSpawningSignature, AgentSpawningContext, NewAgentSpec
from ..state.state_holder import StateHolder


class SpawningTrigger(Enum):
    """Triggers that can cause agent spawning"""
    HIGH_TASK_LOAD = "high_task_load"
    CAPABILITY_GAP = "capability_gap"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SPECIALIZATION_NEEDED = "specialization_needed"
    REDUNDANCY_REQUIRED = "redundancy_required"
    MANUAL_REQUEST = "manual_request"
    STRATEGIC_EXPANSION = "strategic_expansion"


class SpawningStrategy(Enum):
    """Strategies for agent spawning"""
    REACTIVE = "reactive"          # Spawn in response to immediate needs
    PROACTIVE = "proactive"        # Spawn based on predictions
    BALANCED = "balanced"          # Mix of reactive and proactive
    CONSERVATIVE = "conservative"  # Minimal spawning, focus on efficiency
    AGGRESSIVE = "aggressive"      # Liberal spawning for maximum coverage


class SpawningRule(BaseModel):
    """Rule for automatic agent spawning"""
    
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    trigger: SpawningTrigger = Field(..., description="What triggers this rule")
    conditions: Dict[str, Any] = Field(..., description="Conditions that must be met")
    agent_spec: Dict[str, Any] = Field(..., description="Specification for agent to spawn")
    priority: int = Field(default=3, description="Rule priority (1=high, 5=low)")
    enabled: bool = Field(default=True, description="Whether rule is active")
    cooldown_minutes: int = Field(default=30, description="Cooldown between applications")
    last_triggered: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class SpawningDecision(BaseModel):
    """Result of a spawning decision"""
    
    decision_id: str = Field(..., description="Unique decision identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    should_spawn: bool = Field(..., description="Whether to spawn agents")
    agents_to_spawn: List[NewAgentSpec] = Field(default_factory=list)
    reasoning: str = Field(..., description="Reasoning behind the decision")
    confidence: float = Field(..., description="Confidence in decision (0-1)")
    triggered_by: List[SpawningTrigger] = Field(default_factory=list)
    system_context: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class AgentSpawner:
    """
    Intelligent agent spawning system using DSPY signatures
    Makes autonomous decisions about when and what types of agents to spawn
    """
    
    def __init__(
        self,
        agent_hierarchy: AgentHierarchy,
        state_holder: StateHolder,
        strategy: SpawningStrategy = SpawningStrategy.BALANCED,
        llm_config: Dict[str, Any] = None
    ):
        self.agent_hierarchy = agent_hierarchy
        self.state_holder = state_holder
        self.strategy = strategy
        self.llm_config = llm_config or {"model": "gpt-4", "temperature": 0.7}
        
        # DSPY signature
        self.spawning_signature = AgentSpawningSignature()
        
        # Spawning rules and configuration
        self.spawning_rules: Dict[str, SpawningRule] = {}
        self.max_agents = 10  # Maximum number of subordinate agents
        self.min_agents = 1   # Minimum number of subordinate agents
        
        # Spawning history and metrics
        self.spawning_decisions: List[SpawningDecision] = []
        self.spawning_metrics = {
            "total_spawns": 0,
            "successful_spawns": 0,
            "failed_spawns": 0,
            "average_decision_time": 0.0
        }
        
        # Initialize default spawning rules
        self._initialize_default_rules()
        
        logger.info(f"AgentSpawner initialized with strategy: {strategy.value}")
    
    def _initialize_default_rules(self):
        """Initialize default spawning rules"""
        
        # High task load rule
        self.add_spawning_rule(SpawningRule(
            rule_id="high_task_load",
            name="High Task Load Response",
            trigger=SpawningTrigger.HIGH_TASK_LOAD,
            conditions={
                "min_task_queue_size": 5,
                "agent_utilization_threshold": 0.8,
                "max_wait_time_minutes": 10
            },
            agent_spec={
                "role": AgentRole.GENERALIST.value,
                "capabilities": ["task_execution", "general_problem_solving"],
                "max_concurrent_tasks": 3
            },
            priority=1,
            cooldown_minutes=15
        ))
        
        # Capability gap rule
        self.add_spawning_rule(SpawningRule(
            rule_id="capability_gap",
            name="Missing Capability Coverage",
            trigger=SpawningTrigger.CAPABILITY_GAP,
            conditions={
                "required_capabilities": [],  # Will be dynamically set
                "min_capability_coverage": 0.7
            },
            agent_spec={
                "role": AgentRole.SPECIALIST.value,
                "capabilities": [],  # Will be dynamically set
                "max_concurrent_tasks": 2
            },
            priority=2,
            cooldown_minutes=45
        ))
        
        # Performance optimization rule
        self.add_spawning_rule(SpawningRule(
            rule_id="performance_optimization",
            name="Performance Optimization",
            trigger=SpawningTrigger.PERFORMANCE_OPTIMIZATION,
            conditions={
                "success_rate_threshold": 0.75,
                "average_task_time_threshold": 300  # 5 minutes
            },
            agent_spec={
                "role": AgentRole.COORDINATOR.value,
                "capabilities": ["task_coordination", "workflow_management"],
                "max_concurrent_tasks": 4
            },
            priority=3,
            cooldown_minutes=60
        ))
    
    def add_spawning_rule(self, rule: SpawningRule):
        """Add a spawning rule"""
        self.spawning_rules[rule.rule_id] = rule
        logger.info(f"Added spawning rule: {rule.name}")
    
    def remove_spawning_rule(self, rule_id: str) -> bool:
        """Remove a spawning rule"""
        if rule_id in self.spawning_rules:
            del self.spawning_rules[rule_id]
            logger.info(f"Removed spawning rule: {rule_id}")
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a spawning rule"""
        if rule_id in self.spawning_rules:
            self.spawning_rules[rule_id].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a spawning rule"""
        if rule_id in self.spawning_rules:
            self.spawning_rules[rule_id].enabled = False
            return True
        return False
    
    async def evaluate_spawning_needs(self) -> SpawningDecision:
        """
        Evaluate current system state and decide if agents should be spawned
        
        Returns:
            SpawningDecision with recommendation
        """
        decision_start_time = datetime.utcnow()
        
        try:
            logger.debug("Evaluating spawning needs")
            
            # Gather system context
            system_context = await self._gather_spawning_context()
            
            # Check rule-based triggers
            triggered_rules = await self._check_rule_triggers(system_context)
            
            # Use DSPY signature for intelligent decision making
            spawning_decision = await self._make_dspy_spawning_decision(system_context, triggered_rules)
            
            # Record decision
            self.spawning_decisions.append(spawning_decision)
            
            # Keep only last 100 decisions
            if len(self.spawning_decisions) > 100:
                self.spawning_decisions.pop(0)
            
            # Update metrics
            decision_time = (datetime.utcnow() - decision_start_time).total_seconds()
            self._update_decision_metrics(decision_time)
            
            logger.info(f"Spawning evaluation completed: spawn={spawning_decision.should_spawn}, agents={len(spawning_decision.agents_to_spawn)}")
            
            return spawning_decision
            
        except Exception as e:
            logger.error(f"Error evaluating spawning needs: {e}")
            
            # Return conservative decision on error
            return SpawningDecision(
                decision_id=f"error_{int(decision_start_time.timestamp())}",
                should_spawn=False,
                reasoning=f"Error in evaluation: {str(e)}",
                confidence=0.0
            )
    
    async def execute_spawning_decision(self, decision: SpawningDecision) -> Dict[str, Any]:
        """
        Execute a spawning decision by actually creating the agents
        
        Args:
            decision: The spawning decision to execute
        
        Returns:
            Results of spawning execution
        """
        
        if not decision.should_spawn or not decision.agents_to_spawn:
            return {
                "success": True,
                "message": "No spawning required",
                "agents_created": []
            }
        
        execution_results = {
            "success": True,
            "message": "",
            "agents_created": [],
            "errors": []
        }
        
        try:
            for agent_spec in decision.agents_to_spawn:
                try:
                    # Check if we've reached max agents
                    active_subordinates = len(self.agent_hierarchy.get_active_subordinates())
                    if active_subordinates >= self.max_agents:
                        execution_results["errors"].append(f"Maximum agent limit ({self.max_agents}) reached")
                        break
                    
                    # Convert agent spec to spawning parameters
                    role = AgentRole(agent_spec.agent_type) if agent_spec.agent_type in [r.value for r in AgentRole] else AgentRole.GENERALIST
                    
                    # Create capabilities
                    capabilities = [
                        AgentCapability(
                            name=cap_name,
                            description=f"Auto-generated capability: {cap_name}",
                            skill_level=0.7  # Default skill level
                        )
                        for cap_name in agent_spec.capabilities
                    ]
                    
                    # Spawn the agent
                    new_agent = await self.agent_hierarchy.spawn_subordinate_agent(
                        role=role,
                        capabilities=capabilities,
                        specialization=agent_spec.specialization,
                        config={
                            "max_concurrent_tasks": agent_spec.max_concurrent_tasks,
                            "llm_config": agent_spec.model_config
                        }
                    )
                    
                    execution_results["agents_created"].append({
                        "agent_id": new_agent.agent_id,
                        "agent_number": new_agent.agent_number,
                        "display_name": new_agent.get_human_readable_name(),
                        "role": new_agent.role.value,
                        "specialization": agent_spec.specialization
                    })
                    
                    self.spawning_metrics["successful_spawns"] += 1
                    self.spawning_metrics["total_spawns"] += 1
                    
                    logger.info(f"Successfully spawned {new_agent.get_human_readable_name()}")
                    
                except Exception as agent_error:
                    error_msg = f"Failed to spawn agent {agent_spec.agent_name}: {str(agent_error)}"
                    execution_results["errors"].append(error_msg)
                    self.spawning_metrics["failed_spawns"] += 1
                    self.spawning_metrics["total_spawns"] += 1
                    logger.error(error_msg)
            
            if execution_results["errors"]:
                execution_results["success"] = len(execution_results["agents_created"]) > 0
                execution_results["message"] = f"Partial success: {len(execution_results['agents_created'])} created, {len(execution_results['errors'])} errors"
            else:
                execution_results["message"] = f"Successfully created {len(execution_results['agents_created'])} agents"
            
        except Exception as e:
            execution_results["success"] = False
            execution_results["message"] = f"Execution failed: {str(e)}"
            execution_results["errors"].append(str(e))
            logger.error(f"Spawning execution failed: {e}")
        
        return execution_results
    
    async def auto_spawn_if_needed(self) -> Dict[str, Any]:
        """
        Automatically evaluate and execute spawning if needed
        
        Returns:
            Combined results of evaluation and execution
        """
        
        try:
            # Evaluate spawning needs
            decision = await self.evaluate_spawning_needs()
            
            # Execute if needed
            execution_results = await self.execute_spawning_decision(decision)
            
            return {
                "decision": decision.dict(),
                "execution": execution_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Auto-spawning failed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _gather_spawning_context(self) -> AgentSpawningContext:
        """Gather context for spawning decisions"""
        
        try:
            current_state = await self.state_holder.get_current_state()
            
            if not current_state:
                return AgentSpawningContext(
                    current_workload={},
                    existing_agents=[],
                    task_queue_analysis={},
                    system_capacity={},
                    performance_requirements={}
                )
            
            # Current workload analysis
            active_agents = self.agent_hierarchy.get_active_subordinates()
            workload = {
                "active_agent_count": len(active_agents),
                "total_current_tasks": sum(len(agent.current_tasks) for agent in active_agents),
                "average_load_percentage": sum(agent.get_load_percentage() for agent in active_agents) / len(active_agents) if active_agents else 0,
                "overloaded_agents": len([agent for agent in active_agents if agent.get_load_percentage() > 80]),
                "idle_agents": len([agent for agent in active_agents if agent.status.value == "idle"])
            }
            
            # Existing agents summary
            existing_agents_data = [
                {
                    "agent_id": agent.agent_id,
                    "agent_number": agent.agent_number,
                    "role": agent.role.value,
                    "capabilities": agent.get_capability_names(),
                    "current_load": agent.get_load_percentage(),
                    "success_rate": agent.metrics.success_rate,
                    "status": agent.status.value
                }
                for agent in active_agents
            ]
            
            # Task queue analysis
            pending_tasks = current_state.tasks.get("pending", []) if isinstance(current_state.tasks, dict) else []
            task_queue_analysis = {
                "pending_task_count": len(pending_tasks),
                "estimated_total_workload": len(pending_tasks) * 1.5,  # Simplified estimate
                "high_priority_tasks": 0,  # Would need task priority analysis
                "average_wait_time": 0.0   # Would need historical data
            }
            
            # System capacity
            system_capacity = {
                "max_agents_allowed": self.max_agents,
                "current_agent_count": len(active_agents),
                "available_agent_slots": self.max_agents - len(active_agents),
                "resource_utilization": workload["average_load_percentage"] / 100
            }
            
            # Performance requirements (simplified)
            performance_requirements = {
                "target_success_rate": 0.85,
                "target_response_time": 60.0,  # seconds
                "max_queue_wait_time": 300.0,  # 5 minutes
                "desired_redundancy_factor": 1.2
            }
            
            return AgentSpawningContext(
                current_workload=workload,
                existing_agents=existing_agents_data,
                task_queue_analysis=task_queue_analysis,
                system_capacity=system_capacity,
                performance_requirements=performance_requirements
            )
            
        except Exception as e:
            logger.error(f"Error gathering spawning context: {e}")
            return AgentSpawningContext(
                current_workload={"error": str(e)},
                existing_agents=[],
                task_queue_analysis={},
                system_capacity={},
                performance_requirements={}
            )
    
    async def _check_rule_triggers(self, context: AgentSpawningContext) -> List[SpawningRule]:
        """Check which spawning rules are triggered"""
        
        triggered_rules = []
        
        for rule in self.spawning_rules.values():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                time_since_trigger = datetime.utcnow() - rule.last_triggered
                if time_since_trigger.total_seconds() < (rule.cooldown_minutes * 60):
                    continue
            
            # Check rule conditions
            try:
                if await self._evaluate_rule_conditions(rule, context):
                    triggered_rules.append(rule)
                    rule.last_triggered = datetime.utcnow()
                    logger.debug(f"Rule triggered: {rule.name}")
                    
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule.rule_id}: {e}")
        
        return triggered_rules
    
    async def _evaluate_rule_conditions(self, rule: SpawningRule, context: AgentSpawningContext) -> bool:
        """Evaluate if a rule's conditions are met"""
        
        conditions = rule.conditions
        workload = context.current_workload
        
        # High task load rule
        if rule.trigger == SpawningTrigger.HIGH_TASK_LOAD:
            task_queue_size = context.task_queue_analysis.get("pending_task_count", 0)
            avg_utilization = workload.get("average_load_percentage", 0) / 100
            
            return (
                task_queue_size >= conditions.get("min_task_queue_size", 5) or
                avg_utilization >= conditions.get("agent_utilization_threshold", 0.8)
            )
        
        # Capability gap rule
        elif rule.trigger == SpawningTrigger.CAPABILITY_GAP:
            # Simplified capability gap detection
            existing_capabilities = set()
            for agent_data in context.existing_agents:
                existing_capabilities.update(agent_data.get("capabilities", []))
            
            required_capabilities = set(conditions.get("required_capabilities", []))
            if not required_capabilities:
                # Use common capabilities as baseline
                required_capabilities = {"task_execution", "problem_solving", "communication", "analysis"}
            
            coverage = len(existing_capabilities.intersection(required_capabilities)) / len(required_capabilities)
            return coverage < conditions.get("min_capability_coverage", 0.7)
        
        # Performance optimization rule
        elif rule.trigger == SpawningTrigger.PERFORMANCE_OPTIMIZATION:
            if not context.existing_agents:
                return False
            
            avg_success_rate = sum(agent.get("success_rate", 1.0) for agent in context.existing_agents) / len(context.existing_agents)
            return avg_success_rate < conditions.get("success_rate_threshold", 0.75)
        
        # Default: rule not triggered
        return False
    
    async def _make_dspy_spawning_decision(self, context: AgentSpawningContext, triggered_rules: List[SpawningRule]) -> SpawningDecision:
        """Use DSPY signature to make intelligent spawning decision"""
        
        try:
            # Prepare system requirements
            system_requirements = {
                "max_agents": self.max_agents,
                "current_agents": len(context.existing_agents),
                "available_slots": context.system_capacity.get("available_agent_slots", 0),
                "workload_pressure": context.current_workload.get("average_load_percentage", 0),
                "performance_targets": context.performance_requirements
            }
            
            # Boss strategy description
            boss_strategies = {
                SpawningStrategy.REACTIVE: "React to immediate needs and bottlenecks",
                SpawningStrategy.PROACTIVE: "Anticipate future needs and spawn preemptively", 
                SpawningStrategy.BALANCED: "Balance immediate needs with future planning",
                SpawningStrategy.CONSERVATIVE: "Minimize agents while meeting core requirements",
                SpawningStrategy.AGGRESSIVE: "Maximize coverage and redundancy"
            }
            
            boss_strategy = boss_strategies.get(self.strategy, boss_strategies[SpawningStrategy.BALANCED])
            
            # Use DSPY signature
            with dspy.context(lm=dspy.OpenAI(**self.llm_config)):
                spawning_result = self.spawning_signature(
                    spawning_context=context,
                    system_requirements=system_requirements,
                    boss_strategy=boss_strategy
                )
            
            # Create decision
            decision = SpawningDecision(
                decision_id=f"dspy_{int(datetime.utcnow().timestamp())}",
                should_spawn=spawning_result.spawning_decision.should_spawn,
                agents_to_spawn=spawning_result.spawning_decision.agent_specifications,
                reasoning=spawning_result.spawning_decision.spawning_reasoning,
                confidence=0.8,  # DSPY decisions get high confidence
                triggered_by=[rule.trigger for rule in triggered_rules],
                system_context={
                    "workload": context.current_workload,
                    "capacity": context.system_capacity,
                    "triggered_rules": [rule.name for rule in triggered_rules]
                }
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"DSPY spawning decision failed: {e}")
            
            # Fallback to rule-based decision
            return await self._make_fallback_spawning_decision(context, triggered_rules)
    
    async def _make_fallback_spawning_decision(self, context: AgentSpawningContext, triggered_rules: List[SpawningRule]) -> SpawningDecision:
        """Fallback spawning decision when DSPY fails"""
        
        decision = SpawningDecision(
            decision_id=f"fallback_{int(datetime.utcnow().timestamp())}",
            should_spawn=False,
            reasoning="Fallback decision due to DSPY failure",
            confidence=0.3,
            triggered_by=[rule.trigger for rule in triggered_rules]
        )
        
        # Simple rule-based logic
        if not triggered_rules:
            return decision
        
        available_slots = context.system_capacity.get("available_agent_slots", 0)
        if available_slots <= 0:
            decision.reasoning = "No available slots for new agents"
            return decision
        
        # Spawn based on highest priority triggered rule
        highest_priority_rule = min(triggered_rules, key=lambda r: r.priority)
        
        agent_spec = NewAgentSpec(
            agent_name=f"Agent {self.agent_hierarchy.next_agent_number}",
            agent_type=highest_priority_rule.agent_spec.get("role", AgentRole.GENERALIST.value),
            capabilities=highest_priority_rule.agent_spec.get("capabilities", ["task_execution"]),
            specialization="Auto-spawned agent",
            max_concurrent_tasks=highest_priority_rule.agent_spec.get("max_concurrent_tasks", 3),
            model_config={}
        )
        
        decision.should_spawn = True
        decision.agents_to_spawn = [agent_spec]
        decision.reasoning = f"Fallback spawning triggered by rule: {highest_priority_rule.name}"
        decision.confidence = 0.6
        
        return decision
    
    def _update_decision_metrics(self, decision_time: float):
        """Update spawning decision metrics"""
        
        # Update average decision time
        if self.spawning_metrics["average_decision_time"] == 0:
            self.spawning_metrics["average_decision_time"] = decision_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.spawning_metrics["average_decision_time"] = (
                alpha * decision_time + 
                (1 - alpha) * self.spawning_metrics["average_decision_time"]
            )
    
    # Public interface methods
    
    def get_spawning_metrics(self) -> Dict[str, Any]:
        """Get spawning metrics and statistics"""
        
        return {
            **self.spawning_metrics,
            "strategy": self.strategy.value,
            "active_rules": len([r for r in self.spawning_rules.values() if r.enabled]),
            "total_rules": len(self.spawning_rules),
            "recent_decisions": len(self.spawning_decisions),
            "max_agents": self.max_agents,
            "min_agents": self.min_agents,
            "current_agents": len(self.agent_hierarchy.get_active_subordinates())
        }
    
    def get_spawning_rules_summary(self) -> List[Dict[str, Any]]:
        """Get summary of spawning rules"""
        
        return [
            {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "trigger": rule.trigger.value,
                "priority": rule.priority,
                "enabled": rule.enabled,
                "cooldown_minutes": rule.cooldown_minutes,
                "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
            }
            for rule in self.spawning_rules.values()
        ]
    
    def get_recent_decisions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent spawning decisions"""
        
        recent_decisions = self.spawning_decisions[-count:] if self.spawning_decisions else []
        return [decision.dict() for decision in recent_decisions]
    
    def update_strategy(self, new_strategy: SpawningStrategy):
        """Update spawning strategy"""
        
        old_strategy = self.strategy
        self.strategy = new_strategy
        
        logger.info(f"Spawning strategy updated from {old_strategy.value} to {new_strategy.value}")
    
    def set_agent_limits(self, min_agents: int, max_agents: int):
        """Set agent count limits"""
        
        if min_agents < 0 or max_agents < min_agents:
            raise ValueError("Invalid agent limits")
        
        self.min_agents = min_agents
        self.max_agents = max_agents
        
        logger.info(f"Agent limits updated: min={min_agents}, max={max_agents}")
    
    async def force_spawn_agent(
        self, 
        role: AgentRole = AgentRole.GENERALIST,
        capabilities: List[str] = None,
        specialization: str = None
    ) -> Dict[str, Any]:
        """Force spawn an agent (bypassing normal decision logic)"""
        
        try:
            # Check limits
            active_count = len(self.agent_hierarchy.get_active_subordinates())
            if active_count >= self.max_agents:
                return {
                    "success": False,
                    "message": f"Cannot spawn: at maximum limit of {self.max_agents} agents"
                }
            
            # Create agent spec
            agent_spec = NewAgentSpec(
                agent_name=f"Agent {self.agent_hierarchy.next_agent_number}",
                agent_type=role.value,
                capabilities=capabilities or ["task_execution", "general_problem_solving"],
                specialization=specialization or "Manually spawned agent",
                max_concurrent_tasks=3,
                model_config={}
            )
            
            # Create decision
            decision = SpawningDecision(
                decision_id=f"manual_{int(datetime.utcnow().timestamp())}",
                should_spawn=True,
                agents_to_spawn=[agent_spec],
                reasoning="Manual agent spawning requested",
                confidence=1.0,
                triggered_by=[SpawningTrigger.MANUAL_REQUEST]
            )
            
            # Execute spawning
            result = await self.execute_spawning_decision(decision)
            
            return result
            
        except Exception as e:
            logger.error(f"Force spawn failed: {e}")
            return {
                "success": False,
                "message": f"Force spawn failed: {str(e)}"
            }
