
"""
Central agent manager that coordinates the entire agent hierarchy system
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger

from .agent_hierarchy import AgentHierarchy, BaseAgent, BossAgent, SubordinateAgent, AgentRole, AgentStatus
from .agent_spawner import AgentSpawner, SpawningStrategy, SpawningDecision
from .agent_communication import AgentCommunicationHub, MessageType, MessagePriority
from ..signatures.agent_management import TaskDelegationSignature, TaskDelegationContext, TaskAssignment
from ..state.state_holder import StateHolder


class AgentRegistry(BaseModel):
    """Registry of all agents in the system"""
    
    agents: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    boss_agent_id: Optional[str] = None
    active_agent_count: int = 0
    total_spawned: int = 0
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent"""
        self.agents[agent.agent_id] = {
            "agent_number": agent.agent_number,
            "display_name": agent.get_human_readable_name(),
            "role": agent.role.value,
            "status": agent.status.value,
            "registered_at": datetime.utcnow().isoformat(),
            "capabilities": agent.get_capability_names()
        }
        
        if isinstance(agent, BossAgent):
            self.boss_agent_id = agent.agent_id
        
        if agent.status in [AgentStatus.ACTIVE, AgentStatus.IDLE, AgentStatus.BUSY]:
            self.active_agent_count += 1
        
        self.total_spawned += 1
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            agent_info = self.agents[agent_id]
            if agent_info.get("status") in ["active", "idle", "busy"]:
                self.active_agent_count = max(0, self.active_agent_count - 1)
            
            # Mark as terminated rather than removing
            self.agents[agent_id]["status"] = "terminated"
            self.agents[agent_id]["terminated_at"] = datetime.utcnow().isoformat()
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of agent registry"""
        
        status_counts = {}
        role_counts = {}
        
        for agent_info in self.agents.values():
            # Count by status
            status = agent_info.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by role
            role = agent_info.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
        
        return {
            "total_agents": len(self.agents),
            "active_agents": self.active_agent_count,
            "total_spawned": self.total_spawned,
            "boss_agent_id": self.boss_agent_id,
            "status_distribution": status_counts,
            "role_distribution": role_counts
        }


class TaskDistributionStrategy(Enum):
    """Strategies for distributing tasks among agents"""
    LOAD_BALANCED = "load_balanced"          # Distribute based on current load
    CAPABILITY_MATCHED = "capability_matched" # Match tasks to agent capabilities
    PERFORMANCE_BASED = "performance_based"   # Assign to highest performing agents
    ROUND_ROBIN = "round_robin"              # Simple round-robin distribution
    INTELLIGENT = "intelligent"              # Use DSPY for smart distribution


class AgentManager:
    """
    Central manager for the entire agent hierarchy system
    Coordinates agent spawning, communication, task distribution, and lifecycle management
    """
    
    def __init__(
        self,
        state_holder: StateHolder,
        spawning_strategy: SpawningStrategy = SpawningStrategy.BALANCED,
        task_distribution_strategy: TaskDistributionStrategy = TaskDistributionStrategy.INTELLIGENT,
        llm_config: Dict[str, Any] = None
    ):
        self.state_holder = state_holder
        self.spawning_strategy = spawning_strategy
        self.task_distribution_strategy = task_distribution_strategy
        self.llm_config = llm_config or {"model": "gpt-4", "temperature": 0.7}
        
        # Core components
        self.agent_hierarchy = AgentHierarchy(state_holder)
        self.agent_spawner = AgentSpawner(self.agent_hierarchy, state_holder, spawning_strategy, llm_config)
        self.communication_hub = AgentCommunicationHub(self.agent_hierarchy)
        self.agent_registry = AgentRegistry()
        
        # Task delegation
        self.task_delegation_signature = TaskDelegationSignature()
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.management_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.system_metrics = {
            "uptime_start": None,
            "total_tasks_delegated": 0,
            "successful_delegations": 0,
            "failed_delegations": 0,
            "agent_spawns": 0,
            "agent_terminations": 0,
            "communication_messages": 0
        }
        
        # Event callbacks
        self.agent_spawned_callbacks: List[Callable] = []
        self.agent_terminated_callbacks: List[Callable] = []
        self.task_delegated_callbacks: List[Callable] = []
        
        # Setup communication handlers
        self._setup_communication_handlers()
        
        logger.info("AgentManager initialized")
    
    async def initialize_system(self, boss_config: Dict[str, Any] = None) -> bool:
        """
        Initialize the complete agent management system
        
        Args:
            boss_config: Configuration for Boss agent
        
        Returns:
            True if initialization successful
        """
        
        if self.is_initialized:
            logger.warning("Agent management system already initialized")
            return True
        
        try:
            logger.info("Initializing agent management system")
            
            # Initialize Boss agent
            boss_agent = await self.agent_hierarchy.initialize_boss_agent(boss_config)
            self.agent_registry.register_agent(boss_agent)
            
            # Create communication queue for Boss
            self.communication_hub.create_agent_queue(boss_agent.agent_id)
            
            # Start communication processing
            await self.communication_hub.start_processing()
            
            # Send initialization announcement
            await self.communication_hub.send_broadcast(
                sender_id=boss_agent.agent_id,
                message_type=MessageType.SYSTEM_ANNOUNCEMENT,
                subject="System Initialization Complete",
                content={
                    "message": "Agent management system initialized",
                    "boss_agent": boss_agent.get_human_readable_name(),
                    "timestamp": datetime.utcnow().isoformat()
                },
                priority=MessagePriority.HIGH
            )
            
            self.is_initialized = True
            self.system_metrics["uptime_start"] = datetime.utcnow()
            
            logger.info("Agent management system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent management system: {e}")
            return False
    
    async def start_autonomous_management(self) -> bool:
        """
        Start autonomous agent management
        
        Returns:
            True if started successfully
        """
        
        if not self.is_initialized:
            logger.error("System must be initialized before starting autonomous management")
            return False
        
        if self.is_running:
            logger.warning("Autonomous management already running")
            return True
        
        try:
            logger.info("Starting autonomous agent management")
            
            # Start background management task
            self.management_task = asyncio.create_task(self._autonomous_management_loop())
            self.is_running = True
            
            # Announce start of autonomous management
            if self.agent_hierarchy.boss_agent:
                await self.communication_hub.send_broadcast(
                    sender_id=self.agent_hierarchy.boss_agent.agent_id,
                    message_type=MessageType.SYSTEM_ANNOUNCEMENT,
                    subject="Autonomous Management Active",
                    content={
                        "message": "Autonomous agent management is now active",
                        "features": [
                            "Automatic agent spawning",
                            "Intelligent task delegation", 
                            "Performance monitoring",
                            "Load balancing"
                        ]
                    },
                    priority=MessagePriority.HIGH
                )
            
            logger.info("Autonomous agent management started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start autonomous management: {e}")
            return False
    
    async def stop_autonomous_management(self) -> bool:
        """
        Stop autonomous agent management
        
        Returns:
            True if stopped successfully
        """
        
        if not self.is_running:
            return True
        
        try:
            logger.info("Stopping autonomous agent management")
            
            self.is_running = False
            
            # Stop management task
            if self.management_task:
                self.management_task.cancel()
                try:
                    await self.management_task
                except asyncio.CancelledError:
                    pass
            
            # Stop communication processing
            await self.communication_hub.stop_processing()
            
            logger.info("Autonomous agent management stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping autonomous management: {e}")
            return False
    
    async def delegate_task_intelligently(
        self,
        task_id: str,
        task_requirements: Dict[str, Any],
        task_priority: int = 3
    ) -> Dict[str, Any]:
        """
        Intelligently delegate a task using DSPY signatures
        
        Args:
            task_id: Unique task identifier
            task_requirements: Task requirements including capabilities needed
            task_priority: Task priority (1=high, 5=low)
        
        Returns:
            Delegation result
        """
        
        try:
            logger.info(f"Delegating task {task_id} with priority {task_priority}")
            
            if self.task_distribution_strategy == TaskDistributionStrategy.INTELLIGENT:
                return await self._delegate_using_dspy(task_id, task_requirements, task_priority)
            else:
                return await self._delegate_using_strategy(task_id, task_requirements, task_priority)
                
        except Exception as e:
            logger.error(f"Task delegation failed for {task_id}: {e}")
            
            self.system_metrics["failed_delegations"] += 1
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
    
    async def _delegate_using_dspy(
        self,
        task_id: str,
        task_requirements: Dict[str, Any],
        task_priority: int
    ) -> Dict[str, Any]:
        """Delegate task using DSPY signature for intelligent assignment"""
        
        try:
            # Prepare delegation context
            available_tasks = [
                {
                    "task_id": task_id,
                    "requirements": task_requirements,
                    "priority": task_priority,
                    "estimated_duration": task_requirements.get("estimated_duration", 60)
                }
            ]
            
            # Get agent capabilities and loads
            active_agents = self.agent_hierarchy.get_active_subordinates()
            agent_capabilities = {}
            agent_current_load = {}
            
            for agent in active_agents:
                agent_capabilities[agent.agent_id] = agent.get_capability_names()
                agent_current_load[agent.agent_id] = len(agent.current_tasks)
            
            # Task priorities and deadlines
            task_priorities = {task_id: task_priority}
            deadline_constraints = {task_id: task_requirements.get("deadline", "flexible")}
            
            # Create delegation context
            delegation_context = TaskDelegationContext(
                available_tasks=available_tasks,
                agent_capabilities=agent_capabilities,
                agent_current_load=agent_current_load,
                task_priorities=task_priorities,
                deadline_constraints=deadline_constraints
            )
            
            # Optimization goals
            optimization_goals = [
                "maximize_success_probability",
                "balance_agent_workload",
                "meet_deadline_constraints",
                "utilize_agent_capabilities"
            ]
            
            # Use DSPY signature
            with dspy.context(lm=dspy.OpenAI(**self.llm_config)):
                delegation_result = self.task_delegation_signature(
                    delegation_context=delegation_context,
                    optimization_goals=optimization_goals
                )
            
            # Execute the delegation
            delegation_output = delegation_result.delegation_output
            
            if not delegation_output.task_assignments:
                return {
                    "success": False,
                    "message": "No suitable agent found for task",
                    "task_id": task_id
                }
            
            # Find assignment for our task
            task_assignment = None
            for assignment in delegation_output.task_assignments:
                if assignment.task_id == task_id:
                    task_assignment = assignment
                    break
            
            if not task_assignment:
                return {
                    "success": False,
                    "message": "Task not assigned by DSPY delegation",
                    "task_id": task_id
                }
            
            # Assign task to agent
            success = await self.agent_hierarchy.assign_task_to_agent(
                task_assignment.assigned_agent_id,
                task_id
            )
            
            if success:
                # Send task assignment message
                assigned_agent = self.agent_hierarchy.get_agent_by_id(task_assignment.assigned_agent_id)
                
                await self.communication_hub.send_message(
                    sender_id=self.agent_hierarchy.boss_agent.agent_id,
                    recipient_id=task_assignment.assigned_agent_id,
                    message_type=MessageType.TASK_ASSIGNMENT,
                    subject=f"Task Assignment: {task_id}",
                    content={
                        "task_id": task_id,
                        "task_requirements": task_requirements,
                        "priority": task_priority,
                        "assignment_reasoning": task_assignment.assignment_reasoning,
                        "expected_completion_time": task_assignment.expected_completion_time,
                        "success_criteria": task_assignment.success_criteria
                    },
                    priority=MessagePriority.HIGH if task_priority <= 2 else MessagePriority.NORMAL,
                    requires_ack=True,
                    response_expected=True,
                    response_timeout_seconds=300
                )
                
                # Update metrics
                self.system_metrics["total_tasks_delegated"] += 1
                self.system_metrics["successful_delegations"] += 1
                
                # Notify callbacks
                for callback in self.task_delegated_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(task_id, task_assignment.assigned_agent_id, task_assignment)
                        else:
                            callback(task_id, task_assignment.assigned_agent_id, task_assignment)
                    except Exception as e:
                        logger.warning(f"Task delegation callback failed: {e}")
                
                result = {
                    "success": True,
                    "task_id": task_id,
                    "assigned_agent_id": task_assignment.assigned_agent_id,
                    "assigned_agent_name": assigned_agent.get_human_readable_name() if assigned_agent else "Unknown",
                    "assignment_reasoning": task_assignment.assignment_reasoning,
                    "expected_completion_time": task_assignment.expected_completion_time,
                    "delegation_strategy": delegation_output.delegation_strategy,
                    "confidence": "high"  # DSPY delegations get high confidence
                }
                
                logger.info(f"Task {task_id} successfully delegated to {assigned_agent.get_human_readable_name()}")
                return result
            else:
                self.system_metrics["failed_delegations"] += 1
                return {
                    "success": False,
                    "message": "Failed to assign task to selected agent",
                    "task_id": task_id,
                    "selected_agent_id": task_assignment.assigned_agent_id
                }
                
        except Exception as e:
            logger.error(f"DSPY delegation failed: {e}")
            self.system_metrics["failed_delegations"] += 1
            return {
                "success": False,
                "error": f"DSPY delegation error: {str(e)}",
                "task_id": task_id
            }
    
    async def _delegate_using_strategy(
        self,
        task_id: str,
        task_requirements: Dict[str, Any],
        task_priority: int
    ) -> Dict[str, Any]:
        """Delegate task using non-DSPY strategies"""
        
        try:
            active_agents = self.agent_hierarchy.get_active_subordinates()
            
            if not active_agents:
                return {
                    "success": False,
                    "message": "No active agents available",
                    "task_id": task_id
                }
            
            selected_agent = None
            
            # Apply strategy
            if self.task_distribution_strategy == TaskDistributionStrategy.LOAD_BALANCED:
                # Select agent with lowest load
                selected_agent = min(active_agents, key=lambda a: a.get_load_percentage())
                
            elif self.task_distribution_strategy == TaskDistributionStrategy.CAPABILITY_MATCHED:
                # Select agent with best capability match
                required_caps = task_requirements.get("capabilities", [])
                best_match_score = -1
                
                for agent in active_agents:
                    if not agent.is_available():
                        continue
                    
                    match_score = len(set(agent.get_capability_names()).intersection(set(required_caps)))
                    if match_score > best_match_score:
                        best_match_score = match_score
                        selected_agent = agent
                        
            elif self.task_distribution_strategy == TaskDistributionStrategy.PERFORMANCE_BASED:
                # Select agent with highest success rate
                available_agents = [a for a in active_agents if a.is_available()]
                if available_agents:
                    selected_agent = max(available_agents, key=lambda a: a.metrics.success_rate)
                    
            elif self.task_distribution_strategy == TaskDistributionStrategy.ROUND_ROBIN:
                # Simple round-robin (simplified implementation)
                available_agents = [a for a in active_agents if a.is_available()]
                if available_agents:
                    # Use task count as simple round-robin indicator
                    selected_agent = min(available_agents, key=lambda a: a.metrics.tasks_completed)
            
            if not selected_agent or not selected_agent.is_available():
                return {
                    "success": False,
                    "message": "No suitable agent available for task",
                    "task_id": task_id
                }
            
            # Assign task
            success = await self.agent_hierarchy.assign_task_to_agent(selected_agent.agent_id, task_id)
            
            if success:
                # Send notification
                await self.communication_hub.send_message(
                    sender_id=self.agent_hierarchy.boss_agent.agent_id,
                    recipient_id=selected_agent.agent_id,
                    message_type=MessageType.TASK_ASSIGNMENT,
                    subject=f"Task Assignment: {task_id}",
                    content={
                        "task_id": task_id,
                        "task_requirements": task_requirements,
                        "priority": task_priority,
                        "assignment_strategy": self.task_distribution_strategy.value
                    },
                    priority=MessagePriority.HIGH if task_priority <= 2 else MessagePriority.NORMAL
                )
                
                self.system_metrics["successful_delegations"] += 1
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "assigned_agent_id": selected_agent.agent_id,
                    "assigned_agent_name": selected_agent.get_human_readable_name(),
                    "assignment_strategy": self.task_distribution_strategy.value,
                    "confidence": "medium"
                }
            else:
                self.system_metrics["failed_delegations"] += 1
                return {
                    "success": False,
                    "message": "Failed to assign task to agent",
                    "task_id": task_id
                }
                
        except Exception as e:
            logger.error(f"Strategy-based delegation failed: {e}")
            self.system_metrics["failed_delegations"] += 1
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
    
    async def complete_task(
        self,
        task_id: str,
        agent_id: str,
        success: bool = True,
        result: Dict[str, Any] = None,
        feedback: Dict[str, Any] = None
    ) -> bool:
        """
        Mark a task as completed
        
        Args:
            task_id: Task identifier
            agent_id: Agent that completed the task
            success: Whether task was successful
            result: Task result data
            feedback: Feedback for agent learning
        
        Returns:
            True if completion processed successfully
        """
        
        try:
            # Complete task in hierarchy
            completion_success = await self.agent_hierarchy.complete_task_for_agent(
                agent_id, task_id, success, feedback
            )
            
            if completion_success:
                # Send completion notification
                message_type = MessageType.TASK_COMPLETION if success else MessageType.TASK_FAILURE
                
                await self.communication_hub.send_message(
                    sender_id=agent_id,
                    recipient_id=self.agent_hierarchy.boss_agent.agent_id,
                    message_type=message_type,
                    subject=f"Task {'Completed' if success else 'Failed'}: {task_id}",
                    content={
                        "task_id": task_id,
                        "success": success,
                        "result": result or {},
                        "completion_time": datetime.utcnow().isoformat(),
                        "agent_feedback": feedback or {}
                    },
                    priority=MessagePriority.HIGH if not success else MessagePriority.NORMAL
                )
                
                logger.info(f"Task {task_id} {'completed' if success else 'failed'} by agent {agent_id}")
                
            return completion_success
            
        except Exception as e:
            logger.error(f"Error completing task {task_id}: {e}")
            return False
    
    async def spawn_agent_if_needed(self) -> Dict[str, Any]:
        """
        Evaluate and spawn agents if needed
        
        Returns:
            Spawning result
        """
        
        try:
            result = await self.agent_spawner.auto_spawn_if_needed()
            
            # If agents were spawned, register them and set up communication
            if result.get("execution", {}).get("agents_created"):
                for agent_info in result["execution"]["agents_created"]:
                    agent_id = agent_info["agent_id"]
                    
                    # Create communication queue
                    self.communication_hub.create_agent_queue(agent_id)
                    
                    # Register agent
                    agent = self.agent_hierarchy.get_agent_by_id(agent_id)
                    if agent:
                        self.agent_registry.register_agent(agent)
                        self.system_metrics["agent_spawns"] += 1
                        
                        # Notify callbacks
                        for callback in self.agent_spawned_callbacks:
                            try:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(agent)
                                else:
                                    callback(agent)
                            except Exception as e:
                                logger.warning(f"Agent spawned callback failed: {e}")
                        
                        logger.info(f"New agent registered: {agent.get_human_readable_name()}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in agent spawning evaluation: {e}")
            return {"error": str(e)}
    
    async def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate an agent
        
        Args:
            agent_id: Agent to terminate
        
        Returns:
            True if termination successful
        """
        
        try:
            agent = self.agent_hierarchy.get_agent_by_id(agent_id)
            if not agent:
                return False
            
            # Cannot terminate Boss agent
            if isinstance(agent, BossAgent):
                logger.error("Cannot terminate Boss agent")
                return False
            
            # Send termination notification
            await self.communication_hub.send_message(
                sender_id=self.agent_hierarchy.boss_agent.agent_id,
                recipient_id=agent_id,
                message_type=MessageType.AGENT_TERMINATION_NOTIFICATION,
                subject="Agent Termination",
                content={
                    "message": "Your services are no longer required",
                    "termination_time": datetime.utcnow().isoformat(),
                    "reason": "System optimization"
                },
                priority=MessagePriority.HIGH
            )
            
            # Terminate in hierarchy
            success = await self.agent_hierarchy.terminate_subordinate_agent(agent_id)
            
            if success:
                # Update registry
                self.agent_registry.unregister_agent(agent_id)
                
                # Remove communication queue
                self.communication_hub.remove_agent_queue(agent_id)
                
                # Update metrics
                self.system_metrics["agent_terminations"] += 1
                
                # Notify callbacks
                for callback in self.agent_terminated_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(agent)
                        else:
                            callback(agent)
                    except Exception as e:
                        logger.warning(f"Agent terminated callback failed: {e}")
                
                logger.info(f"Agent terminated: {agent.get_human_readable_name()}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error terminating agent {agent_id}: {e}")
            return False
    
    async def rebalance_system(self) -> Dict[str, Any]:
        """
        Rebalance the agent system for optimal performance
        
        Returns:
            Rebalancing results
        """
        
        try:
            logger.info("Starting system rebalancing")
            
            rebalancing_results = {
                "timestamp": datetime.utcnow().isoformat(),
                "actions_taken": [],
                "agent_changes": [],
                "performance_improvements": {}
            }
            
            # Agent hierarchy rebalancing
            hierarchy_rebalance = await self.agent_hierarchy.rebalance_agents()
            rebalancing_results["actions_taken"].extend(hierarchy_rebalance.get("actions_taken", []))
            rebalancing_results["agent_changes"].extend(hierarchy_rebalance.get("agents_affected", []))
            
            # Communication queue cleanup
            comm_metrics = self.communication_hub.get_communication_metrics()
            total_queued = comm_metrics.get("total_queued_messages", 0)
            
            if total_queued > 100:  # High message count
                rebalancing_results["actions_taken"].append("High message queue detected - optimizing communication")
                
                # Clear low priority messages older than 1 hour
                for agent_id in list(self.communication_hub.message_queues.keys()):
                    queue = self.communication_hub.message_queues[agent_id]
                    old_messages = [
                        msg for msg in queue.messages 
                        if (datetime.utcnow() - msg.timestamp).total_seconds() > 3600
                        and msg.priority.value <= 2  # Low and Normal priority
                    ]
                    
                    for msg in old_messages:
                        queue.messages.remove(msg)
                    
                    if old_messages:
                        rebalancing_results["actions_taken"].append(f"Cleared {len(old_messages)} old messages for {agent_id}")
            
            # Performance assessment
            active_agents = self.agent_hierarchy.get_active_subordinates()
            if active_agents:
                avg_success_rate = sum(agent.metrics.success_rate for agent in active_agents) / len(active_agents)
                avg_load = sum(agent.get_load_percentage() for agent in active_agents) / len(active_agents)
                
                rebalancing_results["performance_improvements"] = {
                    "average_success_rate": avg_success_rate,
                    "average_load_percentage": avg_load,
                    "recommendations": []
                }
                
                if avg_success_rate < 0.8:
                    rebalancing_results["performance_improvements"]["recommendations"].append("Consider agent training or capability enhancement")
                
                if avg_load > 80:
                    rebalancing_results["performance_improvements"]["recommendations"].append("Consider spawning additional agents to reduce load")
                elif avg_load < 30:
                    rebalancing_results["performance_improvements"]["recommendations"].append("Consider terminating underutilized agents")
            
            logger.info(f"System rebalancing completed with {len(rebalancing_results['actions_taken'])} actions")
            return rebalancing_results
            
        except Exception as e:
            logger.error(f"System rebalancing failed: {e}")
            return {"error": str(e)}
    
    def _setup_communication_handlers(self):
        """Setup handlers for communication messages"""
        
        async def handle_task_update(message):
            """Handle task update messages"""
            content = message.content
            task_id = content.get("task_id")
            update_type = content.get("update_type", "progress")
            
            logger.debug(f"Task update received for {task_id}: {update_type}")
            
            return {"handled": True, "action": "task_update_processed"}
        
        async def handle_error_report(message):
            """Handle error reports from agents"""
            content = message.content
            error_type = content.get("error_type", "unknown")
            error_details = content.get("error_details", "")
            
            logger.warning(f"Error reported by {message.sender_id}: {error_type} - {error_details}")
            
            # Could implement automatic error handling here
            
            return {"handled": True, "action": "error_logged"}
        
        async def handle_capability_inquiry(message):
            """Handle capability inquiries"""
            content = message.content
            requested_capability = content.get("capability", "")
            
            # Find agents with requested capability
            capable_agents = self.agent_hierarchy.get_agents_with_capability(requested_capability)
            
            # Send response
            await self.communication_hub.send_response(
                original_message_id=message.message_id,
                sender_id=self.agent_hierarchy.boss_agent.agent_id,
                response_content={
                    "requested_capability": requested_capability,
                    "capable_agents": [
                        {
                            "agent_id": agent.agent_id,
                            "agent_name": agent.get_human_readable_name(),
                            "availability": agent.is_available()
                        }
                        for agent in capable_agents
                    ]
                },
                success=True
            )
            
            return {"handled": True, "action": "capability_inquiry_responded"}
        
        # Register handlers
        self.communication_hub.add_message_handler(MessageType.TASK_UPDATE, handle_task_update)
        self.communication_hub.add_message_handler(MessageType.ERROR_REPORT, handle_error_report)
        self.communication_hub.add_message_handler(MessageType.CAPABILITY_INQUIRY, handle_capability_inquiry)
    
    async def _autonomous_management_loop(self):
        """Main autonomous management loop"""
        
        logger.info("Autonomous management loop started")
        
        try:
            while self.is_running:
                # Evaluate spawning needs every 5 minutes
                await asyncio.sleep(300)  # 5 minutes
                
                if not self.is_running:
                    break
                
                try:
                    # Check if agents need to be spawned
                    await self.spawn_agent_if_needed()
                    
                    # Perform system rebalancing every 30 minutes
                    if hasattr(self, '_last_rebalance'):
                        time_since_rebalance = datetime.utcnow() - self._last_rebalance
                        if time_since_rebalance.total_seconds() > 1800:  # 30 minutes
                            await self.rebalance_system()
                            self._last_rebalance = datetime.utcnow()
                    else:
                        self._last_rebalance = datetime.utcnow()
                    
                    # Clean up communication metrics
                    self.system_metrics["communication_messages"] = self.communication_hub.metrics["total_messages"]
                    
                except Exception as loop_error:
                    logger.warning(f"Error in autonomous management loop: {loop_error}")
                
        except asyncio.CancelledError:
            logger.info("Autonomous management loop cancelled")
        except Exception as e:
            logger.error(f"Autonomous management loop error: {e}")
        
        logger.info("Autonomous management loop ended")
    
    # Public interface methods
    
    def add_agent_spawned_callback(self, callback: Callable):
        """Add callback for when agents are spawned"""
        self.agent_spawned_callbacks.append(callback)
    
    def add_agent_terminated_callback(self, callback: Callable):
        """Add callback for when agents are terminated"""
        self.agent_terminated_callbacks.append(callback)
    
    def add_task_delegated_callback(self, callback: Callable):
        """Add callback for when tasks are delegated"""
        self.task_delegated_callbacks.append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        try:
            hierarchy_summary = self.agent_hierarchy.get_hierarchy_summary()
            performance_summary = self.agent_hierarchy.get_agent_performance_summary()
            registry_summary = self.agent_registry.get_registry_summary()
            spawning_metrics = self.agent_spawner.get_spawning_metrics()
            communication_metrics = self.communication_hub.get_communication_metrics()
            
            # Calculate uptime
            uptime_seconds = 0
            if self.system_metrics["uptime_start"]:
                uptime_seconds = (datetime.utcnow() - self.system_metrics["uptime_start"]).total_seconds()
            
            return {
                "system_info": {
                    "is_initialized": self.is_initialized,
                    "is_running": self.is_running,
                    "uptime_seconds": uptime_seconds,
                    "spawning_strategy": self.spawning_strategy.value,
                    "task_distribution_strategy": self.task_distribution_strategy.value
                },
                "hierarchy_summary": hierarchy_summary,
                "performance_summary": performance_summary,
                "registry_summary": registry_summary,
                "spawning_metrics": spawning_metrics,
                "communication_metrics": communication_metrics,
                "system_metrics": self.system_metrics
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_agent_details(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific agent"""
        
        try:
            agent = self.agent_hierarchy.get_agent_by_id(agent_id)
            if not agent:
                return None
            
            # Get message summary
            message_summary = self.communication_hub.get_agent_message_summary(agent_id)
            
            return {
                "agent_info": agent.dict(),
                "human_readable_name": agent.get_human_readable_name(),
                "load_percentage": agent.get_load_percentage(),
                "is_available": agent.is_available(),
                "capability_names": agent.get_capability_names(),
                "message_summary": message_summary,
                "registry_info": self.agent_registry.agents.get(agent_id, {})
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def update_task_distribution_strategy(self, new_strategy: TaskDistributionStrategy):
        """Update task distribution strategy"""
        
        old_strategy = self.task_distribution_strategy
        self.task_distribution_strategy = new_strategy
        
        logger.info(f"Task distribution strategy updated from {old_strategy.value} to {new_strategy.value}")
    
    def update_spawning_strategy(self, new_strategy: SpawningStrategy):
        """Update spawning strategy"""
        
        self.agent_spawner.update_strategy(new_strategy)
        self.spawning_strategy = new_strategy
        
        logger.info(f"Spawning strategy updated to {new_strategy.value}")
    
    async def force_spawn_agent(
        self,
        role: AgentRole = AgentRole.GENERALIST,
        capabilities: List[str] = None,
        specialization: str = None
    ) -> Dict[str, Any]:
        """Force spawn an agent bypassing normal logic"""
        
        result = await self.agent_spawner.force_spawn_agent(role, capabilities, specialization)
        
        # Register if successful
        if result.get("success") and result.get("agents_created"):
            for agent_info in result["agents_created"]:
                agent_id = agent_info["agent_id"]
                agent = self.agent_hierarchy.get_agent_by_id(agent_id)
                
                if agent:
                    self.agent_registry.register_agent(agent)
                    self.communication_hub.create_agent_queue(agent_id)
                    self.system_metrics["agent_spawns"] += 1
        
        return result
    
    def get_delegation_history(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get recent task delegation history"""
        
        # This would typically come from a persistent store
        # For now, return summary based on current metrics
        
        return [{
            "summary": "Task delegation metrics",
            "total_delegated": self.system_metrics["total_tasks_delegated"],
            "successful": self.system_metrics["successful_delegations"],
            "failed": self.system_metrics["failed_delegations"],
            "success_rate": (
                self.system_metrics["successful_delegations"] / 
                max(1, self.system_metrics["total_tasks_delegated"])
            ) * 100,
            "strategy": self.task_distribution_strategy.value
        }]
