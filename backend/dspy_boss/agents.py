
"""
Agent management system for DSPY Boss
"""

import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import uuid

import dspy
from dspy import Signature, InputField, OutputField

from .models import AgentConfig, AgentType, TaskDefinition, TaskStatus, PromptSignature
from .task_manager import TaskManager
from .mcp import MCPManager


class AgentSignature(Signature):
    """Base DSPY signature for agents"""
    task_description = InputField(desc="Description of the task to be performed")
    context = InputField(desc="Additional context and information")
    result = OutputField(desc="Result of the task execution")
    reasoning = OutputField(desc="Reasoning behind the result")


class ReactAgentSignature(Signature):
    """React agent signature for step-by-step reasoning"""
    question = InputField(desc="Question or task to solve")
    context = InputField(desc="Available context and tools")
    thought = OutputField(desc="Current thought process")
    action = OutputField(desc="Action to take")
    observation = OutputField(desc="Observation from the action")
    answer = OutputField(desc="Final answer or result")


class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, config: AgentConfig, task_manager: TaskManager, mcp_manager: MCPManager):
        self.config = config
        self.task_manager = task_manager
        self.mcp_manager = mcp_manager
        
        self.is_active = False
        self.current_tasks: List[str] = []
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Performance metrics
        self.total_tasks_processed = 0
        self.success_rate = 0.0
        self.average_task_duration = 0.0
        self.last_active = datetime.utcnow()
        
        logger.info(f"Initialized {self.config.type.value} agent: {self.config.name}")
    
    def can_accept_task(self, task: TaskDefinition) -> bool:
        """Check if agent can accept a new task"""
        if not self.config.is_available:
            return False
        
        if len(self.current_tasks) >= self.config.max_concurrent_tasks:
            return False
        
        # Check if agent has required capabilities
        if hasattr(task, 'required_capabilities'):
            for capability in task.required_capabilities:
                if capability not in self.config.capabilities:
                    return False
        
        return True
    
    async def assign_task(self, task: TaskDefinition) -> bool:
        """Assign a task to this agent"""
        if not self.can_accept_task(task):
            return False
        
        self.current_tasks.append(task.id)
        task.assigned_agent_id = self.config.id
        self.last_active = datetime.utcnow()
        
        logger.info(f"Assigned task {task.name} to agent {self.config.name}")
        return True
    
    async def execute_task(self, task: TaskDefinition) -> Any:
        """Execute a task (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement execute_task")
    
    def start(self):
        """Start the agent in a separate thread"""
        if self.is_active:
            return
        
        self.is_active = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"Started agent: {self.config.name}")
    
    def stop(self):
        """Stop the agent"""
        if not self.is_active:
            return
        
        self.is_active = False
        self.stop_event.set()
        
        if self.thread:
            self.thread.join(timeout=5)
        
        logger.info(f"Stopped agent: {self.config.name}")
    
    def _run_loop(self):
        """Main execution loop for the agent"""
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        
        try:
            loop.run_until_complete(self._async_run_loop())
        except Exception as e:
            logger.error(f"Error in agent {self.config.name} run loop: {e}")
        finally:
            loop.close()
    
    async def _async_run_loop(self):
        """Async run loop"""
        while self.is_active and not self.stop_event.is_set():
            try:
                # Check for assigned tasks
                if self.current_tasks:
                    task_id = self.current_tasks[0]  # Process first task
                    task = self.task_manager.get_task_status(task_id)
                    
                    if task and task.status == TaskStatus.PENDING:
                        await self._process_task(task)
                
                await asyncio.sleep(1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error in agent {self.config.name} async loop: {e}")
                await asyncio.sleep(5)  # Longer delay on error
    
    async def _process_task(self, task: TaskDefinition):
        """Process a single task"""
        start_time = datetime.utcnow()
        
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = start_time
            
            # Execute the task
            result = await self.execute_task(task)
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.result = result
            
            # Update metrics
            duration = (task.completed_at - start_time).total_seconds()
            self._update_metrics(True, duration)
            
            # Move task to completed
            self.current_tasks.remove(task.id)
            self.completed_tasks.append(task.id)
            
            logger.info(f"Agent {self.config.name} completed task: {task.name}")
            
        except Exception as e:
            # Task failed
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.error_message = str(e)
            
            # Update metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(False, duration)
            
            # Move task to failed
            self.current_tasks.remove(task.id)
            self.failed_tasks.append(task.id)
            
            logger.error(f"Agent {self.config.name} failed task {task.name}: {e}")
    
    def _update_metrics(self, success: bool, duration: float):
        """Update performance metrics"""
        self.total_tasks_processed += 1
        
        # Update success rate
        successful_tasks = len(self.completed_tasks)
        self.success_rate = (successful_tasks / self.total_tasks_processed) * 100
        
        # Update average duration
        total_duration = self.average_task_duration * (self.total_tasks_processed - 1) + duration
        self.average_task_duration = total_duration / self.total_tasks_processed
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "id": self.config.id,
            "name": self.config.name,
            "type": self.config.type.value,
            "is_active": self.is_active,
            "is_available": self.config.is_available,
            "current_tasks": len(self.current_tasks),
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
            "total_tasks_processed": self.total_tasks_processed,
            "success_rate": round(self.success_rate, 2),
            "average_task_duration": round(self.average_task_duration, 2),
            "last_active": self.last_active.isoformat(),
            "capabilities": self.config.capabilities
        }


class AgenticAgent(BaseAgent):
    """Autonomous agent using DSPY"""
    
    def __init__(self, config: AgentConfig, task_manager: TaskManager, mcp_manager: MCPManager, 
                 prompt_signatures: Dict[str, PromptSignature]):
        super().__init__(config, task_manager, mcp_manager)
        
        self.prompt_signatures = prompt_signatures
        self.dspy_module = None
        
        # Initialize DSPY module
        self._initialize_dspy_module()
    
    def _initialize_dspy_module(self):
        """Initialize DSPY module based on configuration"""
        try:
            # Get prompt signature
            signature_name = self.config.prompt_signature
            if signature_name and signature_name in self.prompt_signatures:
                prompt_sig = self.prompt_signatures[signature_name]
                
                if prompt_sig.is_react_agent:
                    self.dspy_module = dspy.ReAct(ReactAgentSignature)
                else:
                    self.dspy_module = dspy.ChainOfThought(AgentSignature)
            else:
                # Default to ChainOfThought
                self.dspy_module = dspy.ChainOfThought(AgentSignature)
            
            logger.info(f"Initialized DSPY module for agent {self.config.name}")
            
        except Exception as e:
            logger.error(f"Error initializing DSPY module for agent {self.config.name}: {e}")
            self.dspy_module = dspy.ChainOfThought(AgentSignature)
    
    async def execute_task(self, task: TaskDefinition) -> Any:
        """Execute task using DSPY"""
        if not self.dspy_module:
            raise RuntimeError("DSPY module not initialized")
        
        try:
            # Prepare context
            context = {
                "task_id": task.id,
                "parameters": task.parameters,
                "capabilities": self.config.capabilities,
                "available_mcp_servers": self.mcp_manager.get_connected_servers()
            }
            
            # Execute DSPY module
            if isinstance(self.dspy_module, dspy.ReAct):
                result = await self._execute_react_agent(task, context)
            else:
                result = await self._execute_chain_of_thought(task, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing task with DSPY: {e}")
            raise
    
    async def _execute_chain_of_thought(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute using ChainOfThought"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    lambda: self.dspy_module(
                        task_description=task.description,
                        context=str(context)
                    )
                )
            
            return {
                "result": result.result,
                "reasoning": result.reasoning,
                "method": "chain_of_thought"
            }
            
        except Exception as e:
            logger.error(f"Error in ChainOfThought execution: {e}")
            raise
    
    async def _execute_react_agent(self, task: TaskDefinition, context: Dict[str, Any]) -> Any:
        """Execute using ReAct agent"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    lambda: self.dspy_module(
                        question=task.description,
                        context=str(context)
                    )
                )
            
            return {
                "answer": result.answer,
                "thought": result.thought,
                "action": result.action,
                "observation": result.observation,
                "method": "react"
            }
            
        except Exception as e:
            logger.error(f"Error in ReAct execution: {e}")
            raise


class HumanAgent(BaseAgent):
    """Human agent that communicates via MCP servers"""
    
    def __init__(self, config: AgentConfig, task_manager: TaskManager, mcp_manager: MCPManager):
        super().__init__(config, task_manager, mcp_manager)
        
        self.pending_human_tasks: Dict[str, TaskDefinition] = {}
        self.response_timeout = 3600  # 1 hour default timeout
    
    async def execute_task(self, task: TaskDefinition) -> Any:
        """Execute task by communicating with human via MCP"""
        if not self.config.contact_method:
            raise ValueError("Human agent must have contact_method configured")
        
        try:
            # Send task to human
            await self._send_task_to_human(task)
            
            # Wait for response
            result = await self._wait_for_human_response(task)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing human task: {e}")
            raise
    
    async def _send_task_to_human(self, task: TaskDefinition):
        """Send task to human via configured contact method"""
        contact_method = self.config.contact_method
        contact_details = self.config.contact_details or {}
        
        # Prepare message
        message = self._format_task_message(task)
        
        if contact_method == "slack":
            await self._send_slack_message(message, contact_details)
        elif contact_method == "close_crm":
            await self._send_crm_message(message, contact_details)
        else:
            raise ValueError(f"Unsupported contact method: {contact_method}")
        
        # Store task as pending
        self.pending_human_tasks[task.id] = task
        logger.info(f"Sent task {task.name} to human agent {self.config.name} via {contact_method}")
    
    def _format_task_message(self, task: TaskDefinition) -> str:
        """Format task as message for human"""
        message = f"""
🤖 **Task Assignment from DSPY Boss**

**Task:** {task.name}
**Description:** {task.description}
**Priority:** {task.priority.name}
**Task ID:** {task.id}

**Parameters:**
{self._format_parameters(task.parameters)}

**Required Capabilities:** {', '.join(getattr(task, 'required_capabilities', []))}

Please complete this task and respond with your results. You can reply with:
- ✅ Success: [your results]
- ❌ Failed: [error description]
- ⏸️ Need more info: [what you need]

**Timeout:** {task.timeout} seconds (if specified)
        """.strip()
        
        return message
    
    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format task parameters for display"""
        if not parameters:
            return "None"
        
        formatted = []
        for key, value in parameters.items():
            formatted.append(f"- **{key}:** {value}")
        
        return "\n".join(formatted)
    
    async def _send_slack_message(self, message: str, contact_details: Dict[str, Any]):
        """Send message via Slack MCP server"""
        slack_servers = self.mcp_manager.find_servers_by_capability("messaging")
        
        if not slack_servers:
            raise RuntimeError("No Slack MCP server available")
        
        server_name = slack_servers[0]  # Use first available
        
        # Prepare Slack message data
        data = {
            "channel": contact_details.get("channel", "#general"),
            "text": message,
            "username": "DSPY Boss"
        }
        
        response = await self.mcp_manager.send_request(server_name, "POST", "chat.postMessage", data)
        
        if not response.success:
            raise RuntimeError(f"Failed to send Slack message: {response.error}")
    
    async def _send_crm_message(self, message: str, contact_details: Dict[str, Any]):
        """Send message via Close CRM MCP server"""
        crm_servers = self.mcp_manager.find_servers_by_capability("crm")
        
        if not crm_servers:
            raise RuntimeError("No CRM MCP server available")
        
        server_name = crm_servers[0]  # Use first available
        
        # Prepare CRM message data
        data = {
            "user_id": contact_details.get("user_id"),
            "message": message,
            "subject": f"Task Assignment: {contact_details.get('task_name', 'New Task')}"
        }
        
        response = await self.mcp_manager.send_request(server_name, "POST", "messages", data)
        
        if not response.success:
            raise RuntimeError(f"Failed to send CRM message: {response.error}")
    
    async def _wait_for_human_response(self, task: TaskDefinition) -> Any:
        """Wait for human response (simplified - in real implementation would listen for responses)"""
        # This is a simplified implementation
        # In a real system, you would set up webhooks or polling to listen for responses
        
        timeout = task.timeout or self.response_timeout
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            # Check if response received (this would be implemented with actual MCP response handling)
            # For now, we'll simulate waiting
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # In real implementation, check for responses from MCP servers
            # and parse them to determine if task is complete
            
        # If we get here, timeout occurred
        raise TimeoutError(f"Human agent {self.config.name} did not respond within {timeout} seconds")
    
    def receive_human_response(self, task_id: str, response: str, success: bool = True):
        """Receive response from human (called by MCP message handlers)"""
        if task_id in self.pending_human_tasks:
            task = self.pending_human_tasks[task_id]
            
            if success:
                task.status = TaskStatus.COMPLETED
                task.result = response
            else:
                task.status = TaskStatus.FAILED
                task.error_message = response
            
            task.completed_at = datetime.utcnow()
            del self.pending_human_tasks[task_id]
            
            logger.info(f"Received human response for task {task_id}: {'Success' if success else 'Failed'}")


class AgentManager:
    """Manages all agents in the system"""
    
    def __init__(self, task_manager: TaskManager, mcp_manager: MCPManager):
        self.task_manager = task_manager
        self.mcp_manager = mcp_manager
        
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.prompt_signatures: Dict[str, PromptSignature] = {}
        
        # Agent spawning settings
        self.max_agentic_agents = 5
        self.agent_spawn_threshold = 8
        self.next_agent_number = 1
        
        logger.info("AgentManager initialized")
    
    def load_agents(self, agent_configs: Dict[str, AgentConfig], 
                   prompt_signatures: Dict[str, PromptSignature]):
        """Load agent configurations"""
        self.agent_configs = agent_configs
        self.prompt_signatures = prompt_signatures
        
        # Create agents from configs
        for name, config in agent_configs.items():
            self.create_agent(config)
        
        logger.info(f"Loaded {len(self.agents)} agents")
    
    def create_agent(self, config: AgentConfig) -> BaseAgent:
        """Create an agent from configuration"""
        if config.type == AgentType.AGENTIC:
            agent = AgenticAgent(config, self.task_manager, self.mcp_manager, self.prompt_signatures)
        elif config.type == AgentType.HUMAN:
            agent = HumanAgent(config, self.task_manager, self.mcp_manager)
        else:
            raise ValueError(f"Unknown agent type: {config.type}")
        
        self.agents[config.id] = agent
        agent.start()
        
        logger.info(f"Created and started agent: {config.name}")
        return agent
    
    def spawn_agentic_agent(self, capabilities: List[str] = None) -> Optional[BaseAgent]:
        """Spawn a new agentic agent when workload is high"""
        if len([a for a in self.agents.values() if a.config.type == AgentType.AGENTIC]) >= self.max_agentic_agents:
            logger.warning("Maximum number of agentic agents reached")
            return None
        
        # Create new agent config
        agent_config = AgentConfig(
            name=f"Agent-{self.next_agent_number}",
            type=AgentType.AGENTIC,
            description=f"Auto-spawned agentic agent #{self.next_agent_number}",
            capabilities=capabilities or ["general", "analysis", "research"],
            model_name="gpt-3.5-turbo",
            prompt_signature="data_analysis_react"
        )
        
        self.next_agent_number += 1
        
        # Create and start agent
        agent = self.create_agent(agent_config)
        
        logger.info(f"Spawned new agentic agent: {agent_config.name}")
        return agent
    
    def assign_task_to_best_agent(self, task: TaskDefinition) -> Optional[BaseAgent]:
        """Assign task to the most suitable available agent"""
        suitable_agents = []
        
        # Find agents that can handle the task
        for agent in self.agents.values():
            if agent.can_accept_task(task):
                # Calculate suitability score
                score = self._calculate_agent_suitability(agent, task)
                suitable_agents.append((agent, score))
        
        if not suitable_agents:
            # Check if we should spawn a new agent
            workload = len(self.state_manager.transition.state_data.pending_tasks)
            if workload >= self.agent_spawn_threshold:
                new_agent = self.spawn_agentic_agent()
                if new_agent and new_agent.can_accept_task(task):
                    suitable_agents.append((new_agent, 1.0))
        
        if not suitable_agents:
            logger.warning(f"No suitable agent found for task: {task.name}")
            return None
        
        # Sort by suitability score (higher is better)
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        best_agent = suitable_agents[0][0]
        
        # Assign task
        asyncio.create_task(best_agent.assign_task(task))
        
        logger.info(f"Assigned task {task.name} to agent {best_agent.config.name}")
        return best_agent
    
    def _calculate_agent_suitability(self, agent: BaseAgent, task: TaskDefinition) -> float:
        """Calculate how suitable an agent is for a task"""
        score = 0.0
        
        # Base score for availability
        if agent.config.is_available:
            score += 1.0
        
        # Capability matching
        task_capabilities = getattr(task, 'required_capabilities', [])
        if task_capabilities:
            matching_capabilities = set(agent.config.capabilities) & set(task_capabilities)
            score += len(matching_capabilities) / len(task_capabilities)
        
        # Performance history
        if agent.total_tasks_processed > 0:
            score += agent.success_rate / 100.0
        
        # Current workload (prefer less busy agents)
        workload_factor = 1.0 - (len(agent.current_tasks) / agent.config.max_concurrent_tasks)
        score += workload_factor
        
        # Human vs Agentic preference
        if task.requires_human and agent.config.type == AgentType.HUMAN:
            score += 2.0
        elif not task.requires_human and agent.config.type == AgentType.AGENTIC:
            score += 1.0
        
        return score
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics for all agents"""
        stats = {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.is_active]),
            "agentic_agents": len([a for a in self.agents.values() if a.config.type == AgentType.AGENTIC]),
            "human_agents": len([a for a in self.agents.values() if a.config.type == AgentType.HUMAN]),
            "agents": {}
        }
        
        for agent_id, agent in self.agents.items():
            stats["agents"][agent_id] = agent.get_status()
        
        return stats
    
    def stop_all_agents(self):
        """Stop all agents"""
        logger.info("Stopping all agents...")
        
        for agent in self.agents.values():
            agent.stop()
        
        logger.info("All agents stopped")
    
    def remove_idle_agents(self, idle_timeout: int = 1800):
        """Remove agents that have been idle for too long"""
        current_time = datetime.utcnow()
        agents_to_remove = []
        
        for agent_id, agent in self.agents.items():
            if (agent.config.type == AgentType.AGENTIC and 
                len(agent.current_tasks) == 0 and
                (current_time - agent.last_active).total_seconds() > idle_timeout):
                agents_to_remove.append(agent_id)
        
        for agent_id in agents_to_remove:
            agent = self.agents[agent_id]
            agent.stop()
            del self.agents[agent_id]
            logger.info(f"Removed idle agent: {agent.config.name}")
        
        return len(agents_to_remove)
