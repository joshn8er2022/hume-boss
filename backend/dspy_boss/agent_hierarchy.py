
"""
Agent Hierarchy - Manages Boss (Agent 0) and subordinate agents with proper numbering
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
from dataclasses import dataclass
from enum import Enum

import dspy
from dspy import Signature, InputField, OutputField, ChainOfThought

from .models import AgentConfig, AgentType, TaskDefinition


class AgentRole(str, Enum):
    BOSS = "boss"          # Agent 0
    SUBORDINATE = "subordinate"  # Agent 1, 2, 3, etc.


@dataclass
class AgentInfo:
    agent_id: int          # 0 for boss, 1, 2, 3... for subordinates
    agent_name: str        # Human readable name
    role: AgentRole
    status: str           # "active", "idle", "busy", "error"
    current_tasks: List[str]
    completed_tasks: int
    created_at: datetime
    last_active: datetime
    capabilities: List[str]
    performance_score: float


class SubordinateAgentSignature(Signature):
    """Signature for subordinate agent task execution"""
    task_description = InputField(desc="Task assigned by the boss")
    context = InputField(desc="Context and resources available")
    agent_capabilities = InputField(desc="This agent's specific capabilities")
    
    result = OutputField(desc="Task execution result")
    status = OutputField(desc="Task completion status")
    feedback = OutputField(desc="Feedback for the boss about task execution")
    resource_usage = OutputField(desc="Resources used during task execution")


class AutonomousAgent:
    """Individual autonomous agent powered by DSPY"""
    
    def __init__(self, agent_info: AgentInfo):
        self.info = agent_info
        self.signature = ChainOfThought(SubordinateAgentSignature)
        self.is_busy = False
        self.current_task: Optional[str] = None
        
    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task autonomously"""
        if self.is_busy:
            return {"error": "Agent is currently busy", "agent_id": self.info.agent_id}
            
        self.is_busy = True
        self.current_task = task
        self.info.last_active = datetime.now()
        
        try:
            logger.info(f"🤖 Agent {self.info.agent_id} ({self.info.agent_name}) executing task: {task}")
            
            # Use DSPY signature to execute task
            result = self.signature(
                task_description=task,
                context=str(context or {}),
                agent_capabilities=", ".join(self.info.capabilities)
            )
            
            # Update agent stats
            self.info.completed_tasks += 1
            self.info.status = "idle"
            
            task_result = {
                "agent_id": self.info.agent_id,
                "agent_name": self.info.agent_name,
                "task": task,
                "result": result.result,
                "status": result.status,
                "feedback": result.feedback,
                "resource_usage": result.resource_usage,
                "completion_time": datetime.now().isoformat(),
                "success": True
            }
            
            logger.info(f"✅ Agent {self.info.agent_id} completed task successfully")
            return task_result
            
        except Exception as e:
            logger.error(f"❌ Agent {self.info.agent_id} failed task: {e}")
            self.info.status = "error"
            return {
                "agent_id": self.info.agent_id,
                "agent_name": self.info.agent_name,
                "task": task,
                "error": str(e),
                "success": False,
                "completion_time": datetime.now().isoformat()
            }
            
        finally:
            self.is_busy = False
            self.current_task = None
            
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.info.agent_id,
            "agent_name": self.info.agent_name,
            "role": self.info.role.value,
            "status": "busy" if self.is_busy else self.info.status,
            "current_task": self.current_task,
            "completed_tasks": self.info.completed_tasks,
            "capabilities": self.info.capabilities,
            "performance_score": self.info.performance_score,
            "last_active": self.info.last_active.isoformat(),
            "uptime": (datetime.now() - self.info.created_at).total_seconds()
        }


class AgentHierarchy:
    """Manages the hierarchy of agents with Boss as Agent 0"""
    
    def __init__(self):
        self.agents: Dict[int, AutonomousAgent] = {}
        self.next_agent_id = 1  # Boss is 0, subordinates start from 1
        self.boss_agent: Optional[AutonomousAgent] = None
        
        # Create the Boss (Agent 0)
        self._create_boss_agent()
        
    def _create_boss_agent(self):
        """Create the Boss agent (Agent 0)"""
        boss_info = AgentInfo(
            agent_id=0,
            agent_name="Boss Agent",
            role=AgentRole.BOSS,
            status="active",
            current_tasks=[],
            completed_tasks=0,
            created_at=datetime.now(),
            last_active=datetime.now(),
            capabilities=[
                "strategic_decision_making",
                "task_delegation",
                "system_orchestration", 
                "agent_management",
                "priority_assessment",
                "resource_allocation"
            ],
            performance_score=1.0
        )
        
        self.boss_agent = AutonomousAgent(boss_info)
        self.agents[0] = self.boss_agent
        
        logger.info("👑 Boss Agent (Agent 0) created and ready")
        
    async def get_or_create_agent(self, agent_name: str) -> AutonomousAgent:
        """Get existing agent or create new subordinate agent"""
        # Check if agent already exists by name
        for agent in self.agents.values():
            if agent.info.agent_name == agent_name and agent.info.role == AgentRole.SUBORDINATE:
                return agent
                
        # Create new subordinate agent
        return await self._create_subordinate_agent(agent_name)
        
    async def _create_subordinate_agent(self, agent_name: str) -> AutonomousAgent:
        """Create a new subordinate agent with proper numbering"""
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        
        # Determine capabilities based on agent name/type
        capabilities = self._determine_agent_capabilities(agent_name)
        
        agent_info = AgentInfo(
            agent_id=agent_id,
            agent_name=agent_name,
            role=AgentRole.SUBORDINATE,
            status="idle",
            current_tasks=[],
            completed_tasks=0,
            created_at=datetime.now(),
            last_active=datetime.now(),
            capabilities=capabilities,
            performance_score=0.8  # Default starting score
        )
        
        agent = AutonomousAgent(agent_info)
        self.agents[agent_id] = agent
        
        logger.info(f"🤖 Created Agent {agent_id} ({agent_name}) as subordinate to Boss")
        return agent
        
    def _determine_agent_capabilities(self, agent_name: str) -> List[str]:
        """Determine agent capabilities based on name/type"""
        # Simple capability assignment - in production this would be more sophisticated
        base_capabilities = ["task_execution", "problem_solving", "data_processing"]
        
        if "trading" in agent_name.lower():
            return base_capabilities + ["market_analysis", "order_execution", "risk_management"]
        elif "analysis" in agent_name.lower():
            return base_capabilities + ["data_analysis", "pattern_recognition", "reporting"]
        elif "research" in agent_name.lower():
            return base_capabilities + ["web_research", "information_gathering", "summarization"]
        elif "communication" in agent_name.lower():
            return base_capabilities + ["message_processing", "notification_handling", "user_interaction"]
        else:
            return base_capabilities + ["general_purpose", "adaptable_execution"]
            
    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents for task assignment"""
        available = []
        for agent in self.agents.values():
            status = agent.get_status()
            if status["status"] in ["idle", "active"]:  # Available for new tasks
                available.append(status)
        return available
        
    def get_active_agents(self) -> List[Dict[str, Any]]:
        """Get list of currently active agents"""
        active = []
        for agent in self.agents.values():
            if agent.is_busy or agent.info.status == "active":
                active.append(agent.get_status())
        return active
        
    def get_all_statuses(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "total_agents": len(self.agents),
            "boss_status": self.boss_agent.get_status() if self.boss_agent else None,
            "subordinate_count": len(self.agents) - 1,  # Exclude boss
            "agents": {
                agent_id: agent.get_status() 
                for agent_id, agent in self.agents.items()
            }
        }
        
    def get_agent_by_id(self, agent_id: int) -> Optional[AutonomousAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
        
    def get_recommended_agents(self) -> List[str]:
        """Get recommended agents for next iteration"""
        recommendations = []
        
        # Always include boss
        recommendations.append("Boss Agent")
        
        # Recommend idle agents with good performance
        for agent in self.agents.values():
            if (agent.info.role == AgentRole.SUBORDINATE and 
                agent.info.status == "idle" and 
                agent.info.performance_score > 0.7):
                recommendations.append(agent.info.agent_name)
                
        return recommendations
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert hierarchy to dictionary representation"""
        return {
            "boss_agent": self.boss_agent.get_status() if self.boss_agent else None,
            "subordinate_agents": [
                agent.get_status() 
                for agent in self.agents.values() 
                if agent.info.role == AgentRole.SUBORDINATE
            ],
            "total_agents": len(self.agents),
            "next_agent_id": self.next_agent_id,
            "hierarchy_established": datetime.now().isoformat()
        }
        
    async def scale_agents(self, target_count: int):
        """Scale number of subordinate agents"""
        current_subordinates = len([a for a in self.agents.values() if a.info.role == AgentRole.SUBORDINATE])
        
        if target_count > current_subordinates:
            # Create new agents
            for i in range(target_count - current_subordinates):
                await self._create_subordinate_agent(f"Agent {self.next_agent_id}")
                
        elif target_count < current_subordinates:
            # Remove idle agents (but keep boss)
            agents_to_remove = []
            for agent_id, agent in self.agents.items():
                if (agent.info.role == AgentRole.SUBORDINATE and 
                    not agent.is_busy and 
                    len(agents_to_remove) < (current_subordinates - target_count)):
                    agents_to_remove.append(agent_id)
                    
            for agent_id in agents_to_remove:
                del self.agents[agent_id]
                logger.info(f"🗑️ Removed Agent {agent_id}")
                
        logger.info(f"📊 Scaled to {target_count} subordinate agents (+ Boss)")
        
    def get_agent_display_info(self) -> List[Dict[str, Any]]:
        """Get agent info for UI display with human-readable numbering"""
        display_info = []
        
        # Sort agents by ID to ensure proper display order
        sorted_agents = sorted(self.agents.items())
        
        for agent_id, agent in sorted_agents:
            status = agent.get_status()
            
            # Human-readable display name
            if agent.info.role == AgentRole.BOSS:
                display_name = "Boss (Agent 0)"
            else:
                display_name = f"Agent {agent_id}"
                
            display_info.append({
                "display_name": display_name,
                "internal_id": agent_id,
                "agent_name": agent.info.agent_name,
                "role": agent.info.role.value,
                "status": status["status"],
                "completed_tasks": status["completed_tasks"],
                "capabilities": status["capabilities"],
                "performance_score": status["performance_score"],
                "last_active": status["last_active"],
                "current_task": status.get("current_task")
            })
            
        return display_info
