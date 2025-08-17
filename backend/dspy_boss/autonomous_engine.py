
"""
Autonomous DSPY Engine - The core decision-making system driven by DSPY signatures
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Type
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass, asdict
from enum import Enum

import dspy
from dspy import Signature, InputField, OutputField, ChainOfThought, Retrieve

from .models import SystemState, AgentHierarchy, TaskDefinition, IterationResult
from .state_holder import StateHolder
from .llm_providers import LLMProviderManager
from .mcp import MCPManager


class IterationPhase(str, Enum):
    PRE_PROCESSING = "pre_processing"
    BOSS_DECISION = "boss_decision"
    EXECUTION = "execution" 
    ERROR_HANDLING = "error_handling"
    NEXT_ITERATION_PREP = "next_iteration_prep"


class BossDecisionSignature(Signature):
    """DSPY Signature for autonomous boss decision making"""
    current_state = InputField(desc="Current system state with all available context")
    previous_states = InputField(desc="Historical states (last 100 iterations)")
    available_agents = InputField(desc="List of available subordinate agents")
    mcp_data = InputField(desc="Data retrieved from MCP servers")
    
    decision = OutputField(desc="The autonomous decision to make")
    agent_assignments = OutputField(desc="Which agents to assign to which tasks")
    priority_tasks = OutputField(desc="Priority ordered list of tasks to execute")
    future_state_forecast = OutputField(desc="Predicted future states and their planned actions")
    reasoning = OutputField(desc="Detailed reasoning behind this decision")


class PreProcessingSignature(Signature):
    """Pre-processing signature for iteration preparation"""
    raw_system_data = InputField(desc="Raw system data from sensors and MCP servers")
    previous_iteration_results = InputField(desc="Results from the previous iteration")
    historical_patterns = InputField(desc="Patterns identified from historical states")
    
    processed_context = OutputField(desc="Processed context ready for boss decision")
    priority_insights = OutputField(desc="Key insights that should influence decisions")
    environmental_changes = OutputField(desc="Any significant changes in the environment")


class ForecastingSignature(Signature):
    """Future state forecasting signature"""
    current_state = InputField(desc="Current system state")
    planned_actions = InputField(desc="Actions planned for execution")
    historical_outcomes = InputField(desc="Historical outcomes of similar actions")
    
    future_states = OutputField(desc="Forecasted future states (next 5-10 iterations)")
    potential_risks = OutputField(desc="Potential risks and mitigation strategies")
    success_probabilities = OutputField(desc="Probability estimates for different outcomes")


@dataclass
class IterationContext:
    iteration_id: int
    phase: IterationPhase
    timestamp: datetime
    system_state: Dict[str, Any]
    pre_processing_result: Optional[Dict[str, Any]] = None
    boss_decision: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None
    next_prep_result: Optional[Dict[str, Any]] = None


class AutonomousEngine:
    """The core autonomous decision-making engine powered by DSPY signatures"""
    
    def __init__(self, state_holder: StateHolder, llm_manager: LLMProviderManager, mcp_manager: MCPManager):
        self.state_holder = state_holder
        self.llm_manager = llm_manager
        self.mcp_manager = mcp_manager
        
        # DSPY modules - these ARE the intelligence
        self.pre_processor = ChainOfThought(PreProcessingSignature)
        self.boss_brain = ChainOfThought(BossDecisionSignature)  # The boss ReAct agent
        self.forecaster = ChainOfThought(ForecastingSignature)
        
        # Agent hierarchy - Boss is Agent 0
        self.agent_hierarchy = AgentHierarchy()
        
        # Autonomous operation control
        self.is_autonomous = False
        self.iteration_count = 0
        self.current_context: Optional[IterationContext] = None
        
        # Background tasks
        self.autonomous_task: Optional[asyncio.Task] = None
        
    async def start_autonomous_operation(self):
        """Start the autonomous decision-making loop"""
        logger.info("🚀 Starting autonomous DSPY-driven operation")
        self.is_autonomous = True
        self.autonomous_task = asyncio.create_task(self._autonomous_loop())
        
    async def stop_autonomous_operation(self):
        """Stop autonomous operation"""
        logger.info("⏹️ Stopping autonomous operation")
        self.is_autonomous = False
        if self.autonomous_task:
            self.autonomous_task.cancel()
            try:
                await self.autonomous_task
            except asyncio.CancelledError:
                pass
                
    async def _autonomous_loop(self):
        """Main autonomous iteration loop"""
        while self.is_autonomous:
            try:
                # Start new iteration
                self.iteration_count += 1
                context = IterationContext(
                    iteration_id=self.iteration_count,
                    phase=IterationPhase.PRE_PROCESSING,
                    timestamp=datetime.now(),
                    system_state=await self._gather_system_state()
                )
                self.current_context = context
                
                # === PRE-PROCESSING PHASE ===
                await self._pre_processing_phase(context)
                
                # === BOSS DECISION PHASE ===
                await self._boss_decision_phase(context)
                
                # === EXECUTION PHASE ===
                await self._execution_phase(context)
                
                # Store iteration result
                await self._store_iteration_result(context)
                
            except Exception as e:
                logger.error(f"❌ Error in autonomous iteration {self.iteration_count}: {e}")
                await self._handle_iteration_error(e)
                
            finally:
                # === NEXT ITERATION PREPARATION ===
                await self._next_iteration_prep_phase(context)
                
                # Brief pause before next iteration
                await asyncio.sleep(1)  # Adjust timing as needed
                
    async def _pre_processing_phase(self, context: IterationContext):
        """Pre-processing phase - prepare context for boss decision"""
        context.phase = IterationPhase.PRE_PROCESSING
        logger.info(f"🔄 Pre-processing iteration {context.iteration_id}")
        
        # Gather data
        raw_data = await self._gather_raw_system_data()
        previous_results = await self._get_previous_iteration_results()
        historical_patterns = self.state_holder.get_historical_patterns()
        
        # Use DSPY signature for pre-processing
        result = self.pre_processor(
            raw_system_data=json.dumps(raw_data, default=str),
            previous_iteration_results=json.dumps(previous_results, default=str),
            historical_patterns=json.dumps(historical_patterns, default=str)
        )
        
        context.pre_processing_result = {
            "processed_context": result.processed_context,
            "priority_insights": result.priority_insights,
            "environmental_changes": result.environmental_changes,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Pre-processing complete: {result.priority_insights}")
        
    async def _boss_decision_phase(self, context: IterationContext):
        """Boss decision phase - autonomous ReAct agent decision making"""
        context.phase = IterationPhase.BOSS_DECISION
        logger.info(f"🧠 Boss making autonomous decision for iteration {context.iteration_id}")
        
        # Prepare input for boss brain
        current_state = await self._get_comprehensive_state()
        previous_states = self.state_holder.get_recent_states(100)  # Last 100 states
        available_agents = self.agent_hierarchy.get_available_agents()
        mcp_data = await self.mcp_manager.get_latest_data()
        
        # THE BOSS DECIDES AUTONOMOUSLY using DSPY signature
        decision = self.boss_brain(
            current_state=json.dumps(current_state, default=str),
            previous_states=json.dumps(previous_states, default=str),
            available_agents=json.dumps(available_agents, default=str),
            mcp_data=json.dumps(mcp_data, default=str)
        )
        
        context.boss_decision = {
            "decision": decision.decision,
            "agent_assignments": self._parse_agent_assignments(decision.agent_assignments),
            "priority_tasks": self._parse_priority_tasks(decision.priority_tasks),
            "future_forecast": decision.future_state_forecast,
            "reasoning": decision.reasoning,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"👑 Boss Decision: {decision.decision}")
        logger.info(f"🎯 Priority Tasks: {decision.priority_tasks}")
        
        # Generate future state forecast
        await self._generate_forecast(context, decision)
        
    async def _execution_phase(self, context: IterationContext):
        """Execution phase - carry out boss decisions"""
        context.phase = IterationPhase.EXECUTION
        logger.info(f"⚡ Executing boss decisions for iteration {context.iteration_id}")
        
        if not context.boss_decision:
            logger.error("No boss decision available for execution")
            return
            
        # Execute agent assignments
        execution_results = []
        for assignment in context.boss_decision["agent_assignments"]:
            try:
                result = await self._execute_agent_assignment(assignment)
                execution_results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute assignment {assignment}: {e}")
                execution_results.append({"error": str(e), "assignment": assignment})
                
        context.execution_result = {
            "results": execution_results,
            "tasks_completed": len([r for r in execution_results if "error" not in r]),
            "tasks_failed": len([r for r in execution_results if "error" in r]),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Execution complete: {context.execution_result['tasks_completed']} successful, {context.execution_result['tasks_failed']} failed")
        
    async def _next_iteration_prep_phase(self, context: IterationContext):
        """Prepare for next iteration"""
        logger.info(f"🔮 Preparing for next iteration after {context.iteration_id}")
        
        # Analyze current iteration results
        iteration_analysis = await self._analyze_iteration_performance(context)
        
        # Update system state with learnings
        await self._update_system_learnings(iteration_analysis)
        
        # Pre-compute context for next iteration (optimization)
        next_context = await self._prepare_next_iteration_context()
        
        context.next_prep_result = {
            "iteration_analysis": iteration_analysis,
            "next_context": next_context,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _handle_iteration_error(self, error: Exception):
        """Handle errors in iteration loop"""
        logger.error(f"🚨 Iteration error: {error}")
        
        if self.current_context:
            self.current_context.error_info = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": datetime.now().isoformat()
            }
            
        # Store error state for learning
        await self.state_holder.store_error_state(error, self.current_context)
        
    async def _gather_system_state(self) -> Dict[str, Any]:
        """Gather comprehensive system state"""
        return {
            "iteration_count": self.iteration_count,
            "timestamp": datetime.now().isoformat(),
            "agent_hierarchy": self.agent_hierarchy.to_dict(),
            "active_agents": self.agent_hierarchy.get_active_agents(),
            "system_health": await self._get_system_health(),
            "mcp_status": await self.mcp_manager.get_status(),
            "resource_usage": await self._get_resource_usage()
        }
        
    async def _generate_forecast(self, context: IterationContext, decision):
        """Generate future state forecast"""
        forecast = self.forecaster(
            current_state=json.dumps(context.system_state, default=str),
            planned_actions=decision.priority_tasks,
            historical_outcomes=json.dumps(self.state_holder.get_historical_outcomes(), default=str)
        )
        
        context.boss_decision["detailed_forecast"] = {
            "future_states": forecast.future_states,
            "potential_risks": forecast.potential_risks,
            "success_probabilities": forecast.success_probabilities
        }
        
    def _parse_agent_assignments(self, assignments_str: str) -> List[Dict[str, Any]]:
        """Parse agent assignments from DSPY output"""
        try:
            # Simple parsing - in production this would be more sophisticated
            assignments = []
            for line in assignments_str.split('\n'):
                if 'Agent' in line and ':' in line:
                    parts = line.split(':')
                    agent_part = parts[0].strip()
                    task_part = parts[1].strip() if len(parts) > 1 else ""
                    assignments.append({
                        "agent": agent_part,
                        "task": task_part
                    })
            return assignments
        except Exception as e:
            logger.error(f"Error parsing agent assignments: {e}")
            return []
            
    def _parse_priority_tasks(self, tasks_str: str) -> List[str]:
        """Parse priority tasks from DSPY output"""
        try:
            tasks = []
            for line in tasks_str.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove numbering if present
                    if '. ' in line:
                        line = line.split('. ', 1)[1]
                    tasks.append(line)
            return tasks
        except Exception as e:
            logger.error(f"Error parsing priority tasks: {e}")
            return []

    async def _execute_agent_assignment(self, assignment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent assignment"""
        agent_name = assignment["agent"]
        task = assignment["task"]
        
        # Get or create agent
        agent = await self.agent_hierarchy.get_or_create_agent(agent_name)
        
        # Execute task
        result = await agent.execute_task(task)
        
        return {
            "agent": agent_name,
            "task": task,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _store_iteration_result(self, context: IterationContext):
        """Store complete iteration result"""
        iteration_result = IterationResult(
            iteration_id=context.iteration_id,
            timestamp=context.timestamp,
            pre_processing=context.pre_processing_result,
            boss_decision=context.boss_decision,
            execution=context.execution_result,
            next_prep=context.next_prep_result,
            error_info=context.error_info
        )
        
        await self.state_holder.store_iteration_result(iteration_result)
        logger.info(f"💾 Stored iteration {context.iteration_id} results")

    # Additional helper methods for data gathering and analysis
    async def _gather_raw_system_data(self) -> Dict[str, Any]:
        """Gather raw system data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": await self._get_system_metrics(),
            "mcp_data": await self.mcp_manager.get_all_data(),
            "agent_statuses": self.agent_hierarchy.get_all_statuses()
        }
        
    async def _get_previous_iteration_results(self) -> Dict[str, Any]:
        """Get results from previous iterations"""
        return self.state_holder.get_recent_iteration_results(5)  # Last 5 iterations
        
    async def _get_comprehensive_state(self) -> Dict[str, Any]:
        """Get comprehensive current state"""
        return {
            **await self._gather_system_state(),
            "pre_processing_insights": self.current_context.pre_processing_result if self.current_context else None
        }
        
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        return {
            "status": "healthy",  # Implement real health checks
            "uptime": datetime.now().isoformat(),
            "memory_usage": "normal",
            "cpu_usage": "normal"
        }
        
    async def _get_resource_usage(self) -> Dict[str, Any]:
        """Get resource usage metrics"""
        return {
            "memory": "512MB",  # Implement real resource monitoring
            "cpu": "25%",
            "disk": "10GB"
        }
        
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics"""
        return {
            "iterations_completed": self.iteration_count,
            "autonomous_uptime": datetime.now().isoformat(),
            "total_decisions": self.state_holder.get_total_decisions(),
            "success_rate": self.state_holder.get_success_rate()
        }
        
    async def _analyze_iteration_performance(self, context: IterationContext) -> Dict[str, Any]:
        """Analyze performance of completed iteration"""
        return {
            "duration": "calculated_duration",
            "success": context.error_info is None,
            "tasks_completed": context.execution_result.get("tasks_completed", 0) if context.execution_result else 0,
            "efficiency_score": 0.85  # Calculate real efficiency
        }
        
    async def _update_system_learnings(self, analysis: Dict[str, Any]):
        """Update system with learnings from iteration"""
        await self.state_holder.store_learning(analysis)
        
    async def _prepare_next_iteration_context(self) -> Dict[str, Any]:
        """Prepare context for next iteration"""
        return {
            "predicted_workload": "normal",
            "recommended_agents": self.agent_hierarchy.get_recommended_agents(),
            "priority_focus": "continue current tasks"
        }
