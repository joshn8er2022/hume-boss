
"""
Core iteration engine implementing the autonomous execution lifecycle
Handles: pre-processing → decision → try/except → finally → next pre-processing
"""

import asyncio
import traceback
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger

import dspy

from ..state.state_holder import StateHolder, AutonomousState
from ..state.historical_manager import HistoricalStateManager
from ..state.forecasting_engine import StateForecaster
from ..signatures.decision_engine import (
    AutonomousDecisionSignature, IterationPreprocessingSignature, NextIterationSignature,
    SystemContext, PreprocessingContext, NextIterationContext, DecisionOutput
)


class IterationPhase(Enum):
    """Phases in the iteration lifecycle"""
    PREPROCESSING = "preprocessing"
    DECISION_MAKING = "decision_making"  
    EXECUTION = "execution"
    ERROR_HANDLING = "error_handling"
    FINALIZATION = "finalization"
    NEXT_ITERATION_PREP = "next_iteration_prep"


class IterationContext(BaseModel):
    """Context for a single iteration"""
    
    iteration_id: str = Field(..., description="Unique iteration identifier")
    iteration_number: int = Field(..., description="Sequential iteration number")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    current_phase: IterationPhase = Field(default=IterationPhase.PREPROCESSING)
    
    # State references
    base_state: Optional[Dict[str, Any]] = None
    preprocessed_context: Optional[Dict[str, Any]] = None
    decision_made: Optional[DecisionOutput] = None
    
    # Execution tracking
    actions_taken: List[str] = Field(default_factory=list)
    execution_results: List[Dict[str, Any]] = Field(default_factory=list)
    errors_encountered: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Performance metrics
    phase_durations: Dict[str, float] = Field(default_factory=dict)
    total_duration: Optional[float] = None
    success: Optional[bool] = None
    
    # Next iteration preparation
    next_iteration_plan: Optional[Dict[str, Any]] = None
    continuation_decision: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class IterationResult(BaseModel):
    """Result of a completed iteration"""
    
    context: IterationContext
    final_state: AutonomousState
    success: bool
    error_message: Optional[str] = None
    lessons_learned: List[str] = Field(default_factory=list)
    next_action_recommendations: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


class IterationEngine:
    """
    Core iteration engine that executes the autonomous lifecycle
    This is the main engine that makes the system truly autonomous
    """
    
    def __init__(
        self,
        state_holder: StateHolder,
        historical_manager: HistoricalStateManager,
        forecaster: StateForecaster,
        llm_config: Dict[str, Any] = None
    ):
        self.state_holder = state_holder
        self.historical_manager = historical_manager
        self.forecaster = forecaster
        
        # DSPY signatures
        self.preprocessing_signature = IterationPreprocessingSignature()
        self.decision_signature = AutonomousDecisionSignature()
        self.next_iteration_signature = NextIterationSignature()
        
        # LLM configuration
        self.llm_config = llm_config or {"model": "gpt-4", "temperature": 0.7}
        
        # Execution control
        self.is_running = False
        self.max_execution_time = 300  # 5 minutes max per iteration
        self.max_consecutive_errors = 3
        self.consecutive_error_count = 0
        
        # Callbacks and hooks
        self.phase_callbacks: Dict[IterationPhase, List[Callable]] = {
            phase: [] for phase in IterationPhase
        }
        self.error_handlers: List[Callable] = []
        
        # Metrics tracking
        self.iteration_metrics: Dict[str, List[float]] = {
            "duration": [],
            "success_rate": [],
            "decision_confidence": [],
            "preprocessing_time": [],
            "execution_time": []
        }
        
        logger.info("IterationEngine initialized")
    
    async def execute_single_iteration(self, force_new_iteration: bool = True) -> IterationResult:
        """
        Execute a single complete iteration of the autonomous lifecycle
        
        Args:
            force_new_iteration: Whether to force start a new iteration
        
        Returns:
            IterationResult with complete execution details
        """
        start_time = time.time()
        current_state = await self.state_holder.get_current_state()
        
        if not current_state:
            logger.error("No current state available for iteration")
            raise ValueError("Current state required for iteration")
        
        # Start new iteration if requested
        if force_new_iteration:
            current_state = await self.state_holder.new_iteration("preprocessing")
        
        # Create iteration context
        iteration_context = IterationContext(
            iteration_id=f"iter_{current_state.iteration_number}_{int(start_time)}",
            iteration_number=current_state.iteration_number,
            base_state=current_state.dict()
        )
        
        logger.info(f"Starting iteration {iteration_context.iteration_number}")
        
        try:
            # Phase 1: Preprocessing
            await self._execute_preprocessing_phase(iteration_context, current_state)
            
            # Phase 2: Decision Making
            await self._execute_decision_phase(iteration_context, current_state)
            
            # Phase 3: Execution (with try/except)
            await self._execute_execution_phase(iteration_context, current_state)
            
        except Exception as e:
            # Phase 4: Error Handling
            await self._execute_error_handling_phase(iteration_context, current_state, e)
        
        finally:
            # Phase 5: Finalization
            await self._execute_finalization_phase(iteration_context, current_state)
            
            # Phase 6: Next Iteration Preparation
            await self._execute_next_iteration_prep_phase(iteration_context, current_state)
        
        # Complete iteration
        iteration_context.total_duration = time.time() - start_time
        iteration_context.success = len(iteration_context.errors_encountered) == 0
        
        # Update state with iteration results
        final_state = await self._finalize_iteration_state(iteration_context, current_state)
        
        # Create result
        result = IterationResult(
            context=iteration_context,
            final_state=final_state,
            success=iteration_context.success,
            error_message=iteration_context.errors_encountered[0] if iteration_context.errors_encountered else None,
            lessons_learned=self._extract_lessons_learned(iteration_context),
            next_action_recommendations=self._generate_next_action_recommendations(iteration_context),
            performance_metrics=self._calculate_performance_metrics(iteration_context)
        )
        
        # Update metrics
        await self._update_iteration_metrics(result)
        
        # Store historical state
        await self.historical_manager.store_state(final_state)
        
        logger.info(f"Completed iteration {iteration_context.iteration_number} in {iteration_context.total_duration:.2f}s")
        return result
    
    async def _execute_preprocessing_phase(self, context: IterationContext, state: AutonomousState):
        """Execute preprocessing phase"""
        phase_start = time.time()
        context.current_phase = IterationPhase.PREPROCESSING
        
        logger.debug("Executing preprocessing phase")
        
        try:
            # Gather context for preprocessing
            recent_states = await self.state_holder.get_recent_states(10)
            historical_patterns = await self.historical_manager.analyze_patterns(lookback_days=3)
            
            # Create preprocessing context
            preprocessing_context = PreprocessingContext(
                previous_iteration_results=recent_states[-1].dict() if recent_states else None,
                system_state_changes=self._detect_state_changes(recent_states),
                new_data_available=self._detect_new_data(state),
                external_triggers=self._detect_external_triggers(state),
                error_reports=[error for error in state.execution_errors[-5:]]  # Last 5 errors
            )
            
            # Use DSPY for preprocessing
            with dspy.context(lm=dspy.OpenAI(**self.llm_config)):
                preprocessing_result = self.preprocessing_signature(
                    preprocessing_context=preprocessing_context,
                    system_state=state.dict(),
                    retrieval_results=self._get_retrieval_context(state, historical_patterns)
                )
            
            # Store preprocessing results
            context.preprocessed_context = preprocessing_result.preprocessing_output.dict()
            context.actions_taken.append("preprocessing_completed")
            
            # Run phase callbacks
            await self._run_phase_callbacks(IterationPhase.PREPROCESSING, context, state)
            
        except Exception as e:
            context.errors_encountered.append(f"Preprocessing error: {str(e)}")
            logger.error(f"Preprocessing phase error: {e}")
        
        finally:
            duration = time.time() - phase_start
            context.phase_durations["preprocessing"] = duration
            logger.debug(f"Preprocessing phase completed in {duration:.2f}s")
    
    async def _execute_decision_phase(self, context: IterationContext, state: AutonomousState):
        """Execute decision making phase"""
        phase_start = time.time()
        context.current_phase = IterationPhase.DECISION_MAKING
        
        logger.debug("Executing decision making phase")
        
        try:
            # Prepare system context for decision making
            recent_states = await self.state_holder.get_recent_states(5)
            historical_patterns = await self.historical_manager.analyze_patterns(lookback_days=2)
            
            system_context = SystemContext(
                current_state=state.boss_state,
                historical_states=[s.dict() for s in recent_states],
                active_agents=[agent for agent in state.agents.values()],
                pending_tasks=[task for task in state.tasks.values() if task.get("status") == "pending"],
                system_metrics=state.metrics,
                mcp_servers_status=state.mcp_servers,
                recent_errors=state.execution_errors[-3:],
                last_iteration_outcome=context.preprocessed_context.get("readiness_status") if context.preprocessed_context else None
            )
            
            # Prepare historical analysis
            historical_analysis = self._prepare_historical_analysis(historical_patterns)
            
            # Current objectives
            current_objectives = self._determine_current_objectives(state, context.preprocessed_context)
            
            # Use DSPY for autonomous decision making
            with dspy.context(lm=dspy.OpenAI(**self.llm_config)):
                decision_result = self.decision_signature(
                    context=system_context,
                    historical_pattern_analysis=historical_analysis,
                    current_objectives=current_objectives
                )
            
            # Store decision
            context.decision_made = decision_result.decision
            context.actions_taken.append(f"decision_made: {decision_result.decision.decision_type}")
            
            # Update state with decision
            await self.state_holder.update_state({
                "last_decision": decision_result.decision.dict(),
                "decision_reasoning": decision_result.decision.reasoning,
                "confidence_score": decision_result.confidence_score,
                "predicted_next_state": decision_result.next_state_prediction
            })
            
            # Run phase callbacks
            await self._run_phase_callbacks(IterationPhase.DECISION_MAKING, context, state)
            
        except Exception as e:
            context.errors_encountered.append(f"Decision making error: {str(e)}")
            logger.error(f"Decision making phase error: {e}")
        
        finally:
            duration = time.time() - phase_start
            context.phase_durations["decision_making"] = duration
            logger.debug(f"Decision making phase completed in {duration:.2f}s")
    
    async def _execute_execution_phase(self, context: IterationContext, state: AutonomousState):
        """Execute the decision with try/except handling"""
        phase_start = time.time()
        context.current_phase = IterationPhase.EXECUTION
        
        logger.debug("Executing decision implementation phase")
        
        if not context.decision_made:
            context.errors_encountered.append("No decision available for execution")
            return
        
        try:
            decision = context.decision_made
            execution_results = []
            
            # Execute each action in the action plan
            for i, action in enumerate(decision.action_plan):
                action_start = time.time()
                
                try:
                    logger.debug(f"Executing action {i+1}/{len(decision.action_plan)}: {action}")
                    
                    # Execute the action (this would interface with actual system components)
                    action_result = await self._execute_action(action, state, context)
                    
                    execution_results.append({
                        "action": action,
                        "result": action_result,
                        "success": True,
                        "duration": time.time() - action_start
                    })
                    
                    context.actions_taken.append(f"executed: {action}")
                    
                except Exception as action_error:
                    execution_results.append({
                        "action": action,
                        "error": str(action_error),
                        "success": False,
                        "duration": time.time() - action_start
                    })
                    
                    context.errors_encountered.append(f"Action execution error: {action} - {str(action_error)}")
                    logger.warning(f"Action execution failed: {action} - {action_error}")
                    
                    # Decide whether to continue or abort based on decision configuration
                    if not self._should_continue_after_error(decision, action_error, i):
                        break
            
            # Store execution results
            context.execution_results = execution_results
            
            # Update state with execution results
            successful_actions = [r for r in execution_results if r["success"]]
            failed_actions = [r for r in execution_results if not r["success"]]
            
            await self.state_holder.update_state({
                "execution_success": len(failed_actions) == 0,
                "execution_errors": [r["error"] for r in failed_actions],
                "success_indicators": {
                    "actions_completed": len(successful_actions),
                    "actions_failed": len(failed_actions),
                    "execution_success_rate": len(successful_actions) / len(execution_results) if execution_results else 0
                }
            })
            
            # Run phase callbacks
            await self._run_phase_callbacks(IterationPhase.EXECUTION, context, state)
            
        except Exception as e:
            context.errors_encountered.append(f"Execution phase error: {str(e)}")
            logger.error(f"Execution phase error: {e}")
            
            # Update state with failure
            await self.state_holder.update_state({
                "execution_success": False,
                "execution_errors": [str(e)]
            })
        
        finally:
            duration = time.time() - phase_start
            context.phase_durations["execution"] = duration
            logger.debug(f"Execution phase completed in {duration:.2f}s")
    
    async def _execute_error_handling_phase(self, context: IterationContext, state: AutonomousState, error: Exception):
        """Handle errors that occurred during execution"""
        phase_start = time.time()
        context.current_phase = IterationPhase.ERROR_HANDLING
        
        logger.debug("Executing error handling phase")
        
        try:
            # Log the error
            error_details = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "iteration_number": context.iteration_number,
                "phase": context.current_phase.value
            }
            
            context.errors_encountered.append(f"Fatal error: {str(error)}")
            
            # Update consecutive error count
            self.consecutive_error_count += 1
            
            # Run error handlers
            for handler in self.error_handlers:
                try:
                    await handler(error, context, state)
                except Exception as handler_error:
                    logger.warning(f"Error handler failed: {handler_error}")
            
            # Update state with error information
            await self.state_holder.update_state({
                "execution_success": False,
                "execution_errors": context.errors_encountered,
                "error_count": state.error_count + 1
            })
            
            # Determine recovery actions
            recovery_actions = self._determine_recovery_actions(error, context, state)
            if recovery_actions:
                context.actions_taken.extend(recovery_actions)
            
        except Exception as handling_error:
            logger.error(f"Error during error handling: {handling_error}")
            context.errors_encountered.append(f"Error handling failed: {str(handling_error)}")
        
        finally:
            duration = time.time() - phase_start
            context.phase_durations["error_handling"] = duration
            logger.debug(f"Error handling phase completed in {duration:.2f}s")
    
    async def _execute_finalization_phase(self, context: IterationContext, state: AutonomousState):
        """Execute finalization phase"""
        phase_start = time.time()
        context.current_phase = IterationPhase.FINALIZATION
        
        logger.debug("Executing finalization phase")
        
        try:
            # Analyze iteration performance
            iteration_success = len(context.errors_encountered) == 0
            
            # Update success tracking
            if iteration_success:
                self.consecutive_error_count = 0
            
            # Calculate performance metrics
            performance_metrics = {
                "iteration_duration": sum(context.phase_durations.values()),
                "success": iteration_success,
                "actions_completed": len([a for a in context.actions_taken if a.startswith("executed:")]),
                "errors_count": len(context.errors_encountered),
                "confidence_score": context.decision_made.expected_outcome if context.decision_made else None
            }
            
            # Update state with finalization
            await self.state_holder.update_state({
                "system_phase": "finalization",
                "metrics": {**state.metrics, **performance_metrics},
                "execution_duration": sum(context.phase_durations.values())
            })
            
            # Store lessons learned
            lessons = self._extract_lessons_learned(context)
            if lessons:
                await self.state_holder.update_state({
                    "insights_learned": state.insights_learned + lessons
                })
            
            # Run phase callbacks
            await self._run_phase_callbacks(IterationPhase.FINALIZATION, context, state)
            
        except Exception as e:
            context.errors_encountered.append(f"Finalization error: {str(e)}")
            logger.error(f"Finalization phase error: {e}")
        
        finally:
            duration = time.time() - phase_start
            context.phase_durations["finalization"] = duration
            logger.debug(f"Finalization phase completed in {duration:.2f}s")
    
    async def _execute_next_iteration_prep_phase(self, context: IterationContext, state: AutonomousState):
        """Execute next iteration preparation phase"""
        phase_start = time.time()
        context.current_phase = IterationPhase.NEXT_ITERATION_PREP
        
        logger.debug("Executing next iteration preparation phase")
        
        try:
            # Prepare context for next iteration planning
            next_iteration_context = NextIterationContext(
                current_iteration_results={
                    "success": context.success,
                    "actions_taken": context.actions_taken,
                    "execution_results": context.execution_results,
                    "errors": context.errors_encountered
                },
                execution_success=context.success or False,
                error_details=context.errors_encountered if context.errors_encountered else None,
                performance_metrics=context.phase_durations,
                time_elapsed=sum(context.phase_durations.values())
            )
            
            # Gather system feedback
            system_feedback = self._gather_system_feedback(state, context)
            
            # Use DSPY for next iteration planning
            with dspy.context(lm=dspy.OpenAI(**self.llm_config)):
                next_iteration_result = self.next_iteration_signature(
                    iteration_context=next_iteration_context,
                    system_feedback=system_feedback
                )
            
            # Store next iteration plan
            context.next_iteration_plan = next_iteration_result.next_iteration_output.dict()
            context.continuation_decision = next_iteration_result.continuation_decision
            
            # Update state with next iteration preparation
            await self.state_holder.update_state({
                "system_phase": "next_iteration_prep",
                "planned_actions": next_iteration_result.next_iteration_output.next_iteration_plan.get("actions", []),
                "improvement_actions": next_iteration_result.next_iteration_output.optimization_suggestions
            })
            
            # Run phase callbacks
            await self._run_phase_callbacks(IterationPhase.NEXT_ITERATION_PREP, context, state)
            
        except Exception as e:
            context.errors_encountered.append(f"Next iteration prep error: {str(e)}")
            logger.error(f"Next iteration prep phase error: {e}")
        
        finally:
            duration = time.time() - phase_start
            context.phase_durations["next_iteration_prep"] = duration
            logger.debug(f"Next iteration prep phase completed in {duration:.2f}s")
    
    # Helper methods
    
    def _detect_state_changes(self, recent_states: List[AutonomousState]) -> List[str]:
        """Detect significant state changes"""
        if len(recent_states) < 2:
            return []
        
        changes = []
        current = recent_states[-1]
        previous = recent_states[-2]
        
        if current.boss_state != previous.boss_state:
            changes.append(f"boss_state: {previous.boss_state} → {current.boss_state}")
        
        if len(current.active_agents) != len(previous.active_agents):
            changes.append(f"active_agents: {len(previous.active_agents)} → {len(current.active_agents)}")
        
        if len(current.active_tasks) != len(previous.active_tasks):
            changes.append(f"active_tasks: {len(previous.active_tasks)} → {len(current.active_tasks)}")
        
        return changes
    
    def _detect_new_data(self, state: AutonomousState) -> List[str]:
        """Detect new data available to the system"""
        new_data = []
        
        # Check for new MCP server data
        for server_id, server_data in state.mcp_servers.items():
            if server_data.get("has_new_data", False):
                new_data.append(f"mcp_server_{server_id}_data")
        
        # Check for new external data
        if state.external_data:
            new_data.append("external_data_updated")
        
        return new_data
    
    def _detect_external_triggers(self, state: AutonomousState) -> List[str]:
        """Detect external triggers that might influence decisions"""
        triggers = []
        
        # Time-based triggers
        current_hour = datetime.utcnow().hour
        if current_hour in [9, 12, 18]:  # Business hours transitions
            triggers.append(f"business_hour_transition_{current_hour}")
        
        # Error-based triggers
        if state.error_count > 0:
            triggers.append("errors_present")
        
        # Load-based triggers
        if len(state.active_tasks) > 10:
            triggers.append("high_task_load")
        
        return triggers
    
    def _get_retrieval_context(self, state: AutonomousState, patterns: List) -> List[str]:
        """Get retrieval context for DSPY retriever"""
        context = []
        
        # Add current state summary
        context.append(f"Current state: {state.boss_state} with {len(state.active_agents)} agents")
        
        # Add recent patterns
        for pattern in patterns[:3]:  # Top 3 patterns
            context.append(f"Pattern: {pattern.description}")
        
        # Add recent performance
        success_rate = state.success_indicators.get("success_rate", 0)
        context.append(f"Recent success rate: {success_rate:.1%}")
        
        return context
    
    def _prepare_historical_analysis(self, patterns: List) -> str:
        """Prepare historical analysis summary"""
        if not patterns:
            return "No significant historical patterns identified."
        
        analysis = f"Identified {len(patterns)} historical patterns:\n"
        for pattern in patterns[:5]:  # Top 5 patterns
            analysis += f"- {pattern.description} (confidence: {pattern.confidence:.1%})\n"
        
        return analysis
    
    def _determine_current_objectives(self, state: AutonomousState, preprocessed_context: Optional[Dict]) -> List[str]:
        """Determine current system objectives"""
        objectives = []
        
        # Performance objectives
        if state.success_indicators.get("success_rate", 1.0) < 0.8:
            objectives.append("Improve success rate above 80%")
        
        # Error reduction objectives
        if state.error_count > 2:
            objectives.append("Reduce error count")
        
        # Efficiency objectives
        if len(state.active_tasks) > 20:
            objectives.append("Optimize task processing efficiency")
        
        # Agent utilization objectives
        if len(state.active_agents) < 2:
            objectives.append("Optimize agent utilization")
        
        # Default objective
        if not objectives:
            objectives.append("Maintain optimal system performance")
        
        return objectives
    
    async def _execute_action(self, action: str, state: AutonomousState, context: IterationContext) -> Dict[str, Any]:
        """Execute a specific action (placeholder for actual implementation)"""
        # This would interface with actual system components
        # For now, return a simulated result
        
        action_type = action.split(":")[0].lower() if ":" in action else action.lower()
        
        if "agent" in action_type:
            return {"type": "agent_action", "status": "simulated", "details": action}
        elif "task" in action_type:
            return {"type": "task_action", "status": "simulated", "details": action}
        elif "mcp" in action_type:
            return {"type": "mcp_action", "status": "simulated", "details": action}
        else:
            return {"type": "general_action", "status": "simulated", "details": action}
    
    def _should_continue_after_error(self, decision: DecisionOutput, error: Exception, action_index: int) -> bool:
        """Determine if execution should continue after an error"""
        
        # Check decision configuration
        if hasattr(decision, 'continue_on_error') and not decision.continue_on_error:
            return False
        
        # Critical errors should stop execution
        if isinstance(error, (SystemExit, KeyboardInterrupt, MemoryError)):
            return False
        
        # If we're past halfway through actions, continue
        if action_index > len(decision.action_plan) / 2:
            return True
        
        # Default: continue unless it's a severe error
        return True
    
    def _determine_recovery_actions(self, error: Exception, context: IterationContext, state: AutonomousState) -> List[str]:
        """Determine recovery actions based on error type"""
        
        recovery_actions = []
        
        if "connection" in str(error).lower():
            recovery_actions.append("retry_connections")
        
        if "timeout" in str(error).lower():
            recovery_actions.append("adjust_timeouts")
        
        if "memory" in str(error).lower():
            recovery_actions.append("cleanup_memory")
        
        # Generic recovery
        recovery_actions.append("log_error_details")
        recovery_actions.append("prepare_graceful_continuation")
        
        return recovery_actions
    
    def _gather_system_feedback(self, state: AutonomousState, context: IterationContext) -> List[str]:
        """Gather feedback from system components"""
        
        feedback = []
        
        # Performance feedback
        if sum(context.phase_durations.values()) > 60:  # Over 1 minute
            feedback.append("Iteration took longer than expected")
        
        # Success feedback
        if not context.errors_encountered:
            feedback.append("Iteration completed successfully")
        else:
            feedback.append(f"Iteration had {len(context.errors_encountered)} errors")
        
        # Agent feedback
        if len(state.active_agents) == 0:
            feedback.append("No agents currently active")
        
        # Task feedback
        if len(state.active_tasks) == 0:
            feedback.append("No tasks currently active")
        
        return feedback
    
    async def _finalize_iteration_state(self, context: IterationContext, state: AutonomousState) -> AutonomousState:
        """Finalize the iteration state"""
        
        # Update state with final iteration results
        final_updates = {
            "system_phase": "completed",
            "execution_duration": context.total_duration,
            "execution_success": context.success,
            "last_iteration_completed": datetime.utcnow().isoformat()
        }
        
        return await self.state_holder.update_state(final_updates)
    
    def _extract_lessons_learned(self, context: IterationContext) -> List[str]:
        """Extract lessons learned from the iteration"""
        
        lessons = []
        
        # Performance lessons
        if context.total_duration and context.total_duration > 120:
            lessons.append("Long iteration duration - consider optimization")
        
        # Error lessons
        if context.errors_encountered:
            lessons.append(f"Common error pattern: {context.errors_encountered[0]}")
        
        # Success lessons
        if context.success and context.decision_made:
            lessons.append(f"Successful strategy: {context.decision_made.decision_type}")
        
        return lessons
    
    def _generate_next_action_recommendations(self, context: IterationContext) -> List[str]:
        """Generate recommendations for next actions"""
        
        recommendations = []
        
        # Error-based recommendations
        if context.errors_encountered:
            recommendations.append("Focus on error prevention and recovery")
        
        # Performance-based recommendations
        if context.total_duration and context.total_duration > 90:
            recommendations.append("Optimize iteration execution time")
        
        # Success-based recommendations
        if context.success:
            recommendations.append("Continue with current strategy")
        
        return recommendations
    
    def _calculate_performance_metrics(self, context: IterationContext) -> Dict[str, Any]:
        """Calculate performance metrics for the iteration"""
        
        return {
            "total_duration": context.total_duration,
            "phase_durations": context.phase_durations,
            "success": context.success,
            "error_count": len(context.errors_encountered),
            "actions_completed": len(context.actions_taken),
            "execution_efficiency": len(context.actions_taken) / context.total_duration if context.total_duration else 0
        }
    
    async def _update_iteration_metrics(self, result: IterationResult):
        """Update overall iteration metrics"""
        
        self.iteration_metrics["duration"].append(result.context.total_duration or 0)
        self.iteration_metrics["success_rate"].append(1.0 if result.success else 0.0)
        
        if result.context.decision_made:
            confidence = getattr(result.context.decision_made, 'confidence_score', 0.5)
            self.iteration_metrics["decision_confidence"].append(confidence)
        
        # Keep only last 100 measurements
        for metric_list in self.iteration_metrics.values():
            if len(metric_list) > 100:
                metric_list.pop(0)
    
    async def _run_phase_callbacks(self, phase: IterationPhase, context: IterationContext, state: AutonomousState):
        """Run callbacks for a specific phase"""
        
        for callback in self.phase_callbacks[phase]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context, state)
                else:
                    callback(context, state)
            except Exception as e:
                logger.warning(f"Phase callback failed for {phase.value}: {e}")
    
    # Public interface methods
    
    def add_phase_callback(self, phase: IterationPhase, callback: Callable):
        """Add a callback for a specific phase"""
        self.phase_callbacks[phase].append(callback)
    
    def add_error_handler(self, handler: Callable):
        """Add an error handler"""
        self.error_handlers.append(handler)
    
    def get_iteration_metrics(self) -> Dict[str, Any]:
        """Get current iteration metrics"""
        
        metrics = {}
        for name, values in self.iteration_metrics.items():
            if values:
                metrics[name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "latest": values[-1],
                    "min": min(values),
                    "max": max(values)
                }
            else:
                metrics[name] = {"count": 0}
        
        return metrics
    
    async def should_continue_autonomous_execution(self) -> bool:
        """Determine if autonomous execution should continue"""
        
        # Check consecutive error limit
        if self.consecutive_error_count >= self.max_consecutive_errors:
            logger.warning(f"Stopping autonomous execution due to {self.consecutive_error_count} consecutive errors")
            return False
        
        # Check system health
        health_summary = await self.state_holder.get_system_health_summary()
        if health_summary.get("status") == "critical":
            logger.warning("Stopping autonomous execution due to critical system health")
            return False
        
        return True
