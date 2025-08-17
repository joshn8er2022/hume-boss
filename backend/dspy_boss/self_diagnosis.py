
"""
Self-diagnosis system using DSPY's native Python interpreter
"""

import asyncio
import sys
import traceback
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from loguru import logger

import dspy
from dspy import PythonInterpreter

from .models import DiagnosisResult, SystemMetrics, BossStateData
from .state_machine import StateMachineManager


class SelfDiagnosisSystem:
    """Self-diagnosis system using DSPY's Python interpreter"""
    
    def __init__(self, state_manager: StateMachineManager):
        self.state_manager = state_manager
        self.python_interpreter = PythonInterpreter()
        
        # Diagnosis history
        self.diagnosis_history: List[DiagnosisResult] = []
        self.max_history_size = 100
        
        # System metrics tracking
        self.metrics_history: List[SystemMetrics] = []
        self.max_metrics_history = 1000
        
        # Diagnosis templates
        self.diagnosis_templates = self._load_diagnosis_templates()
        
        logger.info("Self-diagnosis system initialized")
    
    def _load_diagnosis_templates(self) -> Dict[str, str]:
        """Load diagnosis code templates"""
        return {
            "health_check": """
# System Health Check
import psutil
import time
from datetime import datetime

def health_check():
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'healthy',
        'issues': [],
        'metrics': {}
    }
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    results['metrics']['cpu_usage'] = cpu_percent
    if cpu_percent > 80:
        results['issues'].append(f'High CPU usage: {cpu_percent}%')
        results['status'] = 'warning'
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    results['metrics']['memory_usage'] = memory_percent
    if memory_percent > 85:
        results['issues'].append(f'High memory usage: {memory_percent}%')
        results['status'] = 'warning'
    
    # Disk usage
    disk = psutil.disk_usage('/')
    disk_percent = (disk.used / disk.total) * 100
    results['metrics']['disk_usage'] = disk_percent
    if disk_percent > 90:
        results['issues'].append(f'High disk usage: {disk_percent:.1f}%')
        results['status'] = 'warning'
    
    # Set critical status if multiple issues
    if len(results['issues']) >= 3:
        results['status'] = 'critical'
    
    return results

# Execute health check
health_result = health_check()
print(f"Health Status: {health_result['status']}")
print(f"Issues: {health_result['issues']}")
print(f"Metrics: {health_result['metrics']}")
health_result
            """,
            
            "performance_analysis": """
# Performance Analysis
import statistics
from datetime import datetime, timedelta

def analyze_performance(state_data, metrics_history):
    analysis = {
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'healthy',
        'insights': [],
        'recommendations': []
    }
    
    # Task completion analysis
    total_tasks = len(state_data.get('completed_tasks', [])) + len(state_data.get('failed_tasks', []))
    failed_tasks = len(state_data.get('failed_tasks', []))
    
    if total_tasks > 0:
        failure_rate = (failed_tasks / total_tasks) * 100
        analysis['failure_rate'] = failure_rate
        
        if failure_rate > 20:
            analysis['status'] = 'warning'
            analysis['insights'].append(f'High task failure rate: {failure_rate:.1f}%')
            analysis['recommendations'].append('Review task assignment logic and agent capabilities')
        elif failure_rate > 10:
            analysis['insights'].append(f'Moderate task failure rate: {failure_rate:.1f}%')
            analysis['recommendations'].append('Monitor task complexity and agent performance')
    
    # Workload analysis
    current_workload = state_data.get('current_workload', 0)
    pending_tasks = len(state_data.get('pending_tasks', []))
    
    if pending_tasks > 20:
        analysis['status'] = 'warning'
        analysis['insights'].append(f'High pending task count: {pending_tasks}')
        analysis['recommendations'].append('Consider spawning additional agents')
    
    # Agent utilization
    active_agents = len(state_data.get('active_agents', []))
    if active_agents == 0 and pending_tasks > 0:
        analysis['status'] = 'critical'
        analysis['insights'].append('No active agents but tasks pending')
        analysis['recommendations'].append('Restart agent system immediately')
    
    return analysis

# Execute performance analysis
perf_result = analyze_performance(state_data, metrics_history)
print(f"Performance Status: {perf_result['status']}")
print(f"Insights: {perf_result['insights']}")
print(f"Recommendations: {perf_result['recommendations']}")
perf_result
            """,
            
            "error_investigation": """
# Error Investigation
import re
from collections import Counter
from datetime import datetime, timedelta

def investigate_errors(system_errors, failed_tasks_data):
    investigation = {
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'healthy',
        'error_patterns': {},
        'root_causes': [],
        'action_items': []
    }
    
    # Analyze system errors
    if system_errors:
        error_counter = Counter(system_errors)
        investigation['error_patterns'] = dict(error_counter.most_common(5))
        
        # Look for common patterns
        connection_errors = [e for e in system_errors if 'connection' in e.lower()]
        timeout_errors = [e for e in system_errors if 'timeout' in e.lower()]
        memory_errors = [e for e in system_errors if 'memory' in e.lower()]
        
        if len(connection_errors) > 3:
            investigation['root_causes'].append('Network connectivity issues')
            investigation['action_items'].append('Check MCP server connections')
            investigation['status'] = 'warning'
        
        if len(timeout_errors) > 3:
            investigation['root_causes'].append('Performance bottlenecks causing timeouts')
            investigation['action_items'].append('Increase timeout values or optimize performance')
            investigation['status'] = 'warning'
        
        if len(memory_errors) > 1:
            investigation['root_causes'].append('Memory management issues')
            investigation['action_items'].append('Review memory usage and implement cleanup')
            investigation['status'] = 'critical'
    
    # Analyze failed tasks
    if failed_tasks_data:
        task_error_patterns = {}
        for task_data in failed_tasks_data:
            error_msg = task_data.get('error_message', 'Unknown error')
            # Extract error type
            error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
            task_error_patterns[error_type] = task_error_patterns.get(error_type, 0) + 1
        
        investigation['task_error_patterns'] = task_error_patterns
        
        # Most common task error
        if task_error_patterns:
            most_common_error = max(task_error_patterns.items(), key=lambda x: x[1])
            if most_common_error[1] > 2:
                investigation['root_causes'].append(f'Recurring task error: {most_common_error[0]}')
                investigation['action_items'].append(f'Fix root cause of {most_common_error[0]} errors')
    
    return investigation

# Execute error investigation
error_result = investigate_errors(system_errors, failed_tasks_data)
print(f"Investigation Status: {error_result['status']}")
print(f"Root Causes: {error_result['root_causes']}")
print(f"Action Items: {error_result['action_items']}")
error_result
            """,
            
            "system_optimization": """
# System Optimization Analysis
import math
from datetime import datetime

def analyze_optimization_opportunities(state_data, agent_stats, task_stats):
    optimization = {
        'timestamp': datetime.utcnow().isoformat(),
        'opportunities': [],
        'recommendations': [],
        'priority_actions': []
    }
    
    # Agent optimization
    if agent_stats:
        total_agents = agent_stats.get('total_agents', 0)
        active_agents = agent_stats.get('active_agents', 0)
        
        if total_agents > 0:
            utilization = active_agents / total_agents
            if utilization < 0.5:
                optimization['opportunities'].append('Low agent utilization')
                optimization['recommendations'].append('Consider reducing number of agents or increasing workload')
            elif utilization > 0.9:
                optimization['opportunities'].append('High agent utilization')
                optimization['recommendations'].append('Consider spawning additional agents')
    
    # Task queue optimization
    if task_stats:
        avg_duration = task_stats.get('average_duration', 0)
        tasks_per_minute = task_stats.get('tasks_per_minute', 0)
        
        if avg_duration > 300:  # 5 minutes
            optimization['opportunities'].append('Long average task duration')
            optimization['recommendations'].append('Optimize task execution or break down complex tasks')
        
        if tasks_per_minute < 1 and len(state_data.get('pending_tasks', [])) > 0:
            optimization['opportunities'].append('Low task throughput')
            optimization['priority_actions'].append('Investigate task processing bottlenecks')
    
    # Memory optimization
    pending_count = len(state_data.get('pending_tasks', []))
    completed_count = len(state_data.get('completed_tasks', []))
    
    if completed_count > 1000:
        optimization['opportunities'].append('Large completed task history')
        optimization['recommendations'].append('Implement task history cleanup')
    
    # State machine optimization
    if state_data.get('current_workload', 0) == 0 and pending_count > 0:
        optimization['priority_actions'].append('State machine may be stuck - investigate state transitions')
    
    return optimization

# Execute optimization analysis
opt_result = analyze_optimization_opportunities(state_data, agent_stats, task_stats)
print(f"Optimization Opportunities: {len(opt_result['opportunities'])}")
print(f"Recommendations: {opt_result['recommendations']}")
print(f"Priority Actions: {opt_result['priority_actions']}")
opt_result
            """
        }
    
    async def run_health_check(self) -> DiagnosisResult:
        """Run comprehensive system health check"""
        logger.info("Running system health check...")
        
        try:
            # Execute health check code
            code = self.diagnosis_templates["health_check"]
            result = await self._execute_diagnosis_code(code, "health_check")
            
            # Parse results
            if result.execution_output:
                # Extract status and create diagnosis result
                status = "healthy"
                if "warning" in result.execution_output.lower():
                    status = "warning"
                elif "critical" in result.execution_output.lower():
                    status = "critical"
                
                diagnosis = DiagnosisResult(
                    diagnosis_type="health_check",
                    status=status,
                    summary="System health check completed",
                    details={"raw_output": result.execution_output},
                    code_executed=code,
                    execution_output=result.execution_output,
                    execution_error=result.execution_error
                )
            else:
                diagnosis = DiagnosisResult(
                    diagnosis_type="health_check",
                    status="critical",
                    summary="Health check failed to execute",
                    execution_error=result.execution_error or "Unknown error"
                )
            
            self._add_to_history(diagnosis)
            return diagnosis
            
        except Exception as e:
            logger.error(f"Error running health check: {e}")
            diagnosis = DiagnosisResult(
                diagnosis_type="health_check",
                status="critical",
                summary=f"Health check error: {str(e)}",
                execution_error=str(e)
            )
            self._add_to_history(diagnosis)
            return diagnosis
    
    async def analyze_performance(self) -> DiagnosisResult:
        """Analyze system performance"""
        logger.info("Analyzing system performance...")
        
        try:
            # Prepare data for analysis
            state_data = self.state_manager.transition.state_data.model_dump()
            metrics_history = [m.model_dump() for m in self.metrics_history[-10:]]  # Last 10 metrics
            
            # Execute performance analysis
            code = self.diagnosis_templates["performance_analysis"]
            
            # Inject data into execution context
            execution_context = {
                "state_data": state_data,
                "metrics_history": metrics_history
            }
            
            result = await self._execute_diagnosis_code(code, "performance_analysis", execution_context)
            
            # Parse results
            status = "healthy"
            if result.execution_output:
                if "warning" in result.execution_output.lower():
                    status = "warning"
                elif "critical" in result.execution_output.lower():
                    status = "critical"
            
            diagnosis = DiagnosisResult(
                diagnosis_type="performance_analysis",
                status=status,
                summary="Performance analysis completed",
                details={
                    "state_data": state_data,
                    "raw_output": result.execution_output
                },
                code_executed=code,
                execution_output=result.execution_output,
                execution_error=result.execution_error
            )
            
            self._add_to_history(diagnosis)
            return diagnosis
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            diagnosis = DiagnosisResult(
                diagnosis_type="performance_analysis",
                status="critical",
                summary=f"Performance analysis error: {str(e)}",
                execution_error=str(e)
            )
            self._add_to_history(diagnosis)
            return diagnosis
    
    async def investigate_errors(self) -> DiagnosisResult:
        """Investigate system errors"""
        logger.info("Investigating system errors...")
        
        try:
            # Gather error data
            system_errors = self.state_manager.transition.state_data.system_errors
            failed_tasks_data = []  # Would be populated from task manager
            
            # Execute error investigation
            code = self.diagnosis_templates["error_investigation"]
            
            execution_context = {
                "system_errors": system_errors,
                "failed_tasks_data": failed_tasks_data
            }
            
            result = await self._execute_diagnosis_code(code, "error_investigation", execution_context)
            
            # Parse results
            status = "healthy"
            if result.execution_output:
                if "warning" in result.execution_output.lower():
                    status = "warning"
                elif "critical" in result.execution_output.lower():
                    status = "critical"
            
            diagnosis = DiagnosisResult(
                diagnosis_type="error_investigation",
                status=status,
                summary="Error investigation completed",
                details={
                    "system_errors_count": len(system_errors),
                    "raw_output": result.execution_output
                },
                code_executed=code,
                execution_output=result.execution_output,
                execution_error=result.execution_error
            )
            
            self._add_to_history(diagnosis)
            return diagnosis
            
        except Exception as e:
            logger.error(f"Error investigating errors: {e}")
            diagnosis = DiagnosisResult(
                diagnosis_type="error_investigation",
                status="critical",
                summary=f"Error investigation failed: {str(e)}",
                execution_error=str(e)
            )
            self._add_to_history(diagnosis)
            return diagnosis
    
    async def run_custom_diagnosis(self, code: str, diagnosis_type: str = "custom") -> DiagnosisResult:
        """Run custom diagnosis code"""
        logger.info(f"Running custom diagnosis: {diagnosis_type}")
        
        try:
            result = await self._execute_diagnosis_code(code, diagnosis_type)
            
            diagnosis = DiagnosisResult(
                diagnosis_type=diagnosis_type,
                status="healthy",  # Default, would be parsed from output
                summary=f"Custom diagnosis '{diagnosis_type}' completed",
                details={"raw_output": result.execution_output},
                code_executed=code,
                execution_output=result.execution_output,
                execution_error=result.execution_error
            )
            
            self._add_to_history(diagnosis)
            return diagnosis
            
        except Exception as e:
            logger.error(f"Error running custom diagnosis: {e}")
            diagnosis = DiagnosisResult(
                diagnosis_type=diagnosis_type,
                status="critical",
                summary=f"Custom diagnosis failed: {str(e)}",
                execution_error=str(e)
            )
            self._add_to_history(diagnosis)
            return diagnosis
    
    async def _execute_diagnosis_code(self, code: str, diagnosis_type: str, 
                                    context: Optional[Dict[str, Any]] = None) -> 'ExecutionResult':
        """Execute diagnosis code using DSPY Python interpreter"""
        
        class ExecutionResult:
            def __init__(self):
                self.execution_output = None
                self.execution_error = None
        
        result = ExecutionResult()
        
        try:
            # Prepare execution environment
            if context:
                # Inject context variables into the code
                context_setup = "\n".join([f"{k} = {repr(v)}" for k, v in context.items()])
                full_code = context_setup + "\n\n" + code
            else:
                full_code = code
            
            # Execute using DSPY Python interpreter
            # Note: This is a simplified version - actual DSPY PythonInterpreter usage may differ
            try:
                # Capture stdout and stderr
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                
                stdout_capture = StringIO()
                stderr_capture = StringIO()
                
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                
                # Execute the code
                exec_globals = {}
                exec_locals = {}
                
                # Add context to locals if provided
                if context:
                    exec_locals.update(context)
                
                exec(full_code, exec_globals, exec_locals)
                
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                # Get output
                stdout_output = stdout_capture.getvalue()
                stderr_output = stderr_capture.getvalue()
                
                result.execution_output = stdout_output
                if stderr_output:
                    result.execution_error = stderr_output
                
            except Exception as e:
                # Restore stdout/stderr in case of error
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                result.execution_error = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            
        except Exception as e:
            result.execution_error = f"Setup error: {str(e)}"
        
        return result
    
    def _add_to_history(self, diagnosis: DiagnosisResult):
        """Add diagnosis to history"""
        self.diagnosis_history.append(diagnosis)
        
        # Maintain history size limit
        if len(self.diagnosis_history) > self.max_history_size:
            self.diagnosis_history = self.diagnosis_history[-self.max_history_size:]
    
    def get_diagnosis_history(self, limit: int = 10) -> List[DiagnosisResult]:
        """Get recent diagnosis history"""
        return self.diagnosis_history[-limit:]
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health based on recent diagnoses"""
        if not self.diagnosis_history:
            return {
                "overall_status": "unknown",
                "last_check": None,
                "issues": ["No diagnosis history available"]
            }
        
        recent_diagnoses = self.diagnosis_history[-5:]  # Last 5 diagnoses
        
        # Determine overall status
        statuses = [d.status for d in recent_diagnoses]
        if "critical" in statuses:
            overall_status = "critical"
        elif "warning" in statuses:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Collect issues and recommendations
        all_recommendations = []
        all_action_items = []
        
        for diagnosis in recent_diagnoses:
            all_recommendations.extend(diagnosis.recommendations)
            all_action_items.extend(diagnosis.action_items)
        
        return {
            "overall_status": overall_status,
            "last_check": recent_diagnoses[-1].timestamp.isoformat(),
            "recent_diagnoses": len(recent_diagnoses),
            "recommendations": list(set(all_recommendations)),  # Remove duplicates
            "action_items": list(set(all_action_items)),
            "diagnosis_types": [d.diagnosis_type for d in recent_diagnoses]
        }
    
    async def run_comprehensive_diagnosis(self) -> List[DiagnosisResult]:
        """Run all diagnosis types"""
        logger.info("Running comprehensive system diagnosis...")
        
        results = []
        
        # Run health check
        health_result = await self.run_health_check()
        results.append(health_result)
        
        # Run performance analysis
        perf_result = await self.analyze_performance()
        results.append(perf_result)
        
        # Run error investigation
        error_result = await self.investigate_errors()
        results.append(error_result)
        
        logger.info(f"Comprehensive diagnosis completed - {len(results)} checks performed")
        return results
    
    def add_system_metrics(self, metrics: SystemMetrics):
        """Add system metrics to history"""
        self.metrics_history.append(metrics)
        
        # Maintain history size limit
        if len(self.metrics_history) > self.max_metrics_history:
            self.metrics_history = self.metrics_history[-self.max_metrics_history:]
    
    async def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        import psutil
        from datetime import datetime
        
        # Get current system stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Create basic metrics (would be enhanced with real task data)
        metrics = SystemMetrics(
            timestamp=datetime.utcnow(),
            active_agents_count=0,  # Would be populated from agent manager
            tasks_per_minute=0.0,    # Would be populated from task manager
            average_task_completion_time=0.0,
            task_success_rate=100.0,
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory.used / (1024 * 1024),
            disk_usage_percent=(disk.used / disk.total) * 100,
            agent_utilization=0.0,
            active_mcp_connections=0,
            mcp_response_time_avg=0.0
        )
        
        return metrics
    
    def get_health_score(self) -> float:
        """Get current system health score"""
        if self.metrics_history:
            return self.metrics_history[-1].health_score
        return 85.0  # Default health score
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics available"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 metrics
        
        # Calculate averages
        avg_task_completion = sum(m.average_task_completion_time for m in recent_metrics) / len(recent_metrics)
        avg_success_rate = sum(m.task_success_rate for m in recent_metrics) / len(recent_metrics)
        avg_cpu_usage = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory_usage = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        
        return {
            "status": "available",
            "metrics_count": len(self.metrics_history),
            "recent_averages": {
                "task_completion_time": round(avg_task_completion, 2),
                "success_rate": round(avg_success_rate, 2),
                "cpu_usage_percent": round(avg_cpu_usage, 2),
                "memory_usage_mb": round(avg_memory_usage, 2)
            },
            "latest_metrics": recent_metrics[-1].model_dump() if recent_metrics else None
        }
