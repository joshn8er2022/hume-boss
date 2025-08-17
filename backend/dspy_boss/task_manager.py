
"""
Task management using built-in asyncio.Queue for robust task processing
"""

import asyncio
import threading
import uuid
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from .models import TaskDefinition, TaskStatus, TaskPriority, AgentConfig, ReportEntry, FailureEntry


class TaskResult:
    """Result of task execution"""
    
    def __init__(self, task_id: str, success: bool, result: Any = None, error: str = None, duration: float = 0.0):
        self.task_id = task_id
        self.success = success
        self.result = result
        self.error = error
        self.duration = duration
        self.timestamp = datetime.utcnow()


class TaskExecutor:
    """Executes individual tasks"""
    
    def __init__(self, task_manager: 'TaskManager'):
        self.task_manager = task_manager
        
    async def execute_task(self, task: TaskDefinition) -> TaskResult:
        """Execute a single task"""
        start_time = datetime.utcnow()
        task.started_at = start_time
        task.status = TaskStatus.RUNNING
        
        logger.info(f"Executing task: {task.name} (ID: {task.id})")
        
        try:
            # Get the function to execute
            function = self.task_manager.get_task_function(task.function_name)
            if not function:
                raise ValueError(f"Task function '{task.function_name}' not found")
            
            # Execute with timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    self._execute_function(function, task.parameters),
                    timeout=task.timeout
                )
            else:
                result = await self._execute_function(function, task.parameters)
            
            # Task completed successfully
            duration = (datetime.utcnow() - start_time).total_seconds()
            task.completed_at = datetime.utcnow()
            task.status = TaskStatus.COMPLETED
            task.result = result
            
            logger.info(f"Task completed successfully: {task.name} (duration: {duration:.2f}s)")
            
            return TaskResult(
                task_id=task.id,
                success=True,
                result=result,
                duration=duration
            )
            
        except asyncio.TimeoutError:
            error_msg = f"Task timed out after {task.timeout} seconds"
            return self._handle_task_error(task, error_msg, start_time)
            
        except Exception as e:
            error_msg = f"Task execution error: {str(e)}"
            return self._handle_task_error(task, error_msg, start_time)
    
    async def _execute_function(self, function: Callable, parameters: Dict[str, Any]) -> Any:
        """Execute function with parameters"""
        if asyncio.iscoroutinefunction(function):
            return await function(**parameters)
        else:
            # Run synchronous function in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(executor, lambda: function(**parameters))
    
    def _handle_task_error(self, task: TaskDefinition, error_msg: str, start_time: datetime) -> TaskResult:
        """Handle task execution error"""
        duration = (datetime.utcnow() - start_time).total_seconds()
        task.completed_at = datetime.utcnow()
        task.status = TaskStatus.FAILED
        task.error_message = error_msg
        
        logger.error(f"Task failed: {task.name} - {error_msg}")
        
        return TaskResult(
            task_id=task.id,
            success=False,
            error=error_msg,
            duration=duration
        )


class TaskManager:
    """Main task management system using asyncio.Queue"""
    
    def __init__(self, workers: int = 3):
        self.workers = workers
        
        # Task storage
        self.tasks: Dict[str, TaskDefinition] = {}
        self.completed_tasks: Dict[str, TaskDefinition] = {}
        self.failed_tasks: Dict[str, TaskDefinition] = {}
        
        # Task functions registry
        self.task_functions: Dict[str, Callable] = {}
        
        # Queue and executor
        self.task_queue: Optional[asyncio.Queue] = None
        self.task_executor = TaskExecutor(self)
        
        # Management
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_duration": 0.0,
            "tasks_per_minute": 0.0
        }
        self.completed_count = 0
        self.failed_count = 0
        self.start_time = datetime.utcnow()
        
        logger.info(f"TaskManager initialized with {workers} workers")
    
    async def start(self):
        """Start the task management system"""
        if self.is_running:
            logger.warning("TaskManager is already running")
            return
            
        self.task_queue = asyncio.Queue()
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.workers):
            worker_task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(worker_task)
            
        logger.info(f"TaskManager started with {self.workers} workers")
    
    async def stop(self):
        """Stop the task management system"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        logger.info("TaskManager stopped")
    
    async def add_task(self, task: TaskDefinition) -> str:
        """Add a task to the queue"""
        if not self.is_running:
            raise RuntimeError("TaskManager is not running")
            
        self.tasks[task.id] = task
        await self.task_queue.put(task)
        self.stats["total_tasks"] += 1
        
        logger.info(f"Task added to queue: {task.name} (ID: {task.id})")
        return task.id
    
    def register_task_function(self, name: str, function: Callable):
        """Register a task function"""
        self.task_functions[name] = function
        
    def get_task_function(self, name: str) -> Optional[Callable]:
        """Get a registered task function"""
        return self.task_functions.get(name)
    
    async def _worker(self, worker_name: str):
        """Worker coroutine that processes tasks from the queue"""
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                # Execute the task
                result = await self.task_executor.execute_task(task)
                
                # Update statistics and move task to appropriate storage
                if result.success:
                    self.completed_tasks[task.id] = task
                    self.stats["completed_tasks"] += 1
                else:
                    self.failed_tasks[task.id] = task
                    self.stats["failed_tasks"] += 1
                
                # Remove from active tasks
                if task.id in self.tasks:
                    del self.tasks[task.id]
                
                # Mark task as done in queue
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                continue
        
        logger.info(f"Worker {worker_name} stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.stats.copy()
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.task_queue.qsize() if self.task_queue else 0
    
    def get_active_tasks(self) -> Dict[str, TaskDefinition]:
        """Get currently active tasks"""
        return self.tasks.copy()
    
    def get_all_tasks(self) -> List[TaskDefinition]:
        """Get all tasks (active and completed)"""
        return list(self.tasks.values())
    
    def get_throughput(self) -> float:
        """Get tasks per minute throughput"""
        # Simple calculation - can be enhanced with time windows
        return self.completed_count / max(1, (datetime.utcnow() - self.start_time).total_seconds() / 60) if hasattr(self, 'start_time') else 0.0
