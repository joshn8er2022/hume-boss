
"""
Pydantic BaseModel classes for the DSPY Boss system
"""

from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import uuid


class AgentType(str, Enum):
    """Types of agents in the system"""
    AGENTIC = "agentic"  # Numbered versions of boss running autonomously
    HUMAN = "human"      # Human agents loaded from YAML


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Task priority levels (lower number = higher priority)"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class AgentConfig(BaseModel):
    """Configuration for an agent (both agentic and human)"""
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: AgentType
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    max_concurrent_tasks: int = Field(default=3)
    is_available: bool = Field(default=True)
    
    # For human agents
    contact_method: Optional[str] = None  # "slack", "close_crm", etc.
    contact_details: Optional[Dict[str, Any]] = None
    
    # For agentic agents
    model_name: Optional[str] = None
    prompt_signature: Optional[str] = None
    thread_id: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: Optional[datetime] = None


class TaskDefinition(BaseModel):
    """Definition of a task to be executed"""
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    
    # Task execution details
    function_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout: Optional[int] = None  # seconds
    
    # Assignment details
    assigned_agent_id: Optional[str] = None
    requires_human: bool = Field(default=False)
    
    # Tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional[Any] = None
    error_message: Optional[str] = None
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)


class MCPServerConfig(BaseModel):
    """Configuration for MCP servers"""
    model_config = ConfigDict(extra="allow")
    
    name: str
    url: str
    api_key: Optional[str] = None
    instance_id: Optional[str] = None
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    is_active: bool = Field(default=True)
    connection_timeout: int = Field(default=30)
    retry_attempts: int = Field(default=3)
    
    # Connection details
    headers: Dict[str, str] = Field(default_factory=dict)
    auth_method: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_connected: Optional[datetime] = None


class ReportEntry(BaseModel):
    """Report entry for tracking system activities"""
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    level: Literal["info", "warning", "error", "debug"] = "info"
    category: str  # "task", "agent", "system", "mcp", etc.
    message: str
    details: Optional[Dict[str, Any]] = None
    
    # Related entities
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    mcp_server: Optional[str] = None


class FailureEntry(BaseModel):
    """Failure tracking entry"""
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    failure_type: str  # "task_failure", "agent_error", "mcp_connection", etc.
    description: str
    error_details: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Context
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    mcp_server: Optional[str] = None
    
    # Resolution
    is_resolved: bool = Field(default=False)
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None


class BossStateData(BaseModel):
    """Internal state data for the boss"""
    model_config = ConfigDict(extra="allow")
    
    current_workload: int = Field(default=0)
    active_agents: List[str] = Field(default_factory=list)
    pending_tasks: List[str] = Field(default_factory=list)
    completed_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)
    
    # Performance metrics
    total_tasks_processed: int = Field(default=0)
    success_rate: float = Field(default=0.0)
    average_task_duration: float = Field(default=0.0)
    
    # System health
    last_health_check: Optional[datetime] = None
    system_errors: List[str] = Field(default_factory=list)
    
    # Reflection data
    last_reflection: Optional[datetime] = None
    reflection_notes: Optional[str] = None
    improvement_actions: List[str] = Field(default_factory=list)


class PromptSignature(BaseModel):
    """DSPY prompt signature configuration"""
    model_config = ConfigDict(extra="allow")
    
    name: str
    signature: str
    description: Optional[str] = None
    input_fields: List[str] = Field(default_factory=list)
    output_fields: List[str] = Field(default_factory=list)
    examples: List[Dict[str, str]] = Field(default_factory=list)
    
    # React agent specific
    is_react_agent: bool = Field(default=False)
    react_steps: Optional[int] = None
    react_tools: List[str] = Field(default_factory=list)


class SystemMetrics(BaseModel):
    """System performance metrics"""
    model_config = ConfigDict(extra="allow")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Task metrics
    tasks_per_minute: float = Field(default=0.0)
    average_task_completion_time: float = Field(default=0.0)
    task_success_rate: float = Field(default=0.0)
    
    # Agent metrics
    active_agents_count: int = Field(default=0)
    agent_utilization: float = Field(default=0.0)
    
    # System metrics
    memory_usage_mb: float = Field(default=0.0)
    cpu_usage_percent: float = Field(default=0.0)
    disk_usage_percent: float = Field(default=0.0)
    
    # MCP metrics
    active_mcp_connections: int = Field(default=0)
    mcp_response_time_avg: float = Field(default=0.0)


class DiagnosisResult(BaseModel):
    """Result from self-diagnosis"""
    model_config = ConfigDict(extra="allow")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    diagnosis_type: str  # "health_check", "performance_analysis", "error_investigation"
    
    # Results
    status: Literal["healthy", "warning", "critical"] = "healthy"
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    
    # Code execution results (from DSPY Python interpreter)
    code_executed: Optional[str] = None
    execution_output: Optional[str] = None
    execution_error: Optional[str] = None


# New models for autonomous DSPY-driven system
class SystemState(BaseModel):
    """Complete system state snapshot"""
    model_config = ConfigDict(extra="allow")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    iteration_count: int = Field(default=0)
    autonomous_mode: bool = Field(default=False)
    active_agents: List[Dict[str, Any]] = Field(default_factory=list)
    recent_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    system_health: Dict[str, Any] = Field(default_factory=dict)
    current_phase: Optional[str] = None


class IterationResult(BaseModel):
    """Complete iteration result from autonomous engine"""
    model_config = ConfigDict(extra="allow")
    
    iteration_id: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    pre_processing: Optional[Dict[str, Any]] = None
    boss_decision: Optional[Dict[str, Any]] = None
    execution: Optional[Dict[str, Any]] = None
    next_prep: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None


class LearningEntry(BaseModel):
    """System learning entry from autonomous operations"""
    model_config = ConfigDict(extra="allow")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    learning_type: str  # "iteration_analysis", "error_analysis", "pattern_recognition"
    content: Dict[str, Any] = Field(default_factory=dict)
    iteration_id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence_score: Optional[float] = None
    applied: bool = Field(default=False)


class AgentHierarchy(BaseModel):
    """Agent hierarchy state"""
    model_config = ConfigDict(extra="allow")
    
    boss_agent: Optional[Dict[str, Any]] = None
    subordinate_agents: List[Dict[str, Any]] = Field(default_factory=list)
    total_agents: int = Field(default=1)  # Including boss
    next_agent_id: int = Field(default=1)
    hierarchy_established: datetime = Field(default_factory=datetime.utcnow)


class LLMProviderConfig(BaseModel):
    """LLM Provider configuration"""
    model_config = ConfigDict(extra="allow")
    
    provider_name: str
    api_key: Optional[str] = None  # Will be encrypted in storage
    base_url: Optional[str] = None
    model: str = "gpt-4-turbo-preview"
    max_tokens: int = Field(default=4000)
    temperature: float = Field(default=0.7)
    is_active: bool = Field(default=True)
    is_initialized: bool = Field(default=False)
    last_tested: Optional[datetime] = None
    test_status: Optional[str] = None


class AutonomousConfig(BaseModel):
    """Configuration for autonomous operation"""
    model_config = ConfigDict(extra="allow")
    
    is_enabled: bool = Field(default=False)
    iteration_interval_seconds: float = Field(default=1.0)
    max_concurrent_agents: int = Field(default=10)
    state_history_limit: int = Field(default=100)
    auto_scale_agents: bool = Field(default=True)
    error_recovery_enabled: bool = Field(default=True)
    
    # DSPY specific settings
    primary_llm_provider: Optional[str] = None
    fallback_llm_provider: Optional[str] = None
    signature_optimization: bool = Field(default=True)
    retrieval_augmented: bool = Field(default=True)
