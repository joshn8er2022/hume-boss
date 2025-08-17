
"""
Agent communication system for coordinating between Boss and subordinate agents
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Callable
from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger
import uuid

from .agent_hierarchy import AgentHierarchy, BaseAgent, BossAgent, SubordinateAgent


class MessageType(Enum):
    """Types of messages in the agent communication system"""
    
    # Task-related messages
    TASK_ASSIGNMENT = "task_assignment"
    TASK_UPDATE = "task_update"
    TASK_COMPLETION = "task_completion"
    TASK_FAILURE = "task_failure"
    
    # Coordination messages
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    STATUS_UPDATE = "status_update"
    
    # Control messages
    AGENT_SPAWN_NOTIFICATION = "agent_spawn_notification"
    AGENT_TERMINATION_NOTIFICATION = "agent_termination_notification"
    SYSTEM_ANNOUNCEMENT = "system_announcement"
    
    # Request/Response messages
    CAPABILITY_INQUIRY = "capability_inquiry"
    RESOURCE_REQUEST = "resource_request"
    INFORMATION_REQUEST = "information_request"
    
    # Error and warning messages
    ERROR_REPORT = "error_report"
    WARNING_NOTIFICATION = "warning_notification"
    ESCALATION = "escalation"


class MessagePriority(Enum):
    """Priority levels for messages"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class Message(BaseModel):
    """Individual message in the communication system"""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = Field(..., description="Type of message")
    priority: MessagePriority = Field(default=MessagePriority.NORMAL)
    
    # Routing information
    sender_id: str = Field(..., description="ID of sending agent")
    recipient_id: str = Field(..., description="ID of receiving agent")
    broadcast: bool = Field(default=False, description="Whether this is a broadcast message")
    
    # Message content
    subject: str = Field(..., description="Message subject")
    content: Dict[str, Any] = Field(..., description="Message content")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    requires_acknowledgment: bool = Field(default=False)
    acknowledged_at: Optional[datetime] = None
    
    # Response tracking
    response_to: Optional[str] = None  # Message ID this is responding to
    response_expected: bool = Field(default=False)
    response_timeout: Optional[datetime] = None
    
    # Processing tracking
    delivered_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    processing_result: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def is_response_overdue(self) -> bool:
        """Check if response is overdue"""
        if not self.response_expected or not self.response_timeout:
            return False
        return datetime.utcnow() > self.response_timeout


class MessageQueue(BaseModel):
    """Message queue for an agent"""
    
    agent_id: str = Field(..., description="ID of agent this queue belongs to")
    messages: List[Message] = Field(default_factory=list)
    max_size: int = Field(default=100, description="Maximum queue size")
    
    def add_message(self, message: Message) -> bool:
        """Add message to queue"""
        
        # Check if queue is full
        if len(self.messages) >= self.max_size:
            # Remove oldest low-priority message to make room
            for i, msg in enumerate(self.messages):
                if msg.priority == MessagePriority.LOW:
                    self.messages.pop(i)
                    break
            else:
                # If no low priority messages, fail to add
                logger.warning(f"Message queue full for agent {self.agent_id}")
                return False
        
        # Insert message in priority order
        inserted = False
        for i, existing_msg in enumerate(self.messages):
            if message.priority.value > existing_msg.priority.value:
                self.messages.insert(i, message)
                inserted = True
                break
        
        if not inserted:
            self.messages.append(message)
        
        message.delivered_at = datetime.utcnow()
        return True
    
    def get_next_message(self) -> Optional[Message]:
        """Get next message for processing"""
        
        if not self.messages:
            return None
        
        # Return highest priority message
        return self.messages.pop(0)
    
    def get_messages_by_type(self, message_type: MessageType) -> List[Message]:
        """Get all messages of a specific type"""
        return [msg for msg in self.messages if msg.message_type == message_type]
    
    def remove_expired_messages(self) -> int:
        """Remove expired messages and return count"""
        
        initial_count = len(self.messages)
        self.messages = [msg for msg in self.messages if not msg.is_expired()]
        return initial_count - len(self.messages)
    
    def get_queue_summary(self) -> Dict[str, Any]:
        """Get summary of queue state"""
        
        type_counts = {}
        priority_counts = {}
        
        for msg in self.messages:
            # Count by type
            msg_type = msg.message_type.value
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
            
            # Count by priority
            priority = msg.priority.value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return {
            "total_messages": len(self.messages),
            "by_type": type_counts,
            "by_priority": priority_counts,
            "oldest_message": self.messages[-1].timestamp.isoformat() if self.messages else None,
            "newest_message": self.messages[0].timestamp.isoformat() if self.messages else None
        }


class AgentCommunicationHub:
    """
    Central communication hub for agent-to-agent messaging
    Handles routing, queuing, and processing of messages between agents
    """
    
    def __init__(self, agent_hierarchy: AgentHierarchy):
        self.agent_hierarchy = agent_hierarchy
        
        # Message queues for each agent
        self.message_queues: Dict[str, MessageQueue] = {}
        
        # Message handlers
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        
        # Communication metrics
        self.metrics = {
            "total_messages": 0,
            "messages_by_type": {},
            "messages_by_priority": {},
            "delivery_failures": 0,
            "processing_failures": 0,
            "average_delivery_time": 0.0
        }
        
        # Message routing rules
        self.routing_rules: Dict[str, Callable] = {}
        
        # Background processing
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("AgentCommunicationHub initialized")
    
    async def start_processing(self):
        """Start background message processing"""
        
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._message_processing_loop())
        
        logger.info("Message processing started")
    
    async def stop_processing(self):
        """Stop background message processing"""
        
        self.is_running = False
        
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Message processing stopped")
    
    async def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: MessageType,
        subject: str,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        requires_ack: bool = False,
        expires_in_seconds: Optional[int] = None,
        response_expected: bool = False,
        response_timeout_seconds: Optional[int] = None
    ) -> str:
        """
        Send a message between agents
        
        Args:
            sender_id: ID of sending agent
            recipient_id: ID of receiving agent
            message_type: Type of message
            subject: Message subject
            content: Message content
            priority: Message priority
            requires_ack: Whether acknowledgment is required
            expires_in_seconds: Message expiration time
            response_expected: Whether a response is expected
            response_timeout_seconds: Timeout for expected response
        
        Returns:
            Message ID
        """
        
        try:
            # Create message
            message = Message(
                message_type=message_type,
                priority=priority,
                sender_id=sender_id,
                recipient_id=recipient_id,
                subject=subject,
                content=content,
                requires_acknowledgment=requires_ack,
                response_expected=response_expected
            )
            
            # Set expiration if specified
            if expires_in_seconds:
                message.expires_at = datetime.utcnow() + timedelta(seconds=expires_in_seconds)
            
            # Set response timeout if specified
            if response_timeout_seconds:
                message.response_timeout = datetime.utcnow() + timedelta(seconds=response_timeout_seconds)
            
            # Route message
            success = await self._route_message(message)
            
            if success:
                # Update metrics
                self.metrics["total_messages"] += 1
                msg_type = message_type.value
                self.metrics["messages_by_type"][msg_type] = self.metrics["messages_by_type"].get(msg_type, 0) + 1
                priority_val = priority.value
                self.metrics["messages_by_priority"][priority_val] = self.metrics["messages_by_priority"].get(priority_val, 0) + 1
                
                logger.debug(f"Message sent: {message.message_id} from {sender_id} to {recipient_id}")
                return message.message_id
            else:
                self.metrics["delivery_failures"] += 1
                logger.warning(f"Failed to deliver message from {sender_id} to {recipient_id}")
                return ""
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.metrics["delivery_failures"] += 1
            return ""
    
    async def send_broadcast(
        self,
        sender_id: str,
        message_type: MessageType,
        subject: str,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        target_roles: Optional[List[str]] = None
    ) -> List[str]:
        """
        Send broadcast message to multiple agents
        
        Args:
            sender_id: ID of sending agent
            message_type: Type of message
            subject: Message subject
            content: Message content  
            priority: Message priority
            target_roles: Specific roles to target (None for all)
        
        Returns:
            List of message IDs
        """
        
        message_ids = []
        
        try:
            # Get target agents
            target_agents = []
            
            if target_roles:
                # Target specific roles
                for agent in self.agent_hierarchy.agents.values():
                    if agent.role.value in target_roles and agent.agent_id != sender_id:
                        target_agents.append(agent)
            else:
                # Target all agents except sender
                target_agents = [
                    agent for agent in self.agent_hierarchy.agents.values()
                    if agent.agent_id != sender_id
                ]
            
            # Send to each target
            for agent in target_agents:
                message_id = await self.send_message(
                    sender_id=sender_id,
                    recipient_id=agent.agent_id,
                    message_type=message_type,
                    subject=subject,
                    content=content,
                    priority=priority
                )
                
                if message_id:
                    message_ids.append(message_id)
            
            logger.info(f"Broadcast sent to {len(message_ids)} agents")
            
        except Exception as e:
            logger.error(f"Error sending broadcast: {e}")
        
        return message_ids
    
    async def get_messages_for_agent(self, agent_id: str, message_type: Optional[MessageType] = None) -> List[Message]:
        """
        Get messages for a specific agent
        
        Args:
            agent_id: Agent ID
            message_type: Optional filter by message type
        
        Returns:
            List of messages
        """
        
        if agent_id not in self.message_queues:
            return []
        
        queue = self.message_queues[agent_id]
        
        if message_type:
            return queue.get_messages_by_type(message_type)
        else:
            return queue.messages.copy()
    
    async def process_next_message_for_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Process the next message for a specific agent
        
        Args:
            agent_id: Agent ID
        
        Returns:
            Processing result or None if no messages
        """
        
        if agent_id not in self.message_queues:
            return None
        
        queue = self.message_queues[agent_id]
        message = queue.get_next_message()
        
        if not message:
            return None
        
        try:
            # Process the message
            processing_result = await self._process_message(message)
            
            message.processed_at = datetime.utcnow()
            message.processing_result = processing_result
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            self.metrics["processing_failures"] += 1
            return {"error": str(e)}
    
    async def acknowledge_message(self, message_id: str, agent_id: str) -> bool:
        """
        Acknowledge receipt of a message
        
        Args:
            message_id: Message ID
            agent_id: Agent acknowledging the message
        
        Returns:
            True if acknowledgment successful
        """
        
        try:
            # Find message in agent's queue
            if agent_id not in self.message_queues:
                return False
            
            queue = self.message_queues[agent_id]
            
            for message in queue.messages:
                if message.message_id == message_id:
                    message.acknowledged_at = datetime.utcnow()
                    logger.debug(f"Message {message_id} acknowledged by {agent_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging message: {e}")
            return False
    
    async def send_response(
        self,
        original_message_id: str,
        sender_id: str,
        response_content: Dict[str, Any],
        success: bool = True
    ) -> str:
        """
        Send a response to a message
        
        Args:
            original_message_id: ID of message being responded to
            sender_id: ID of agent sending response
            response_content: Response content
            success: Whether the response indicates success
        
        Returns:
            Response message ID
        """
        
        try:
            # Find original message to get sender info
            original_sender = None
            
            for queue in self.message_queues.values():
                for msg in queue.messages:
                    if msg.message_id == original_message_id:
                        original_sender = msg.sender_id
                        break
                if original_sender:
                    break
            
            if not original_sender:
                logger.warning(f"Cannot find original message {original_message_id} to respond to")
                return ""
            
            # Send response
            response_id = await self.send_message(
                sender_id=sender_id,
                recipient_id=original_sender,
                message_type=MessageType.COORDINATION_RESPONSE,
                subject=f"Response to message {original_message_id}",
                content={
                    "original_message_id": original_message_id,
                    "success": success,
                    "response_data": response_content
                },
                priority=MessagePriority.NORMAL
            )
            
            return response_id
            
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            return ""
    
    def add_message_handler(self, message_type: MessageType, handler: Callable):
        """Add a message handler for a specific type"""
        
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        logger.debug(f"Added handler for message type: {message_type.value}")
    
    def remove_message_handler(self, message_type: MessageType, handler: Callable) -> bool:
        """Remove a message handler"""
        
        if message_type in self.message_handlers:
            if handler in self.message_handlers[message_type]:
                self.message_handlers[message_type].remove(handler)
                return True
        
        return False
    
    async def _route_message(self, message: Message) -> bool:
        """Route message to appropriate agent queue"""
        
        recipient_id = message.recipient_id
        
        # Check if recipient exists
        recipient = self.agent_hierarchy.get_agent_by_id(recipient_id)
        if not recipient:
            logger.warning(f"Cannot route message: recipient {recipient_id} not found")
            return False
        
        # Ensure queue exists
        if recipient_id not in self.message_queues:
            self.message_queues[recipient_id] = MessageQueue(agent_id=recipient_id)
        
        # Add message to queue
        success = self.message_queues[recipient_id].add_message(message)
        
        if success:
            logger.debug(f"Message routed to {recipient_id}: {message.message_id}")
        
        return success
    
    async def _process_message(self, message: Message) -> Dict[str, Any]:
        """Process a message using registered handlers"""
        
        try:
            processing_result = {
                "message_id": message.message_id,
                "processed_at": datetime.utcnow().isoformat(),
                "success": True,
                "results": []
            }
            
            # Run handlers for this message type
            if message.message_type in self.message_handlers:
                for handler in self.message_handlers[message.message_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            result = await handler(message)
                        else:
                            result = handler(message)
                        
                        processing_result["results"].append(result)
                        
                    except Exception as handler_error:
                        logger.warning(f"Message handler failed: {handler_error}")
                        processing_result["success"] = False
                        processing_result["results"].append({"error": str(handler_error)})
            
            # Default processing based on message type
            if not processing_result["results"]:
                processing_result["results"].append(await self._default_message_processing(message))
            
            return processing_result
            
        except Exception as e:
            return {
                "message_id": message.message_id,
                "processed_at": datetime.utcnow().isoformat(),
                "success": False,
                "error": str(e)
            }
    
    async def _default_message_processing(self, message: Message) -> Dict[str, Any]:
        """Default processing for messages without specific handlers"""
        
        # Basic acknowledgment for messages requiring it
        if message.requires_acknowledgment:
            await self.acknowledge_message(message.message_id, message.recipient_id)
        
        return {
            "action": "default_processing",
            "message_type": message.message_type.value,
            "acknowledged": message.requires_acknowledgment
        }
    
    async def _message_processing_loop(self):
        """Background loop for processing messages"""
        
        logger.info("Message processing loop started")
        
        try:
            while self.is_running:
                # Process messages for all agents
                for agent_id in list(self.message_queues.keys()):
                    try:
                        # Process up to 5 messages per agent per cycle
                        for _ in range(5):
                            result = await self.process_next_message_for_agent(agent_id)
                            if not result:
                                break  # No more messages for this agent
                        
                        # Clean up expired messages
                        queue = self.message_queues[agent_id]
                        expired_count = queue.remove_expired_messages()
                        if expired_count > 0:
                            logger.debug(f"Removed {expired_count} expired messages for {agent_id}")
                    
                    except Exception as agent_error:
                        logger.warning(f"Error processing messages for {agent_id}: {agent_error}")
                
                # Sleep between processing cycles
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            logger.info("Message processing loop cancelled")
        except Exception as e:
            logger.error(f"Message processing loop error: {e}")
        
        logger.info("Message processing loop ended")
    
    # Public interface methods
    
    def get_communication_metrics(self) -> Dict[str, Any]:
        """Get communication system metrics"""
        
        queue_summaries = {}
        total_queued_messages = 0
        
        for agent_id, queue in self.message_queues.items():
            summary = queue.get_queue_summary()
            queue_summaries[agent_id] = summary
            total_queued_messages += summary["total_messages"]
        
        return {
            **self.metrics,
            "total_queued_messages": total_queued_messages,
            "active_queues": len(self.message_queues),
            "queue_summaries": queue_summaries,
            "processing_active": self.is_running
        }
    
    def get_agent_message_summary(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get message summary for specific agent"""
        
        if agent_id not in self.message_queues:
            return None
        
        return self.message_queues[agent_id].get_queue_summary()
    
    async def clear_agent_messages(self, agent_id: str, message_type: Optional[MessageType] = None) -> int:
        """
        Clear messages for an agent
        
        Args:
            agent_id: Agent ID
            message_type: Optional filter by type
        
        Returns:
            Number of messages cleared
        """
        
        if agent_id not in self.message_queues:
            return 0
        
        queue = self.message_queues[agent_id]
        initial_count = len(queue.messages)
        
        if message_type:
            queue.messages = [msg for msg in queue.messages if msg.message_type != message_type]
        else:
            queue.messages = []
        
        cleared_count = initial_count - len(queue.messages)
        
        logger.info(f"Cleared {cleared_count} messages for {agent_id}")
        return cleared_count
    
    def create_agent_queue(self, agent_id: str) -> bool:
        """Create message queue for new agent"""
        
        if agent_id in self.message_queues:
            return False
        
        self.message_queues[agent_id] = MessageQueue(agent_id=agent_id)
        logger.debug(f"Created message queue for agent {agent_id}")
        return True
    
    def remove_agent_queue(self, agent_id: str) -> bool:
        """Remove message queue for terminated agent"""
        
        if agent_id not in self.message_queues:
            return False
        
        del self.message_queues[agent_id]
        logger.debug(f"Removed message queue for agent {agent_id}")
        return True
