"""
FastAPI server for DSPY Boss system - Bridges frontend and backend
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from .boss import DSPYBoss
from .models import (
    TaskDefinition, TaskPriority, AgentConfig, SystemMetrics, 
    BossStateData, LLMProviderConfig
)
from .state_machine import BossState


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


# Global instances
boss_instance: Optional[DSPYBoss] = None
connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global boss_instance
    
    # Startup
    logger.info("Starting DSPY Boss API server...")
    boss_instance = DSPYBoss()
    
    # Start background task for real-time updates
    update_task = asyncio.create_task(broadcast_updates())
    
    yield
    
    # Shutdown
    logger.info("Shutting down DSPY Boss API server...")
    update_task.cancel()
    if boss_instance:
        await boss_instance.shutdown()


app = FastAPI(
    title="DSPY Boss API",
    description="API server for DSPY Boss system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:51137", "http://localhost:56509"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class BossStateUpdate(BaseModel):
    state: str
    reason: Optional[str] = None


class AgentModelUpdate(BaseModel):
    agent_id: str
    model_name: str
    provider: Optional[str] = None


class TaskCreate(BaseModel):
    title: str
    description: str
    priority: str = "MEDIUM"
    assigned_agent: Optional[str] = None
    capabilities_required: List[str] = []


# API Routes

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "boss_running": boss_instance is not None and boss_instance.is_running
    }


@app.get("/api/system/overview")
async def get_system_overview():
    """Get system overview with real-time metrics"""
    if not boss_instance:
        raise HTTPException(status_code=503, detail="Boss system not initialized")
    
    try:
        # Get current system state
        current_state = boss_instance.state_manager.current_state
        metrics = await boss_instance.get_system_metrics()
        
        return {
            "boss_state": current_state.value,
            "state_data": boss_instance.state_manager.get_state_data(),
            "metrics": {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": (datetime.utcnow() - boss_instance.start_time).total_seconds() if boss_instance.start_time else 0,
                "total_agents": len(boss_instance.agent_manager.agents),
                "active_agents": len([a for a in boss_instance.agent_manager.agents.values() if a.is_available]),
                "total_tasks": boss_instance.task_manager.get_queue_size(),
                "completed_tasks": boss_instance.task_manager.completed_count,
                "failed_tasks": boss_instance.task_manager.failed_count,
                "tasks_per_minute": boss_instance.task_manager.get_throughput(),
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "memory_usage_mb": metrics.memory_usage_mb,
                "disk_usage_percent": metrics.disk_usage_percent
            },
            "health_score": boss_instance.diagnosis_system.get_health_score()
        }
    except Exception as e:
        logger.error(f"Error getting system overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/system/boss-state")
async def get_boss_state():
    """Get current boss state"""
    if not boss_instance:
        raise HTTPException(status_code=503, detail="Boss system not initialized")
    
    return {
        "state": boss_instance.state_manager.current_state.value,
        "data": boss_instance.state_manager.get_state_data(),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/system/boss-state")
async def set_boss_state(state_update: BossStateUpdate):
    """Set boss state"""
    if not boss_instance:
        raise HTTPException(status_code=503, detail="Boss system not initialized")
    
    try:
        # Validate state
        new_state = BossState(state_update.state)
        
        # Transition to new state
        boss_instance.state_manager.transition_to(
            new_state, 
            state_update.reason or "Manual state change via API"
        )
        
        # Broadcast update
        await connection_manager.broadcast({
            "type": "boss_state_change",
            "data": {
                "state": new_state.value,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": state_update.reason
            }
        })
        
        return {"success": True, "new_state": new_state.value}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid state: {e}")
    except Exception as e:
        logger.error(f"Error setting boss state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents")
async def get_agents():
    """Get all agents"""
    if not boss_instance:
        raise HTTPException(status_code=503, detail="Boss system not initialized")
    
    agents_data = []
    for agent_id, agent in boss_instance.agent_manager.agents.items():
        agents_data.append({
            "id": agent_id,
            "name": agent.name,
            "type": agent.type,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "is_available": agent.is_available,
            "max_concurrent_tasks": agent.max_concurrent_tasks,
            "current_tasks": len(agent.current_tasks) if hasattr(agent, 'current_tasks') else 0,
            "model_name": agent.model_name,
            "contact_method": agent.contact_method,
            "created_at": agent.created_at,
            "last_active": agent.last_active
        })
    
    return agents_data


@app.post("/api/agents/{agent_id}/model")
async def update_agent_model(agent_id: str, model_update: AgentModelUpdate):
    """Update agent's model"""
    if not boss_instance:
        raise HTTPException(status_code=503, detail="Boss system not initialized")
    
    try:
        # Update agent model
        success = await boss_instance.agent_manager.update_agent_model(
            agent_id, 
            model_update.model_name,
            model_update.provider
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Broadcast update
        await connection_manager.broadcast({
            "type": "agent_model_update",
            "data": {
                "agent_id": agent_id,
                "model_name": model_update.model_name,
                "provider": model_update.provider,
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"Error updating agent model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/llm-providers")
async def get_llm_providers():
    """Get available LLM providers and models"""
    if not boss_instance:
        raise HTTPException(status_code=503, detail="Boss system not initialized")
    
    try:
        providers = boss_instance.llm_provider_manager.get_available_providers()
        return {
            "providers": [
                {
                    "name": provider.name,
                    "type": provider.provider_type,
                    "models": provider.available_models,
                    "is_active": provider.is_active,
                    "config": provider.config
                }
                for provider in providers
            ]
        }
    except Exception as e:
        logger.error(f"Error getting LLM providers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks")
async def get_tasks():
    """Get all tasks"""
    if not boss_instance:
        raise HTTPException(status_code=503, detail="Boss system not initialized")
    
    try:
        tasks = boss_instance.task_manager.get_all_tasks()
        return [
            {
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "priority": task.priority.value,
                "status": task.status.value,
                "assigned_agent": task.assigned_agent,
                "capabilities_required": task.capabilities_required,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat() if task.updated_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
            for task in tasks
        ]
    except Exception as e:
        logger.error(f"Error getting tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tasks")
async def create_task(task_data: TaskCreate):
    """Create a new task"""
    if not boss_instance:
        raise HTTPException(status_code=503, detail="Boss system not initialized")
    
    try:
        # Create task
        task = TaskDefinition(
            title=task_data.title,
            description=task_data.description,
            priority=TaskPriority(task_data.priority),
            assigned_agent=task_data.assigned_agent,
            capabilities_required=task_data.capabilities_required
        )
        
        # Add to task manager
        task_id = await boss_instance.task_manager.add_task(task)
        
        # Broadcast update
        await connection_manager.broadcast({
            "type": "task_created",
            "data": {
                "task_id": task_id,
                "title": task_data.title,
                "priority": task_data.priority,
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
        return {"success": True, "task_id": task_id}
        
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mcp-servers")
async def get_mcp_servers():
    """Get MCP server status"""
    if not boss_instance:
        raise HTTPException(status_code=503, detail="Boss system not initialized")
    
    try:
        servers = []
        for server_id, server in boss_instance.mcp_manager.servers.items():
            servers.append({
                "id": server_id,
                "name": server.config.name,
                "url": server.config.url,
                "description": server.config.description,
                "capabilities": server.config.capabilities,
                "is_active": server.config.is_active,
                "is_connected": server.is_connected,
                "last_connected": server.last_connected.isoformat() if server.last_connected else None,
                "connection_timeout": server.config.connection_timeout,
                "retry_attempts": server.config.retry_attempts
            })
        
        return servers
    except Exception as e:
        logger.error(f"Error getting MCP servers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for now (can be extended for client commands)
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


async def broadcast_updates():
    """Background task to broadcast system updates"""
    while True:
        try:
            if boss_instance and boss_instance.is_running:
                # Get current system state
                overview = {
                    "type": "system_update",
                    "data": {
                        "boss_state": boss_instance.state_manager.current_state.value,
                        "timestamp": datetime.utcnow().isoformat(),
                        "active_agents": len([a for a in boss_instance.agent_manager.agents.values() if a.is_available]),
                        "total_tasks": boss_instance.task_manager.get_queue_size(),
                        "health_score": boss_instance.diagnosis_system.get_health_score()
                    }
                }
                
                await connection_manager.broadcast(overview)
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            logger.error(f"Error in broadcast updates: {e}")
            await asyncio.sleep(5)  # Wait longer on error


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)