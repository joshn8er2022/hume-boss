
"""
MCP (Model Context Protocol) server integration
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from loguru import logger
from pydantic import BaseModel

from .models import MCPServerConfig, ReportEntry, FailureEntry


class MCPResponse(BaseModel):
    """Response from MCP server"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None


class MCPConnection:
    """Individual MCP server connection"""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_connected = False
        self.last_health_check: Optional[datetime] = None
        self.consecutive_failures = 0
        self.total_requests = 0
        self.successful_requests = 0
        
    async def connect(self) -> bool:
        """Establish connection to MCP server"""
        try:
            # Create session with timeout and headers
            timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
            headers = self.config.headers.copy()
            
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            if self.config.instance_id:
                headers["X-Instance-ID"] = self.config.instance_id
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
            
            # Test connection with health check
            health_response = await self.health_check()
            if health_response.success:
                self.is_connected = True
                self.config.last_connected = datetime.utcnow()
                self.consecutive_failures = 0
                logger.info(f"Connected to MCP server: {self.config.name}")
                return True
            else:
                logger.error(f"Failed to connect to MCP server {self.config.name}: {health_response.error}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to MCP server {self.config.name}: {e}")
            self.consecutive_failures += 1
            return False
    
    async def disconnect(self):
        """Close connection to MCP server"""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False
        logger.info(f"Disconnected from MCP server: {self.config.name}")
    
    async def health_check(self) -> MCPResponse:
        """Perform health check on MCP server"""
        if not self.session:
            return MCPResponse(success=False, error="No active session")
        
        try:
            start_time = datetime.utcnow()
            
            # Try to access the base URL or a health endpoint
            health_url = f"{self.config.url.rstrip('/')}/health" if "/health" not in self.config.url else self.config.url
            
            async with self.session.get(health_url) as response:
                response_time = (datetime.utcnow() - start_time).total_seconds()
                
                if response.status == 200:
                    self.last_health_check = datetime.utcnow()
                    return MCPResponse(
                        success=True,
                        status_code=response.status,
                        response_time=response_time
                    )
                else:
                    return MCPResponse(
                        success=False,
                        error=f"Health check failed with status {response.status}",
                        status_code=response.status,
                        response_time=response_time
                    )
                    
        except Exception as e:
            return MCPResponse(
                success=False,
                error=f"Health check error: {str(e)}"
            )
    
    async def send_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> MCPResponse:
        """Send request to MCP server"""
        if not self.session or not self.is_connected:
            return MCPResponse(success=False, error="Not connected to server")
        
        try:
            start_time = datetime.utcnow()
            url = f"{self.config.url.rstrip('/')}/{endpoint.lstrip('/')}"
            
            self.total_requests += 1
            
            # Send request based on method
            if method.upper() == "GET":
                async with self.session.get(url, params=data) as response:
                    return await self._process_response(response, start_time)
            elif method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    return await self._process_response(response, start_time)
            elif method.upper() == "PUT":
                async with self.session.put(url, json=data) as response:
                    return await self._process_response(response, start_time)
            elif method.upper() == "DELETE":
                async with self.session.delete(url) as response:
                    return await self._process_response(response, start_time)
            else:
                return MCPResponse(success=False, error=f"Unsupported HTTP method: {method}")
                
        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"Error sending request to {self.config.name}: {e}")
            return MCPResponse(success=False, error=str(e))
    
    async def _process_response(self, response: aiohttp.ClientResponse, start_time: datetime) -> MCPResponse:
        """Process HTTP response"""
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        try:
            if response.status >= 200 and response.status < 300:
                self.successful_requests += 1
                self.consecutive_failures = 0
                
                # Try to parse JSON response
                try:
                    data = await response.json()
                except:
                    data = await response.text()
                
                return MCPResponse(
                    success=True,
                    data=data,
                    status_code=response.status,
                    response_time=response_time
                )
            else:
                self.consecutive_failures += 1
                error_text = await response.text()
                return MCPResponse(
                    success=False,
                    error=f"HTTP {response.status}: {error_text}",
                    status_code=response.status,
                    response_time=response_time
                )
                
        except Exception as e:
            self.consecutive_failures += 1
            return MCPResponse(
                success=False,
                error=f"Error processing response: {str(e)}",
                status_code=response.status,
                response_time=response_time
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "name": self.config.name,
            "is_connected": self.is_connected,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": round(success_rate, 2),
            "consecutive_failures": self.consecutive_failures,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "last_connected": self.config.last_connected.isoformat() if self.config.last_connected else None
        }


class MCPManager:
    """Manages multiple MCP server connections"""
    
    def __init__(self, servers: Dict[str, MCPServerConfig]):
        self.servers = servers
        self.connections: Dict[str, MCPConnection] = {}
        self.health_check_task: Optional[asyncio.Task] = None
        self.health_check_interval = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize all MCP connections"""
        logger.info(f"Initializing {len(self.servers)} MCP server connections...")
        
        for name, config in self.servers.items():
            if config.is_active:
                connection = MCPConnection(config)
                self.connections[name] = connection
                
                # Attempt to connect
                success = await connection.connect()
                if not success:
                    logger.warning(f"Failed to connect to MCP server: {name}")
        
        # Start health check task
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Initialized {len(self.connections)} MCP connections")
    
    async def shutdown(self):
        """Shutdown all MCP connections"""
        logger.info("Shutting down MCP connections...")
        
        # Cancel health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all connections
        for connection in self.connections.values():
            await connection.disconnect()
        
        self.connections.clear()
        logger.info("MCP connections shutdown complete")
    
    async def send_request(self, server_name: str, method: str, endpoint: str, data: Optional[Dict] = None) -> MCPResponse:
        """Send request to specific MCP server"""
        if server_name not in self.connections:
            return MCPResponse(success=False, error=f"Server {server_name} not found")
        
        connection = self.connections[server_name]
        
        # Retry logic
        for attempt in range(connection.config.retry_attempts):
            response = await connection.send_request(method, endpoint, data)
            
            if response.success:
                return response
            
            # If connection failed, try to reconnect
            if not connection.is_connected and attempt < connection.config.retry_attempts - 1:
                logger.info(f"Attempting to reconnect to {server_name} (attempt {attempt + 1})")
                await connection.connect()
        
        return response
    
    async def broadcast_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, MCPResponse]:
        """Send request to all connected MCP servers"""
        results = {}
        
        tasks = []
        for name, connection in self.connections.items():
            if connection.is_connected:
                task = asyncio.create_task(
                    self.send_request(name, method, endpoint, data),
                    name=name
                )
                tasks.append(task)
        
        # Wait for all requests to complete
        if tasks:
            completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
            
            for task, result in zip(tasks, completed_tasks):
                server_name = task.get_name()
                if isinstance(result, Exception):
                    results[server_name] = MCPResponse(success=False, error=str(result))
                else:
                    results[server_name] = result
        
        return results
    
    async def _health_check_loop(self):
        """Periodic health check for all connections"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                logger.debug("Performing MCP health checks...")
                
                for name, connection in self.connections.items():
                    if connection.is_connected:
                        health_response = await connection.health_check()
                        
                        if not health_response.success:
                            logger.warning(f"Health check failed for {name}: {health_response.error}")
                            connection.is_connected = False
                            
                            # Try to reconnect
                            logger.info(f"Attempting to reconnect to {name}")
                            await connection.connect()
                    else:
                        # Try to reconnect disconnected servers
                        logger.info(f"Attempting to reconnect to disconnected server: {name}")
                        await connection.connect()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in MCP health check loop: {e}")
    
    def get_server_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all servers"""
        return {name: conn.get_stats() for name, conn in self.connections.items()}
    
    def get_connected_servers(self) -> List[str]:
        """Get list of currently connected server names"""
        return [name for name, conn in self.connections.items() if conn.is_connected]
    
    def get_server_capabilities(self, server_name: str) -> List[str]:
        """Get capabilities of a specific server"""
        if server_name in self.servers:
            return self.servers[server_name].capabilities
        return []
    
    def find_servers_by_capability(self, capability: str) -> List[str]:
        """Find servers that have a specific capability"""
        matching_servers = []
        for name, config in self.servers.items():
            if capability in config.capabilities and name in self.connections and self.connections[name].is_connected:
                matching_servers.append(name)
        return matching_servers
    
    async def add_server(self, name: str, config: MCPServerConfig):
        """Add a new MCP server at runtime"""
        self.servers[name] = config
        
        if config.is_active:
            connection = MCPConnection(config)
            self.connections[name] = connection
            
            success = await connection.connect()
            if success:
                logger.info(f"Added and connected to new MCP server: {name}")
            else:
                logger.warning(f"Added MCP server {name} but failed to connect")
    
    async def remove_server(self, name: str):
        """Remove an MCP server"""
        if name in self.connections:
            await self.connections[name].disconnect()
            del self.connections[name]
        
        if name in self.servers:
            del self.servers[name]
        
        logger.info(f"Removed MCP server: {name}")


# Utility functions for common MCP operations

async def test_mcp_connection(config: MCPServerConfig) -> MCPResponse:
    """Test connection to a single MCP server"""
    connection = MCPConnection(config)
    
    try:
        success = await connection.connect()
        if success:
            health_response = await connection.health_check()
            await connection.disconnect()
            return health_response
        else:
            return MCPResponse(success=False, error="Failed to establish connection")
    except Exception as e:
        return MCPResponse(success=False, error=str(e))


async def discover_mcp_capabilities(config: MCPServerConfig) -> List[str]:
    """Discover capabilities of an MCP server"""
    connection = MCPConnection(config)
    capabilities = []
    
    try:
        if await connection.connect():
            # Try common capability endpoints
            endpoints_to_try = [
                "capabilities",
                "info",
                "status",
                "tools",
                "functions"
            ]
            
            for endpoint in endpoints_to_try:
                response = await connection.send_request("GET", endpoint)
                if response.success and response.data:
                    # Extract capabilities from response
                    if isinstance(response.data, dict):
                        if "capabilities" in response.data:
                            capabilities.extend(response.data["capabilities"])
                        if "tools" in response.data:
                            capabilities.extend(response.data["tools"])
            
            await connection.disconnect()
    
    except Exception as e:
        logger.error(f"Error discovering capabilities for {config.name}: {e}")
    
    return list(set(capabilities))  # Remove duplicates
