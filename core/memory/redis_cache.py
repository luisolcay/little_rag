"""
Azure Redis Cache client with connection pooling and automatic reconnection.
"""

import os
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from redis.exceptions import RedisError, ConnectionError, TimeoutError

logger = logging.getLogger(__name__)

class RedisCacheClient:
    """Azure Redis Cache client with connection pooling and error handling."""
    
    def __init__(self):
        self.host = os.getenv("AZURE_REDIS_HOST")
        self.port = int(os.getenv("AZURE_REDIS_PORT", "6380"))
        self.primary_key = os.getenv("AZURE_REDIS_PRIMARY_KEY")
        self.secondary_key = os.getenv("AZURE_REDIS_SECONDARY_KEY")
        self.ssl = os.getenv("AZURE_REDIS_SSL", "true").lower() == "true"
        self.session_ttl = int(os.getenv("REDIS_SESSION_TTL", "86400"))  # 24 hours
        self.username = os.getenv("AZURE_REDIS_USERNAME", "default")
        
        # Azure Redis can work without primary key for testing
        # Will fail gracefully if not configured
        
        self.pool = None
        self.redis = None
        self._connection_retries = 0
        self._max_retries = 3
        
    async def initialize(self):
        """Initialize Redis connection pool."""
        try:
            # For Azure Redis with SSL, pass SSL parameters to Redis client
            if self.ssl:
                self.redis = redis.Redis(
                    host=self.host,
                    port=self.port,
                    password=self.primary_key,
                    ssl=True,
                    ssl_cert_reqs=None,  # Azure uses self-signed certs
                    decode_responses=True,
                    retry_on_timeout=True,
                    socket_keepalive=True,
                    health_check_interval=30
                )
            else:
                # Connection pool for non-SSL
                connection_kwargs = {
                    "host": self.host,
                    "port": self.port,
                    "password": self.primary_key,
                    "decode_responses": True,
                    "max_connections": 20,
                    "retry_on_timeout": True,
                    "socket_keepalive": True,
                    "health_check_interval": 30
                }
                self.pool = ConnectionPool(**connection_kwargs)
                self.redis = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.redis.ping()
            
            logger.info("Redis client initialized successfully")
            self._connection_retries = 0
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self._connection_retries += 1
            if self._connection_retries < self._max_retries:
                await asyncio.sleep(2 ** self._connection_retries)  # Exponential backoff
                await self.initialize()
            else:
                raise
    
    async def _ensure_connection(self):
        """Ensure Redis connection is active."""
        try:
            await self.redis.ping()
        except (ConnectionError, TimeoutError):
            logger.warning("Redis connection lost, reconnecting...")
            await self.initialize()
    
    async def set_session(self, session_id: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store session data in Redis."""
        try:
            await self._ensure_connection()
            
            key = f"session:{session_id}"
            value = json.dumps({
                "data": data,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            
            ttl = ttl or self.session_ttl
            await self.redis.setex(key, ttl, value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set session '{session_id}': {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data from Redis."""
        try:
            await self._ensure_connection()
            
            key = f"session:{session_id}"
            value = await self.redis.get(key)
            
            if not value:
                return None
            
            data = json.loads(value)
            return data.get("data")
            
        except Exception as e:
            logger.error(f"Failed to get session '{session_id}': {e}")
            return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Update session data in Redis."""
        try:
            await self._ensure_connection()
            
            key = f"session:{session_id}"
            existing = await self.redis.get(key)
            
            if existing:
                existing_data = json.loads(existing)
                existing_data["data"].update(data)
                existing_data["updated_at"] = datetime.now().isoformat()
                value = json.dumps(existing_data)
            else:
                value = json.dumps({
                    "data": data,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                })
            
            ttl = ttl or self.session_ttl
            await self.redis.setex(key, ttl, value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session '{session_id}': {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session from Redis."""
        try:
            await self._ensure_connection()
            
            key = f"session:{session_id}"
            result = await self.redis.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete session '{session_id}': {e}")
            return False
    
    async def set_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set cache value with TTL."""
        try:
            await self._ensure_connection()
            
            cache_key = f"cache:{key}"
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            
            await self.redis.setex(cache_key, ttl, serialized_value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache '{key}': {e}")
            return False
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """Get cache value."""
        try:
            await self._ensure_connection()
            
            cache_key = f"cache:{key}"
            value = await self.redis.get(cache_key)
            
            if not value:
                return None
            
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
                
        except Exception as e:
            logger.error(f"Failed to get cache '{key}': {e}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Delete cache value."""
        try:
            await self._ensure_connection()
            
            cache_key = f"cache:{key}"
            result = await self.redis.delete(cache_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete cache '{key}': {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics."""
        try:
            await self._ensure_connection()
            
            info = await self.redis.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {}
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
        if self.pool:
            await self.pool.disconnect()

# Global instance
redis_client = RedisCacheClient()

# Alias for compatibility
redis_cache = redis_client

