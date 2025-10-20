"""
Azure Cosmos DB client configuration and connection management.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError, CosmosResourceExistsError
from azure.core.exceptions import AzureError

logger = logging.getLogger(__name__)

class CosmosDBClient:
    """Azure Cosmos DB client with connection pooling and error handling."""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_COSMOS_DB_ENDPOINT")
        self.primary_key = os.getenv("AZURE_COSMOS_DB_PRIMARY_KEY")
        self.secondary_key = os.getenv("AZURE_COSMOS_DB_SECONDARY_KEY")
        self.database_name = os.getenv("AZURE_COSMOS_DB_DATABASE_NAME", "collahuasi-rag")
        
        if not all([self.endpoint, self.primary_key]):
            raise ValueError("Azure Cosmos DB credentials not configured")
        
        self.client = None
        self.database = None
        self.containers = {}
        self._initialized = False
        
    def initialize(self):
        """Initialize Cosmos DB client and create containers."""
        if self._initialized:
            logger.debug("Cosmos DB client already initialized")
            return
            
        try:
            # Initialize client with retry policy
            self.client = CosmosClient(
                url=self.endpoint,
                credential=self.primary_key,
                consistency_level="Session"
            )
            
            # Create database
            self._create_database()
            
            # Create containers
            self._create_containers()
            
            self._initialized = True
            logger.info("Cosmos DB client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB client: {e}")
            raise
    
    def _create_database(self):
        """Create database if it doesn't exist."""
        try:
            # For serverless accounts, don't specify offer_throughput
            self.database = self.client.create_database_if_not_exists(
                id=self.database_name
            )
            logger.info(f"Database '{self.database_name}' ready")
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise
    
    def _create_containers(self):
        """Create required containers."""
        containers_config = [
            {
                "id": "conversations",
                "partition_key": PartitionKey(path="/session_id"),
                "default_ttl": 2592000  # 30 days
            },
            {
                "id": "prompt_metrics",
                "partition_key": PartitionKey(path="/prompt_version")
            },
            {
                "id": "llm_analytics",
                "partition_key": PartitionKey(path="/date")
            }
        ]
        
        for config in containers_config:
            try:
                # For serverless accounts, don't specify default_ttl in create_container_if_not_exists
                container = self.database.create_container_if_not_exists(
                    id=config["id"],
                    partition_key=config["partition_key"]
                )
                
                # Set TTL separately if specified
                if config.get("default_ttl"):
                    try:
                        container.replace(
                            id=config["id"],
                            partition_key=config["partition_key"],
                            default_ttl=config["default_ttl"]
                        )
                    except Exception as ttl_error:
                        logger.warning(f"Could not set TTL for container '{config['id']}': {ttl_error}")
                
                self.containers[config["id"]] = container
                logger.info(f"Container '{config['id']}' ready")
            except Exception as e:
                logger.error(f"Failed to create container '{config['id']}': {e}")
                raise
    
    def get_container(self, container_name: str):
        """Get container by name."""
        if container_name not in self.containers:
            raise ValueError(f"Container '{container_name}' not found")
        return self.containers[container_name]
    
    def create_item(self, container_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Create item in container."""
        try:
            container = self.get_container(container_name)
            result = container.create_item(body=item)
            return result
        except Exception as e:
            logger.error(f"Failed to create item in '{container_name}': {e}")
            raise
    
    def read_item(self, container_name: str, item_id: str, partition_key: str) -> Dict[str, Any]:
        """Read item from container."""
        try:
            container = self.get_container(container_name)
            result = container.read_item(item=item_id, partition_key=partition_key)
            return result
        except CosmosResourceNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Failed to read item from '{container_name}': {e}")
            raise
    
    def upsert_item(self, container_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert item in container."""
        try:
            container = self.get_container(container_name)
            result = container.upsert_item(body=item)
            return result
        except Exception as e:
            logger.error(f"Failed to upsert item in '{container_name}': {e}")
            raise
    
    def query_items(self, container_name: str, query: str, parameters: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Query items from container."""
        try:
            container = self.get_container(container_name)
            items = list(container.query_items(
                query=query,
                parameters=parameters or [],
                enable_cross_partition_query=True
            ))
            return items
        except Exception as e:
            logger.error(f"Failed to query items from '{container_name}': {e}")
            raise
    
    def delete_item(self, container_name: str, item_id: str, partition_key: str) -> bool:
        """Delete item from container."""
        try:
            container = self.get_container(container_name)
            container.delete_item(item=item_id, partition_key=partition_key)
            return True
        except CosmosResourceNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Failed to delete item from '{container_name}': {e}")
            raise

# Global instance
cosmos_client = CosmosDBClient()

# Alias for compatibility
cosmos_store = cosmos_client