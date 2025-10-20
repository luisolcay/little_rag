"""
Custom Cosmos DB NoSQL API Chat Message History for LangChain compatibility.
"""

import os
import uuid
from datetime import datetime
from typing import List, Optional
import logging

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, messages_from_dict, messages_to_dict

logger = logging.getLogger(__name__)

class CosmosDBNoSQLChatMessageHistory(BaseChatMessageHistory):
    """Custom chat message history using Azure Cosmos DB NoSQL API."""
    
    def __init__(self, cosmos_client, session_id: str, user_id: str = "default"):
        """
        Initialize Cosmos DB NoSQL chat message history.
        
        Args:
            cosmos_client: Initialized CosmosDBClient instance
            session_id: Unique session identifier
            user_id: User identifier (default: "default")
        """
        self.cosmos_client = cosmos_client
        self.session_id = session_id
        self.user_id = user_id
        self.container_name = "conversations"
        
        # Ensure Cosmos DB is initialized
        if not hasattr(self.cosmos_client, '_initialized') or not self.cosmos_client._initialized:
            logger.info(f"[COSMOS_CHAT] Initializing Cosmos DB for session {session_id}")
            self.cosmos_client.initialize()
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages for this session."""
        try:
            # Query messages for this session ordered by timestamp
            query = """
            SELECT c.id, c.type, c.content, c.timestamp 
            FROM c 
            WHERE c.session_id = @session_id 
            ORDER BY c.timestamp ASC
            """
            
            parameters = [{"name": "@session_id", "value": self.session_id}]
            
            items = self.cosmos_client.query_items(
                container_name=self.container_name,
                query=query,
                parameters=parameters
            )
            
            # Convert Cosmos DB items to LangChain messages
            messages = []
            for item in items:
                try:
                    if item["type"] == "human":
                        message = HumanMessage(content=item["content"])
                    elif item["type"] == "ai":
                        message = AIMessage(content=item["content"])
                    else:
                        logger.warning(f"Unknown message type: {item['type']}")
                        continue
                    
                    messages.append(message)
                    
                except Exception as e:
                    logger.error(f"Error converting message {item.get('id', 'unknown')}: {e}")
                    continue
            
            logger.debug(f"[COSMOS_CHAT] Retrieved {len(messages)} messages for session {self.session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving messages for session {self.session_id}: {e}")
            return []
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the conversation history."""
        try:
            # Determine message type
            if isinstance(message, HumanMessage):
                message_type = "human"
            elif isinstance(message, AIMessage):
                message_type = "ai"
            else:
                logger.warning(f"Unsupported message type: {type(message)}")
                return
            
            # Create document for Cosmos DB
            document = {
                "id": str(uuid.uuid4()),
                "session_id": self.session_id,
                "user_id": self.user_id,
                "type": message_type,
                "content": message.content,
                "timestamp": datetime.utcnow().isoformat(),
                "partition_key": self.session_id  # Use session_id as partition key
            }
            
            # Store in Cosmos DB
            self.cosmos_client.create_item(
                container_name=self.container_name,
                item=document
            )
            
            logger.debug(f"[COSMOS_CHAT] Added {message_type} message to session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error adding message to session {self.session_id}: {e}")
            raise
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation history."""
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the conversation history."""
        self.add_message(AIMessage(content=message))
    
    def clear(self) -> None:
        """Clear all messages for this session."""
        try:
            # Query all message IDs for this session
            query = "SELECT c.id FROM c WHERE c.session_id = @session_id"
            parameters = [{"name": "@session_id", "value": self.session_id}]
            
            items = self.cosmos_client.query_items(
                container_name=self.container_name,
                query=query,
                parameters=parameters
            )
            
            # Delete each message
            deleted_count = 0
            for item in items:
                try:
                    success = self.cosmos_client.delete_item(
                        container_name=self.container_name,
                        item_id=item["id"],
                        partition_key=self.session_id
                    )
                    if success:
                        deleted_count += 1
                        
                except Exception as e:
                    logger.error(f"Error deleting message {item['id']}: {e}")
                    continue
            
            logger.info(f"[COSMOS_CHAT] Cleared {deleted_count} messages for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error clearing messages for session {self.session_id}: {e}")
            raise
    
    def get_message_count(self) -> int:
        """Get the number of messages for this session."""
        try:
            query = "SELECT VALUE COUNT(1) FROM c WHERE c.session_id = @session_id"
            parameters = [{"name": "@session_id", "value": self.session_id}]
            
            result = self.cosmos_client.query_items(
                container_name=self.container_name,
                query=query,
                parameters=parameters
            )
            
            # Query should return a single count value
            count = next(iter(result), 0)
            return count
            
        except Exception as e:
            logger.error(f"Error getting message count for session {self.session_id}: {e}")
            return 0
    
    def get_session_info(self) -> dict:
        """Get information about this session."""
        try:
            message_count = self.get_message_count()
            
            # Get first and last message timestamps
            query = """
            SELECT c.timestamp 
            FROM c 
            WHERE c.session_id = @session_id 
            ORDER BY c.timestamp ASC
            """
            parameters = [{"name": "@session_id", "value": self.session_id}]
            
            timestamps = list(self.cosmos_client.query_items(
                container_name=self.container_name,
                query=query,
                parameters=parameters
            ))
            
            info = {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "message_count": message_count,
                "first_message": timestamps[0]["timestamp"] if timestamps else None,
                "last_message": timestamps[-1]["timestamp"] if timestamps else None,
                "container": self.container_name
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting session info for {self.session_id}: {e}")
            return {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "message_count": 0,
                "error": str(e)
            }
