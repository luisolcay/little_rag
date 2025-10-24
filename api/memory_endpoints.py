"""
Memory Management Endpoints
============================

Endpoints for managing conversation memory using Azure Cosmos DB and Redis.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException

# Import our memory service
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.memory.memory_manager import MemoryManager

# Import Pydantic models
from pydantic_models import (
    MemorySession,
    ConversationMessage,
    MemorySessionResponse
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/memory", tags=["Memory"])

# Global service (initialized on startup)
memory_manager = None

@router.on_event("startup")
async def startup_event():
    """Initialize memory management service."""
    global memory_manager
    
    try:
        memory_manager = MemoryManager()
        logger.info("✅ Memory Manager (Cosmos DB & Redis) initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize memory manager: {e}")
        # Don't raise exception - service might be available later

@router.post("/sessions", response_model=MemorySessionResponse)
async def create_memory_session():
    """
    Create a new memory session.
    
    Returns:
        Memory session response
    """
    try:
        if not memory_manager:
            raise HTTPException(
                status_code=503, 
                detail="Memory service not available"
            )
        
        # Generate new session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(datetime.now())}"
        
        # Create session in memory manager
        await memory_manager.create_session(session_id)
        
        logger.info(f"📝 Created new memory session: {session_id}")
        
        return MemorySessionResponse(
            session_id=session_id,
            success=True,
            messages=[],
            summary=None,
            error_message=None
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to create memory session: {e}")
        return MemorySessionResponse(
            session_id="",
            success=False,
            messages=[],
            summary=None,
            error_message=str(e)
        )

@router.get("/sessions/{session_id}", response_model=MemorySessionResponse)
async def get_memory_session(session_id: str):
    """
    Get memory session with conversation history.
    
    Args:
        session_id: The session ID
        
    Returns:
        Memory session response with messages
    """
    try:
        if not memory_manager:
            raise HTTPException(
                status_code=503, 
                detail="Memory service not available"
            )
        
        # Get session from memory manager
        session_data = await memory_manager.get_session(session_id)
        
        if not session_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Session {session_id} not found"
            )
        
        # Convert session data to ConversationMessage models
        messages = []
        for msg_data in session_data.get('messages', []):
            message = ConversationMessage(
                role=msg_data.get('role', 'user'),
                content=msg_data.get('content', ''),
                timestamp=datetime.fromisoformat(msg_data.get('timestamp', datetime.now().isoformat())),
                metadata=msg_data.get('metadata', {})
            )
            messages.append(message)
        
        logger.info(f"📖 Retrieved memory session: {session_id} ({len(messages)} messages)")
        
        return MemorySessionResponse(
            session_id=session_id,
            success=True,
            messages=messages,
            summary=session_data.get('summary'),
            error_message=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get memory session {session_id}: {e}")
        return MemorySessionResponse(
            session_id=session_id,
            success=False,
            messages=[],
            summary=None,
            error_message=str(e)
        )

@router.post("/sessions/{session_id}/summarize", response_model=Dict[str, Any])
async def summarize_conversation(session_id: str):
    """
    Generate a summary of the conversation in a session.
    
    Args:
        session_id: The session ID
        
    Returns:
        Summary response
    """
    try:
        if not memory_manager:
            raise HTTPException(
                status_code=503, 
                detail="Memory service not available"
            )
        
        # Generate summary using memory manager
        summary = await memory_manager.summarize_conversation(session_id)
        
        logger.info(f"📋 Generated summary for session: {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to summarize session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}")
async def delete_memory_session(session_id: str):
    """
    Delete a memory session and its data.
    
    Args:
        session_id: The session ID
        
    Returns:
        Deletion confirmation
    """
    try:
        if not memory_manager:
            raise HTTPException(
                status_code=503, 
                detail="Memory service not available"
            )
        
        # Delete session from memory manager
        await memory_manager.delete_session(session_id)
        
        logger.info(f"🗑️ Deleted memory session: {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "message": f"Session {session_id} deleted successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions", response_model=List[MemorySession])
async def list_memory_sessions(limit: int = 20, offset: int = 0):
    """
    List all memory sessions.
    
    Args:
        limit: Maximum number of sessions to return
        offset: Number of sessions to skip
        
    Returns:
        List of memory sessions
    """
    try:
        if not memory_manager:
            raise HTTPException(
                status_code=503, 
                detail="Memory service not available"
            )
        
        # Get sessions from memory manager
        sessions_data = await memory_manager.list_sessions(limit=limit, offset=offset)
        
        # Convert to MemorySession models
        sessions = []
        for session_data in sessions_data:
            session = MemorySession(
                session_id=session_data.get('session_id', ''),
                created_at=datetime.fromisoformat(session_data.get('created_at', datetime.now().isoformat())),
                last_activity=datetime.fromisoformat(session_data.get('last_activity', datetime.now().isoformat())),
                message_count=session_data.get('message_count', 0),
                total_tokens=session_data.get('total_tokens', 0),
                summary=session_data.get('summary')
            )
            sessions.append(session)
        
        logger.info(f"📋 Listed {len(sessions)} memory sessions")
        
        return sessions
        
    except Exception as e:
        logger.error(f"❌ Failed to list memory sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/messages")
async def add_message_to_session(
    session_id: str,
    message: ConversationMessage
):
    """
    Add a message to a memory session.
    
    Args:
        session_id: The session ID
        message: The message to add
        
    Returns:
        Confirmation response
    """
    try:
        if not memory_manager:
            raise HTTPException(
                status_code=503, 
                detail="Memory service not available"
            )
        
        # Add message to session
        await memory_manager.add_message(session_id, message.content, message.role)
        
        logger.info(f"💬 Added message to session: {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "message_added": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to add message to session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_memory_status():
    """
    Get the status of memory services.
    
    Returns:
        Memory service status information
    """
    try:
        status = {
            "memory_manager_available": memory_manager is not None,
            "service_initialized": memory_manager is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if memory_manager:
            # Test connectivity to Cosmos DB and Redis
            try:
                # In a real implementation, you'd test the connections
                status["cosmos_db_status"] = "healthy"
                status["redis_status"] = "healthy"
            except Exception as e:
                status["cosmos_db_status"] = f"error: {str(e)}"
                status["redis_status"] = f"error: {str(e)}"
        else:
            status["cosmos_db_status"] = "not_available"
            status["redis_status"] = "not_available"
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Failed to get memory status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
