"""
Admin endpoints for prompt version management, metrics, and memory statistics.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.llm.prompt_manager import prompt_manager
from core.llm.orchestrator import llm_orchestrator
from core.memory.memory_manager import memory_manager
from core.memory.cosmos_store import cosmos_client
from core.memory.redis_cache import redis_client

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/admin", tags=["Admin"])

# Request/Response models
class PromptVersionRequest(BaseModel):
    """Request to activate a prompt version."""
    prompt_type: str = Field(..., description="Type of prompt (e.g., environmental_consultant)")
    version: str = Field(..., description="Version to activate (e.g., v1.1.0)")

class PromptMetricsRequest(BaseModel):
    """Request to track prompt metrics."""
    prompt_type: str
    version: str
    metrics: Dict[str, Any]

# Prompt management endpoints
@router.get("/prompts/versions")
async def list_prompt_versions(prompt_type: Optional[str] = None):
    """List all prompt versions."""
    try:
        if prompt_type:
            versions = prompt_manager.list_versions(prompt_type)
            return {
                "prompt_type": prompt_type,
                "versions": versions
            }
        else:
            # List all prompt types and their versions
            all_versions = {}
            prompt_types = ["environmental_consultant", "contextualization"]
            
            for pt in prompt_types:
                all_versions[pt] = prompt_manager.list_versions(pt)
            
            return {
                "prompt_types": all_versions,
                "active_versions": prompt_manager.active_versions
            }
            
    except Exception as e:
        logger.error(f"Failed to list prompt versions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list prompt versions: {str(e)}")

@router.post("/prompts/activate")
async def activate_prompt_version(request: PromptVersionRequest):
    """Activate a specific prompt version."""
    try:
        success = prompt_manager.activate_version(request.prompt_type, request.version)
        
        if success:
            return {
                "message": f"Activated version {request.version} for {request.prompt_type}",
                "active_versions": prompt_manager.active_versions
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to activate version {request.version} for {request.prompt_type}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate prompt version: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to activate prompt version: {str(e)}")

@router.get("/prompts/performance")
async def get_prompt_performance(
    prompt_type: str = Query(..., description="Prompt type"),
    version: str = Query(..., description="Prompt version"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """Get prompt performance metrics."""
    try:
        performance = prompt_manager.get_prompt_performance(prompt_type, version, days)
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get prompt performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prompt performance: {str(e)}")

@router.post("/prompts/metrics")
async def track_prompt_metrics(request: PromptMetricsRequest):
    """Track prompt performance metrics."""
    try:
        await prompt_manager.track_prompt_metrics(
            request.prompt_type,
            request.version,
            request.metrics
        )
        
        return {"message": "Metrics tracked successfully"}
        
    except Exception as e:
        logger.error(f"Failed to track prompt metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to track prompt metrics: {str(e)}")

# Memory management endpoints
@router.get("/memory/stats")
async def get_memory_statistics():
    """Get memory usage statistics."""
    try:
        # Get Redis stats
        redis_stats = await redis_client.get_stats()
        
        # Get Cosmos DB stats (simplified)
        cosmos_stats = {
            "database": cosmos_client.database_name,
            "containers": list(cosmos_client.containers.keys()),
            "status": "connected" if cosmos_client.client else "disconnected"
        }
        
        # Get memory manager stats
        memory_stats = await memory_manager.get_memory_stats()
        
        return {
            "redis": redis_stats,
            "cosmos_db": cosmos_stats,
            "memory_manager": memory_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get memory statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory statistics: {str(e)}")

@router.get("/memory/conversations")
async def list_conversations(
    limit: int = Query(50, ge=1, le=1000, description="Number of conversations to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """List recent conversations."""
    try:
        # Query Cosmos DB for recent conversations
        query = """
        SELECT TOP @limit c.session_id, c.created_at, c.updated_at, c.summary
        FROM c
        ORDER BY c.updated_at DESC
        OFFSET @offset LIMIT @limit
        """
        
        parameters = [
            {"name": "@limit", "value": limit},
            {"name": "@offset", "value": offset}
        ]
        
        conversations = await cosmos_client.query_items(
            "conversations",
            query,
            parameters
        )
        
        return {
            "conversations": conversations,
            "limit": limit,
            "offset": offset,
            "total_returned": len(conversations)
        }
        
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")

@router.delete("/memory/conversations/{session_id}")
async def delete_conversation_admin(session_id: str):
    """Delete conversation (admin endpoint)."""
    try:
        success = await memory_manager.clear_conversation(session_id)
        
        if success:
            return {"message": f"Conversation {session_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")

# LLM metrics and costs
@router.get("/llm/costs")
async def get_llm_costs(
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    model: Optional[str] = Query(None, description="Filter by model")
):
    """Get LLM usage costs and metrics."""
    try:
        # Get current metrics from orchestrator
        current_metrics = llm_orchestrator.get_metrics()
        
        # Query historical metrics from Cosmos DB
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        query = """
        SELECT c.model, c.tokens_used, c.cost_usd, c.success, c.timestamp
        FROM c
        WHERE c.timestamp >= @start_date AND c.timestamp <= @end_date
        """
        
        parameters = [
            {"name": "@start_date", "value": start_date.isoformat()},
            {"name": "@end_date", "value": end_date.isoformat()}
        ]
        
        if model:
            query += " AND c.model = @model"
            parameters.append({"name": "@model", "value": model})
        
        historical_metrics = await cosmos_client.query_items(
            "llm_analytics",
            query,
            parameters
        )
        
        # Calculate aggregated metrics
        total_cost = sum(metric.get("cost_usd", 0) for metric in historical_metrics)
        total_tokens = sum(metric.get("tokens_used", 0) for metric in historical_metrics)
        total_requests = len(historical_metrics)
        success_rate = sum(1 for metric in historical_metrics if metric.get("success", False)) / max(total_requests, 1)
        
        # Group by model
        model_stats = {}
        for metric in historical_metrics:
            model_name = metric.get("model", "unknown")
            if model_name not in model_stats:
                model_stats[model_name] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                    "success_rate": 0.0
                }
            
            model_stats[model_name]["requests"] += 1
            model_stats[model_name]["tokens"] += metric.get("tokens_used", 0)
            model_stats[model_name]["cost"] += metric.get("cost_usd", 0)
        
        # Calculate success rates per model
        for model_name in model_stats:
            model_metrics = [m for m in historical_metrics if m.get("model") == model_name]
            successful = sum(1 for m in model_metrics if m.get("success", False))
            model_stats[model_name]["success_rate"] = successful / max(len(model_metrics), 1)
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "summary": {
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 6),
                "average_cost_per_request": round(total_cost / max(total_requests, 1), 6),
                "success_rate": round(success_rate, 3)
            },
            "by_model": model_stats,
            "current_session": current_metrics
        }
        
    except Exception as e:
        logger.error(f"Failed to get LLM costs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get LLM costs: {str(e)}")

@router.get("/llm/metrics")
async def get_llm_metrics():
    """Get current LLM performance metrics."""
    try:
        metrics = llm_orchestrator.get_metrics()
        
        return {
            "current_metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "service_status": {
                "llm_orchestrator": llm_orchestrator is not None,
                "memory_manager": memory_manager is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get LLM metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get LLM metrics: {str(e)}")

@router.post("/llm/reset-metrics")
async def reset_llm_metrics():
    """Reset LLM performance metrics."""
    try:
        llm_orchestrator.reset_metrics()
        
        return {"message": "LLM metrics reset successfully"}
        
    except Exception as e:
        logger.error(f"Failed to reset LLM metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset LLM metrics: {str(e)}")

# System health and diagnostics
@router.get("/system/health")
async def system_health_check():
    """Comprehensive system health check."""
    try:
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "services": {}
        }
        
        # Check LLM orchestrator
        try:
            llm_metrics = llm_orchestrator.get_metrics()
            health_status["services"]["llm_orchestrator"] = {
                "status": "healthy",
                "metrics": llm_metrics
            }
        except Exception as e:
            health_status["services"]["llm_orchestrator"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Check memory manager
        try:
            memory_stats = await memory_manager.get_memory_stats()
            health_status["services"]["memory_manager"] = {
                "status": "healthy",
                "stats": memory_stats
            }
        except Exception as e:
            health_status["services"]["memory_manager"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Check Redis
        try:
            redis_stats = await redis_client.get_stats()
            health_status["services"]["redis"] = {
                "status": "healthy",
                "stats": redis_stats
            }
        except Exception as e:
            health_status["services"]["redis"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Check Cosmos DB
        try:
            cosmos_status = {
                "status": "healthy" if cosmos_client.client else "unhealthy",
                "database": cosmos_client.database_name,
                "containers": list(cosmos_client.containers.keys())
            }
            health_status["services"]["cosmos_db"] = cosmos_status
            
            if not cosmos_client.client:
                health_status["overall_status"] = "degraded"
                
        except Exception as e:
            health_status["services"]["cosmos_db"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unhealthy",
            "error": str(e)
        }

