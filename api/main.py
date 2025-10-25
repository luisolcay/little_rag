"""
FastAPI application with Azure AI Search integration.
"""

import os
import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import shutil
import uuid

# Import document processing components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import our new vector search components
from core.vector.retrieval_service import AdvancedRetrievalService, RetrievalRequest
from core.vector.indexing.pipeline import IndexingPipeline
from core.vector.azure_search.index_manager import IndexManager

# Import new LLM layer components
from core.llm.orchestrator import llm_orchestrator
from core.memory.memory_manager import memory_manager
from core.retrieval.strategy_router import query_classifier

# Import existing components
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest, DocumentMetadata

# Import new endpoints
from llm_endpoints import router as llm_router
from admin_endpoints import router as admin_router
from document_processing_endpoints import router as document_processing_router
from embedding_endpoints import router as embedding_router
from search_endpoints import router as search_router
from memory_endpoints import router as memory_router

# Import Azure services manager
from azure_services_manager import azure_services_manager

app = FastAPI(
    title="Collahuasi RAG API",
    description="Advanced RAG system with Azure AI Search",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for React frontend (must be before routers)
# Check if frontend build directory exists
frontend_build_path = Path(__file__).parent.parent / "frontend-react" / "build"
if frontend_build_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_build_path / "static")), name="static")
    # Serve React app for any route that doesn't match API endpoints
    app.mount("/", StaticFiles(directory=str(frontend_build_path), html=True), name="frontend")

# Include all routers
app.include_router(llm_router)
app.include_router(admin_router)
app.include_router(document_processing_router)
app.include_router(embedding_router)
app.include_router(search_router)
app.include_router(memory_router)

@app.on_event("startup")
async def startup_event():
    """Initialize all services on startup."""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Collahuasi RAG API...")
    
    try:
        # Initialize Azure services
        init_results = await azure_services_manager.initialize_all_services()
        
        logger.info("✅ API startup completed successfully")
        logger.info(f"   Services initialized: {len(init_results['successful'])}")
        logger.info(f"   Services failed: {len(init_results['failed'])}")
        logger.info(f"   Services with warnings: {len(init_results['warnings'])}")
        
    except Exception as e:
        logger.error(f"❌ API startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger = logging.getLogger(__name__)
    logger.info("🛑 Shutting down Collahuasi RAG API...")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Collahuasi RAG API",
        "version": "2.0.0",
        "description": "Advanced RAG system with Azure AI Search",
        "endpoints": {
            "document_processing": "/documents",
            "embeddings": "/embeddings", 
            "search": "/search",
            "memory": "/memory",
            "llm": "/llm",
            "admin": "/admin"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check for all services."""
    try:
        # Get Azure services health
        azure_health = await azure_services_manager.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "azure_services": azure_health,
            "api_version": "2.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "api_version": "2.0.0"
        }

@app.get("/status")
async def get_status():
    """Get detailed status of all services."""
    try:
        service_status = azure_services_manager.get_service_status()
        configuration = azure_services_manager.get_configuration()
        
        return {
            "api_status": "running",
            "timestamp": datetime.now().isoformat(),
            "services": service_status,
            "configuration": configuration
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/initialize-index")
async def initialize_search_index():
    """Initialize the Azure AI Search index."""
    try:
        result = await azure_services_manager.create_search_index()
        
        if result["success"]:
            return {
                "message": "Search index initialized successfully",
                "index_created": result["index_created"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize search index: {result['error']}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints for backward compatibility
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(query: QueryInput):
    """Legacy chat endpoint for backward compatibility."""
    try:
        # Use the LLM orchestrator for processing
        response = await llm_orchestrator.generate_response(
            user_query=query.question,
            session_id=query.session_id,
            structured_output_schema=None
        )
        
        return QueryResponse(
            answer=response.response_text,
            session_id=query.session_id or "default",
            model=query.model,
            citations=response.citations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all processed documents."""
    try:
        # In a real implementation, you'd query a database
        return {
            "documents": [],
            "total_count": 0,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)