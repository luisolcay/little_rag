"""
Search Endpoints
================

Advanced search endpoints with hybrid search, semantic search, and keyword search.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException

# Import our search services
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.vector.azure_search.search_client import AzureSearchClient, HybridSearchRequest
from core.vector.retrieval_service import AdvancedRetrievalService
from core.vector.embeddings.azure_embedding_service import AzureEmbeddingService

# Import Pydantic models
from pydantic_models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchSuggestion
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/search", tags=["Search"])

# Global services (initialized on startup)
search_client = None
retrieval_service = None
embedding_service = None

@router.on_event("startup")
async def startup_event():
    """Initialize search services."""
    global search_client, retrieval_service, embedding_service
    
    try:
        # Initialize Azure AI Search client
        search_client = AzureSearchClient()
        logger.info("✅ Azure AI Search Client initialized")
        
        # Initialize embedding service
        embedding_service = AzureEmbeddingService()
        logger.info("✅ Azure OpenAI Embedding Service initialized")
        
        # Initialize advanced retrieval service
        retrieval_service = AdvancedRetrievalService(
            search_client=search_client,
            embedding_service=embedding_service,
            index_name="orbe-documents",  # Default index name
            top_k=5,
            rerank=True
        )
        logger.info("✅ Advanced Retrieval Service initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize search services: {e}")
        # Don't raise exception - services might be available later

@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """
    Perform hybrid search combining vector and keyword search.
    
    Args:
        request: Search request with query and parameters
        
    Returns:
        Search response with results
    """
    try:
        if not retrieval_service:
            raise HTTPException(
                status_code=503, 
                detail="Search service not available"
            )
        
        logger.info(f"🔍 Performing hybrid search: '{request.query[:50]}...'")
        
        start_time = datetime.now()
        
        # Generate query embedding if not provided
        if not request.query_vector:
            if embedding_service:
                embedding_results = await embedding_service.generate_embeddings([request.query])
                query_vector = embedding_results[0].embedding
            else:
                raise HTTPException(
                    status_code=503, 
                    detail="Embedding service not available for query vector generation"
                )
        else:
            query_vector = request.query_vector
        
        # Perform hybrid search
        search_results = await retrieval_service.hybrid_search(
            query=request.query,
            query_vector=query_vector,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Convert results to SearchResult models
        results = []
        for result in search_results:
            search_result = SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score,
                metadata=result.metadata,
                relevance_score=result.relevance_score,
                document_name=result.metadata.get('blob_name', 'Unknown'),
                page_number=result.metadata.get('page_number')
            )
            results.append(search_result)
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Hybrid search completed: {len(results)} results in {search_time:.2f}s")
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time=search_time,
            search_type="hybrid"
        )
        
    except Exception as e:
        logger.error(f"❌ Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search using vector similarity.
    
    Args:
        request: Search request with query and parameters
        
    Returns:
        Search response with results
    """
    try:
        if not search_client or not embedding_service:
            raise HTTPException(
                status_code=503, 
                detail="Search or embedding service not available"
            )
        
        logger.info(f"🧠 Performing semantic search: '{request.query[:50]}...'")
        
        start_time = datetime.now()
        
        # Generate query embedding
        embedding_results = await embedding_service.generate_embeddings([request.query])
        query_vector = embedding_results[0].embedding
        
        # Perform vector search
        vector_results = search_client.vector_search(
            query_vector=query_vector,
            top=request.top_k,
            filters=request.filters
        )
        
        # Convert results to SearchResult models
        results = []
        for result in vector_results:
            search_result = SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score,
                metadata=result.metadata,
                relevance_score=result.score,  # Use score as relevance for vector search
                document_name=result.metadata.get('blob_name', 'Unknown'),
                page_number=result.metadata.get('page_number')
            )
            results.append(search_result)
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Semantic search completed: {len(results)} results in {search_time:.2f}s")
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time=search_time,
            search_type="semantic"
        )
        
    except Exception as e:
        logger.error(f"❌ Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/keyword", response_model=SearchResponse)
async def keyword_search(request: SearchRequest):
    """
    Perform keyword search using BM25.
    
    Args:
        request: Search request with query and parameters
        
    Returns:
        Search response with results
    """
    try:
        if not search_client:
            raise HTTPException(
                status_code=503, 
                detail="Search service not available"
            )
        
        logger.info(f"🔤 Performing keyword search: '{request.query[:50]}...'")
        
        start_time = datetime.now()
        
        # Perform keyword search
        keyword_results = search_client.keyword_search(
            query_text=request.query,
            top=request.top_k,
            filters=request.filters
        )
        
        # Convert results to SearchResult models
        results = []
        for result in keyword_results:
            search_result = SearchResult(
                chunk_id=result.chunk_id,
                content=result.content,
                score=result.score,
                metadata=result.metadata,
                relevance_score=result.score,  # Use score as relevance for keyword search
                document_name=result.metadata.get('blob_name', 'Unknown'),
                page_number=result.metadata.get('page_number')
            )
            results.append(search_result)
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Keyword search completed: {len(results)} results in {search_time:.2f}s")
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time=search_time,
            search_type="keyword"
        )
        
    except Exception as e:
        logger.error(f"❌ Keyword search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/suggestions", response_model=List[SearchSuggestion])
async def get_search_suggestions(query: str = "", limit: int = 5):
    """
    Get search suggestions based on query.
    
    Args:
        query: Partial query string
        limit: Maximum number of suggestions
        
    Returns:
        List of search suggestions
    """
    try:
        # In a real implementation, you'd use AI to generate suggestions
        # For now, return placeholder suggestions
        
        suggestions = []
        
        if query:
            # Generate contextual suggestions based on query
            suggestions = [
                SearchSuggestion(
                    suggestion=f"{query} análisis ambiental",
                    confidence=0.8,
                    type="query"
                ),
                SearchSuggestion(
                    suggestion=f"{query} impacto ambiental",
                    confidence=0.7,
                    type="query"
                ),
                SearchSuggestion(
                    suggestion=f"{query} gestión ambiental",
                    confidence=0.6,
                    type="query"
                )
            ]
        else:
            # Default suggestions
            suggestions = [
                SearchSuggestion(
                    suggestion="análisis ambiental",
                    confidence=0.9,
                    type="topic"
                ),
                SearchSuggestion(
                    suggestion="impacto ambiental",
                    confidence=0.8,
                    type="topic"
                ),
                SearchSuggestion(
                    suggestion="gestión ambiental",
                    confidence=0.7,
                    type="topic"
                ),
                SearchSuggestion(
                    suggestion="monitoreo ambiental",
                    confidence=0.6,
                    type="topic"
                ),
                SearchSuggestion(
                    suggestion="compliance ambiental",
                    confidence=0.5,
                    type="topic"
                )
            ]
        
        return suggestions[:limit]
        
    except Exception as e:
        logger.error(f"❌ Failed to get search suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_search_status():
    """
    Get the status of search services.
    
    Returns:
        Search service status information
    """
    try:
        status = {
            "search_client_available": search_client is not None,
            "retrieval_service_available": retrieval_service is not None,
            "embedding_service_available": embedding_service is not None,
            "services_initialized": all([search_client, retrieval_service, embedding_service]),
            "timestamp": datetime.now().isoformat()
        }
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Failed to get search status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
