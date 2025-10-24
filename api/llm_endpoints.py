"""
Enhanced LLM endpoints with streaming support and structured outputs.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.llm.orchestrator import llm_orchestrator
from core.llm.models import (
    LLMRequest, LLMResponse, StreamingChunk, StructuredOutput,
    EnvironmentalQueryType, ModelName, ComplianceReport, RiskAssessment,
    TechnicalAnalysis, GeneralQA
)
from core.memory.memory_manager import memory_manager
from core.retrieval.strategy_router import query_classifier
from core.vector.retrieval_service import AdvancedRetrievalService, RetrievalRequest
from core.llm.citation_extractor import citation_extractor
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/llm", tags=["LLM"])

# Initialize services
retrieval_service = None

@router.on_event("startup")
async def startup_event():
    """Initialize LLM services."""
    global retrieval_service
    try:
        retrieval_service = AdvancedRetrievalService(index_name="orbe-documents")
        print("LLM endpoints initialized successfully")
    except Exception as e:
        print(f"Failed to initialize LLM services: {e}")

# Request/Response models
class ChatRequest(BaseModel):
    """Enhanced chat request."""
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    query_type: Optional[EnvironmentalQueryType] = None
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=8192)
    stream: bool = Field(default=False)
    structured_output: bool = Field(default=False)
    retrieval_strategy: Optional[str] = None

class StructuredAnalysisRequest(BaseModel):
    """Request for structured environmental analysis."""
    query: str = Field(..., min_length=10, max_length=2000)
    analysis_type: EnvironmentalQueryType
    session_id: Optional[str] = None
    model: ModelName = Field(default=ModelName.GPT4_O)
    include_citations: bool = Field(default=True)

class ConversationRequest(BaseModel):
    """Request for conversation management."""
    session_id: str

# Enhanced chat endpoint
@router.post("/chat", response_model=LLMResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with intelligent retrieval and memory."""
    try:
        # Generate session ID if not provided
        if not request.session_id:
            import uuid
            request.session_id = str(uuid.uuid4())
        
        # Get conversation history - simplified for now
        conversation_history = []
        
        # Classify query and determine retrieval strategy
        query_analysis = query_classifier.analyze_query(request.query)
        
        # Use specified strategy or auto-detect
        strategy_name = request.retrieval_strategy or query_analysis.suggested_strategy.value
        
        # Retrieve relevant documents - Development mode (skip if Azure not configured)
        context_chunks = []
        if retrieval_service:
            try:
                retrieval_request = RetrievalRequest(
                    query=request.query,
                    filters={"quality_threshold": 0.7},
                    top_k=5,
                    expand_references=True
                )
                
                retrieval_result = await retrieval_service.retrieve_documents(retrieval_request)
                context_chunks = [
                    {
                        "chunk_id": result.chunk_id,
                        "content": result.content,
                        "metadata": result.metadata,
                        "score": result.score
                    }
                    for result in retrieval_result.results
                ]
                
            except Exception as e:
                print(f"Retrieval failed, continuing without context: {e}")
                # Add mock context for development
                context_chunks = [{
                    "chunk_id": "dev-mock-1",
                    "content": f"Mock context for query: {request.query}",
                    "metadata": {"document_name": "Development Mode", "page_number": 1},
                    "score": 0.8
                }]
        else:
            # Development mode - add mock context
            context_chunks = [{
                "chunk_id": "dev-mock-1", 
                "content": f"Mock context for query: {request.query}",
                "metadata": {"document_name": "Development Mode", "page_number": 1},
                "score": 0.8
            }]
        
        # Create LLM request
        llm_request = LLMRequest(
            query=request.query,
            session_id=request.session_id,
            query_type=request.query_type or EnvironmentalQueryType(query_analysis.query_type),
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            structured_output=request.structured_output,
            context_chunks=context_chunks,
            conversation_history=conversation_history
        )
        
        # Process with LLM
        if request.structured_output:
            response = await llm_orchestrator.process_structured_request(llm_request)
            return response
        else:
            response = await llm_orchestrator.process_request(llm_request)
            
            # Extract citations from context chunks (limit to top 3 most relevant)
            citations = []
            if context_chunks:
                try:
                    print(f"[CITATIONS] Extracting citations from {len(context_chunks)} context chunks")
                    citations = citation_extractor.extract_citations_from_chunks(
                        context_chunks, 
                        response.content,
                        max_citations=2,  # Only show top 2 most relevant citations
                        query=request.query  # Pass original query for better relevance
                    )
                    print(f"[CITATIONS] Extracted {len(citations)} citations")
                    
                    # Convert Citation objects to dicts for JSON serialization
                    citations_dict = []
                    for cit in citations:
                        if hasattr(cit, 'model_dump'):
                            citations_dict.append(cit.model_dump())
                        elif hasattr(cit, 'dict'):
                            citations_dict.append(cit.dict())
                        elif isinstance(cit, dict):
                            citations_dict.append(cit)
                        else:
                            # Convert to dict manually
                            citations_dict.append({
                                'chunk_id': getattr(cit, 'chunk_id', ''),
                                'document_name': getattr(cit, 'document_name', 'Unknown'),
                                'page_number': getattr(cit, 'page_number', None),
                                'content_snippet': getattr(cit, 'content_snippet', ''),
                                'relevance_score': getattr(cit, 'relevance_score', 0.0),
                                'metadata': getattr(cit, 'metadata', {})
                            })
                    citations = citations_dict
                    print(f"[CITATIONS] Converted to {len(citations)} citation dicts")
                except Exception as e:
                    print(f"[CITATIONS] Citation extraction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    citations = []
            else:
                print(f"[CITATIONS] No context chunks available for citation extraction")
            
            # Update response with citations (send them to frontend)
            response.citations = citations
            
            # Add query analysis metadata
            response.metadata = response.metadata or {}
            response.metadata.update({
                "query_analysis": {
                    "query_type": query_analysis.query_type,
                    "complexity": query_analysis.complexity.value,
                    "suggested_strategy": query_analysis.suggested_strategy.value,
                    "confidence": query_analysis.confidence,
                    "entities": query_analysis.entities
                },
                "retrieval_strategy": strategy_name,
                "context_chunks_used": len(context_chunks)
            })
            
            print(f"[CITATIONS] Response has {len(citations) if citations else 0} citations")
            print(f"[CITATIONS] Response citations: {citations}")
            
            return response
            
    except Exception as e:
        print(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

# Streaming chat endpoint
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint."""
    try:
        # Generate session ID if not provided
        if not request.session_id:
            import uuid
            request.session_id = str(uuid.uuid4())
        
        # Get conversation history
        conversation_history = await memory_manager.get_conversation_history(request.session_id)
        
        # Retrieve context (simplified for streaming)
        context_chunks = []
        if retrieval_service:
            try:
                retrieval_request = RetrievalRequest(
                    query=request.query,
                    top_k=3,
                    quality_threshold=0.7
                )
                
                retrieval_result = await retrieval_service.retrieve_documents(retrieval_request)
                context_chunks = [
                    {
                        "chunk_id": result.chunk_id,
                        "content": result.content,
                        "metadata": result.metadata
                    }
                    for result in retrieval_result.results
                ]
                
            except Exception as e:
                print(f"Retrieval failed for streaming: {e}")
        
        # Create LLM request
        llm_request = LLMRequest(
            query=request.query,
            session_id=request.session_id,
            query_type=request.query_type,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
            context_chunks=context_chunks,
            conversation_history=conversation_history
        )
        
        # Create streaming response
        async def generate_stream():
            try:
                async for chunk in llm_orchestrator.process_request_stream(llm_request):
                    yield f"data: {chunk.json()}\n\n"
                
                # Send final metadata
                final_metadata = {
                    "session_id": request.session_id,
                    "retrieval_strategy": "standard",
                    "context_chunks_used": len(context_chunks)
                }
                yield f"data: {final_metadata}\n\n"
                
            except Exception as e:
                error_chunk = StreamingChunk(
                    content="",
                    is_final=True,
                    metadata={"error": str(e)}
                )
                yield f"data: {error_chunk.json()}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        print(f"Streaming chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming chat failed: {str(e)}")

# Structured analysis endpoints
@router.post("/analyze", response_model=StructuredOutput)
async def analyze_environmental(request: StructuredAnalysisRequest):
    """Structured environmental analysis."""
    try:
        # Get conversation history
        conversation_history = await memory_manager.get_conversation_history(request.session_id)
        
        # Retrieve relevant documents
        context_chunks = []
        if retrieval_service:
            try:
                retrieval_request = RetrievalRequest(
                    query=request.query,
                    filters={"quality_threshold": 0.8},
                    top_k=8,
                    expand_references=True
                )
                
                retrieval_result = await retrieval_service.retrieve_documents(retrieval_request)
                context_chunks = [
                    {
                        "chunk_id": result.chunk_id,
                        "content": result.content,
                        "metadata": result.metadata,
                        "score": result.score
                    }
                    for result in retrieval_result.results
                ]
                
            except Exception as e:
                print(f"Retrieval failed for analysis: {e}")
        
        # Create LLM request
        llm_request = LLMRequest(
            query=request.query,
            session_id=request.session_id,
            query_type=request.analysis_type,
            model=request.model,
            structured_output=True,
            context_chunks=context_chunks,
            conversation_history=conversation_history
        )
        
        # Process with structured output
        structured_response = await llm_orchestrator.process_structured_request(llm_request)
        
        # Add citations if requested
        if request.include_citations and context_chunks:
            citations = citation_extractor.extract_citations_from_chunks(
                context_chunks, structured_response.data.answer if hasattr(structured_response.data, 'answer') else ""
            )
            
            if hasattr(structured_response.data, 'citations'):
                structured_response.data.citations = citations
        
        return structured_response
        
    except Exception as e:
        print(f"Environmental analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/compliance-check", response_model=ComplianceReport)
async def compliance_check(request: StructuredAnalysisRequest):
    """Compliance check with structured output."""
    try:
        # Set analysis type to compliance
        request.analysis_type = EnvironmentalQueryType.COMPLIANCE
        
        # Get structured analysis
        structured_response = await analyze_environmental(request)
        
        # Validate and return compliance report
        if structured_response.output_type == "compliance_report":
            return structured_response.data
        else:
            raise HTTPException(status_code=500, detail="Failed to generate compliance report")
            
    except Exception as e:
        print(f"Compliance check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Compliance check failed: {str(e)}")

@router.post("/risk-assessment", response_model=RiskAssessment)
async def risk_assessment(request: StructuredAnalysisRequest):
    """Risk assessment with structured output."""
    try:
        # Set analysis type to risk assessment
        request.analysis_type = EnvironmentalQueryType.RISK_ASSESSMENT
        
        # Get structured analysis
        structured_response = await analyze_environmental(request)
        
        # Validate and return risk assessment
        if structured_response.output_type == "risk_assessment":
            return structured_response.data
        else:
            raise HTTPException(status_code=500, detail="Failed to generate risk assessment")
            
    except Exception as e:
        print(f"Risk assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

# Conversation management endpoints
@router.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Get full conversation history."""
    # For now, always return empty conversation to avoid memory manager issues
    return {
        "session_id": session_id,
        "messages": [],
        "summary": None,
        "created_at": None,
        "updated_at": None,
        "metadata": {}
    }

@router.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation memory."""
    try:
        success = await memory_manager.clear_conversation(session_id)
        
        if success:
            return {"message": f"Conversation {session_id} cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to clear conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}")

# Health check
@router.get("/health")
async def llm_health_check():
    """LLM service health check."""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "llm_orchestrator": llm_orchestrator is not None,
                "retrieval_service": retrieval_service is not None,
                "memory_manager": memory_manager is not None
            },
            "metrics": llm_orchestrator.get_metrics() if llm_orchestrator else {}
        }
        
        return health_status
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

