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
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

def rerank_with_previous_sources(
    current_results: List[Dict],
    previous_chunk_ids: List[str],
    boost_factor: float = 0.5,
    min_retention: float = 0.5
) -> List[Dict]:
    """
    Re-rank results to maintain consistency with previous sources.
    
    Args:
        current_results: Current retrieval results
        previous_chunk_ids: Chunk IDs from previous query
        boost_factor: Score boost for previous sources (0.5 = 50% boost)
        min_retention: Minimum ratio of previous sources to retain (0.5 = 50%)
    """
    if not previous_chunk_ids:
        return current_results
    
    # Boost scores for chunks that appeared before
    boosted_results = []
    for result in current_results:
        if result["chunk_id"] in previous_chunk_ids:
            result["score"] = result["score"] * (1 + boost_factor)
            result["metadata"]["boosted"] = True
        boosted_results.append(result)
    
    # Re-sort by boosted scores
    boosted_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Ensure minimum retention of previous sources
    previous_in_results = sum(1 for r in boosted_results[:5] if r["chunk_id"] in previous_chunk_ids)
    retention_ratio = previous_in_results / min(len(previous_chunk_ids), 5)
    
    logger.info(f"[RERANK] Retention: {previous_in_results}/5 sources ({retention_ratio:.1%})")
    
    return boosted_results[:5]

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
    query_type: Optional[str] = None  # Accept string, will convert to enum
    model: Optional[str] = None  # Accept string, will convert to enum
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=8192)
    stream: bool = Field(default=False)
    structured_output: bool = Field(default=False)
    retrieval_strategy: Optional[str] = None
    
    def convert_enums(self):
        """Convert string values to proper enums."""
        # Convert query_type string to enum
        if self.query_type and isinstance(self.query_type, str):
            try:
                # Try to convert to EnvironmentalQueryType
                self.query_type = EnvironmentalQueryType(self.query_type.lower())
            except ValueError:
                # Default to GENERAL_QA if invalid
                self.query_type = EnvironmentalQueryType.GENERAL_QA
        
        # Convert model string to enum  
        if self.model and isinstance(self.model, str):
            try:
                self.model = ModelName(self.model)
            except ValueError:
                self.model = ModelName.GPT4_O_MINI
        elif not self.model:
            self.model = ModelName.GPT4_O_MINI

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
@router.post("/chat")
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with intelligent retrieval and memory."""
    logger.info(f"[CHAT] Received request: query='{request.query[:50]}...', session_id={request.session_id}")
    
    try:
        # Convert string parameters to enums
        request.convert_enums()
        logger.info(f"[CHAT] After conversion - query_type={request.query_type}, model={request.model}")
        
        # Generate session ID if not provided
        if not request.session_id:
            import uuid
            request.session_id = str(uuid.uuid4())
        
        # Get conversation history from memory manager
        conversation_history = []
        try:
            if memory_manager:
                logger.info(f"[CHAT] Attempting to load history for session: {request.session_id}")
                history = await memory_manager.get_conversation_history(request.session_id)
                logger.info(f"[CHAT] Retrieved history: {len(history) if history else 0} messages")
                if history and len(history) > 0:
                    conversation_history = history
                    logger.info(f"[CHAT] Successfully loaded {len(conversation_history)} messages from history")
                else:
                    logger.info(f"[CHAT] No history found for session {request.session_id}")
            else:
                logger.warning(f"[CHAT] Memory manager not available")
        except Exception as e:
            logger.warning(f"[CHAT] Could not load conversation history: {e}", exc_info=True)
            conversation_history = []
        
        # Expand query using conversation history if available
        expanded_query = request.query
        
        # Check if query is generic/vague
        GENERIC_QUESTIONS = ["que me recomiendas", "de que estamos hablando", "mas informacion", 
                            "más información", "que otra cosa", "continua", "dime mas"]
        is_generic = any(gq in request.query.lower() for gq in GENERIC_QUESTIONS)
        
        if conversation_history and len(conversation_history) > 0 and is_generic:
            try:
                # For generic questions, manually extract topic from last user+assistant messages
                recent_messages = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
                
                # Extract key terms from conversation
                all_content = []
                for msg in recent_messages:
                    if hasattr(msg, 'content'):
                        all_content.append(msg.content)
                
                # Combine all content and extract meaningful terms
                combined_text = " ".join(all_content).lower()
                
                # Extract important words (4+ chars, filter common words)
                common_words = {'sistema', 'que', 'son', 'para', 'con', 'del', 'las', 'una', 'está', 'este', 'actividad', 'horas', 'hombre'}
                words = [w for w in combined_text.split() if len(w) >= 4 and w not in common_words]
                
                # Keep unique, meaningful keywords
                topic_keywords = list(dict.fromkeys(words[:15]))  # Preserve order
                topic_context = " ".join(topic_keywords) if topic_keywords else ""
                
                # Manually expand generic query with topic
                if topic_context:
                    expanded_query = f"{request.query} sobre {topic_context}"
                    logger.info(f"[CHAT] Generic query expanded: '{request.query}' -> '{expanded_query}'")
                    logger.info(f"[CHAT] Topic extracted: {topic_context[:100]}")
                else:
                    expanded_query = request.query
            except Exception as e:
                logger.warning(f"[CHAT] Query expansion failed: {e}")
                expanded_query = request.query
        elif conversation_history and len(conversation_history) > 0:
            # For specific questions, just add context from history
            try:
                llm = llm_orchestrator._get_llm(request.model)
                contextualize_chain = llm_orchestrator.contextualize_prompt | llm | StrOutputParser()
                expanded_query = await contextualize_chain.ainvoke({
                    "chat_history": conversation_history,
                    "input": request.query
                })
                logger.info(f"[CHAT] Expanded query: '{request.query}' -> '{expanded_query}'")
            except Exception as e:
                logger.warning(f"[CHAT] Query expansion failed: {e}")
                expanded_query = request.query
        
        # Classify expanded query
        query_analysis = query_classifier.analyze_query(expanded_query)
        
        # Use specified strategy or auto-detect
        strategy_name = request.retrieval_strategy or query_analysis.suggested_strategy.value
        
        # Retrieve relevant documents - Development mode (skip if Azure not configured)
        context_chunks = []
        if retrieval_service:
            try:
                retrieval_request = RetrievalRequest(
                    query=expanded_query,
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
        
        # Re-rank with previous sources if this is a follow-up question
        previous_chunk_ids = []
        if conversation_history and len(conversation_history) > 0:
            # Get previous chunk IDs from session metadata
            try:
                from core.memory.redis_cache import redis_cache
                session_key = f"session:{request.session_id}:chunks"
                cached_chunks = await redis_cache.get_cache(session_key)
                if cached_chunks:
                    previous_chunk_ids = cached_chunks.get("chunk_ids", [])
                    logger.info(f"[CHAT] Loaded {len(previous_chunk_ids)} previous chunk IDs")
            except Exception as e:
                logger.debug(f"[CHAT] Could not load previous chunks: {e}")
        
        # Re-rank if we have previous sources
        if previous_chunk_ids and context_chunks:
            context_chunks = rerank_with_previous_sources(
                context_chunks,
                previous_chunk_ids,
                boost_factor=1.0,      # Increased from 0.5 for better consistency
                min_retention=0.6      # Increased from 0.5 to retain more previous sources
            )
        
        # Store current chunk IDs for next query
        current_chunk_ids = [chunk["chunk_id"] for chunk in context_chunks]
        try:
            from core.memory.redis_cache import redis_cache
            session_key = f"session:{request.session_id}:chunks"
            await redis_cache.set_cache(
                session_key,
                {"chunk_ids": current_chunk_ids, "query": request.query},
                ttl=3600  # 1 hour
            )
        except Exception as e:
            logger.debug(f"[CHAT] Could not store chunk IDs: {e}")
        
        # Convert LangChain messages to dict format for LLMRequest
        history_dicts = []
        if conversation_history and len(conversation_history) > 0:
            for msg in conversation_history:
                if hasattr(msg, 'content'):
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                    history_dicts.append({"role": role, "content": msg.content})
        
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
            conversation_history=history_dicts
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
            
            # Get content from response object
            content = response.content
            
            # Get model value (handle both ModelName enum and string)
            model_value = response.model.value if hasattr(response.model, 'value') else str(response.model)
            
            # Convert response to frontend-compatible format
            return {
                "content": content,
                "session_id": response.session_id,
                "model": model_value,
                "query_type": request.query_type.value if request.query_type else None,
                "tokens_used": response.tokens_used,
                "processing_time": response.processing_time,
                "confidence_score": query_analysis.confidence,
                "citations": citations,
                "metadata": {
                    "query_analysis": {
                        "query_type": query_analysis.query_type,
                        "complexity": query_analysis.complexity.value,
                        "suggested_strategy": query_analysis.suggested_strategy.value,
                        "confidence": query_analysis.confidence,
                        "entities": query_analysis.entities
                    },
                    "retrieval_strategy": strategy_name,
                    "context_chunks_used": len(context_chunks)
                },
                "created_at": datetime.now().isoformat()
            }
            
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
        
        # Convert LangChain messages to dict format
        history_dicts = []
        if conversation_history and len(conversation_history) > 0:
            for msg in conversation_history:
                if hasattr(msg, 'content'):
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                    history_dicts.append({"role": role, "content": msg.content})
        
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
            conversation_history=history_dicts
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
        
        # Convert LangChain messages to dict format
        history_dicts = []
        if conversation_history and len(conversation_history) > 0:
            for msg in conversation_history:
                if hasattr(msg, 'content'):
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                    history_dicts.append({"role": role, "content": msg.content})
        
        # Create LLM request
        llm_request = LLMRequest(
            query=request.query,
            session_id=request.session_id,
            query_type=request.analysis_type,
            model=request.model,
            structured_output=True,
            context_chunks=context_chunks,
            conversation_history=history_dicts
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
    try:
        # Try to get conversation from memory manager
        if memory_manager:
            conversation = await memory_manager.get_conversation_history(session_id)
            if conversation:
                # Convert LangChain messages to frontend format
                messages = []
                for msg in conversation:
                    # Handle LangChain BaseMessage format
                    if hasattr(msg, 'content'):
                        # Map LangChain message types to frontend roles
                        msg_type = getattr(msg, 'type', 'human')
                        role = 'user' if msg_type in ['human', 'user'] else 'assistant'
                        
                        messages.append({
                            "role": role,
                            "content": msg.content,
                            "timestamp": datetime.now().isoformat(),
                            "metadata": {}
                        })
                
                now = datetime.now().isoformat()
                return {
                    "session_id": session_id,
                    "messages": messages,
                    "summary": None,
                    "created_at": now,
                    "updated_at": now,
                    "metadata": {}
                }
    except Exception as e:
        logger.debug(f"Memory manager not available or failed: {e}")
    
    # Return empty conversation if no memory or conversation not found
    now = datetime.now().isoformat()
    return {
        "session_id": session_id,
        "messages": [],
        "summary": None,
        "created_at": now,
        "updated_at": now,
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
        # Get metrics if available, otherwise use empty dict
        metrics = {}
        if llm_orchestrator and hasattr(llm_orchestrator, 'get_metrics'):
            try:
                metrics = llm_orchestrator.get_metrics()
            except Exception:
                pass
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "llm_orchestrator": llm_orchestrator is not None,
                "retrieval_service": retrieval_service is not None,
                "memory_manager": memory_manager is not None
            },
            "metrics": metrics
        }
        
        return health_status
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

