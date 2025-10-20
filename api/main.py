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
from pydantic import BaseModel, Field
import shutil
import uuid

# Import document processing components
from core.ingest.document_processor import DocumentProcessor

# Import our new vector search components
from core.vector.retrieval_service import AdvancedRetrievalService, RetrievalRequest
from core.vector.indexing.pipeline import IndexingPipeline
from core.vector.azure_search.index_manager import IndexManager

# Import new LLM layer components
from core.llm.orchestrator import llm_orchestrator
from core.memory.memory_manager import memory_manager
from core.retrieval.strategy_router import query_classifier

# Import existing components
from .pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest, DocumentMetadata

# Import new endpoints
from .llm_endpoints import router as llm_router
from .admin_endpoints import router as admin_router
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

# Initialize services
retrieval_service = None
index_manager = None
document_processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global retrieval_service, index_manager, document_processor
    
    try:
        # Initialize document processor
        document_processor = DocumentProcessor(container_name="orbe")
        
        # Initialize memory manager
        await memory_manager.initialize()
        logging.info("Memory manager initialized successfully")
        
        # Initialize retrieval service
        retrieval_service = AdvancedRetrievalService(index_name="collahuasi-documents")
        
        # Initialize index manager
        index_manager = IndexManager(index_name="collahuasi-documents")
        
        # Ensure index exists
        if not index_manager.index_exists():
            logging.info("Creating Azure AI Search index...")
            index_manager.create_index()
        
        logging.info("Azure AI Search services initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        # Continue with degraded functionality

# Pydantic models for new endpoints
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filter criteria")
    top_k: int = Field(10, ge=1, le=50, description="Number of results")
    quality_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum quality score")
    expand_references: bool = Field(True, description="Expand context with references")

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float
    metadata: Dict[str, Any]

class IndexingRequest(BaseModel):
    chunks_file: str = Field("artifacts/chunks.jsonl", description="Path to chunks file")
    batch_size: int = Field(100, ge=1, le=200, description="Batch size for processing")

class IndexingResponse(BaseModel):
    success: bool
    total_chunks: int
    total_uploaded: int
    total_failed: int
    processing_time: float
    index_document_count: int
    embedding_stats: Dict[str, Any]

# Main chat endpoint (compatible with Streamlit frontend)
@app.post("/chat", response_model=QueryResponse)
async def chat(query_input: QueryInput):
    """Main chat endpoint with Azure LLM and intelligent retrieval."""
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")
    
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())

    try:
        # Get conversation history
        conversation_history = await memory_manager.get_conversation_history(session_id)
        
        # Classify query for intelligent retrieval
        query_analysis = query_classifier.analyze_query(query_input.question)
        
        # Retrieve relevant documents
        context_chunks = []
        citations = []
        
        if query_input.use_context and retrieval_service:
            try:
                retrieval_request = RetrievalRequest(
                    query=query_input.question,
                    filters={"quality_threshold": 0.7},
                    top_k=5,
                    expand_references=True
                )
                
                retrieval_result = await retrieval_service.retrieve_documents(retrieval_request)
                
                # Extract chunks and citations
                for result in retrieval_result.results:
                    context_chunks.append({
                        "chunk_id": result.chunk_id,
                        "content": result.content,
                        "metadata": result.metadata,
                        "score": result.score
                    })
                    
                    # Create citation
                    citation = {
                        "document_name": result.metadata.get("filename", "Unknown"),
                        "chunk_id": result.chunk_id,
                        "page_number": result.metadata.get("page_number"),
                        "section": result.metadata.get("section"),
                        "relevance_score": result.score,
                        "content_snippet": result.content[:200] + "..." if len(result.content) > 200 else result.content
                    }
                    citations.append(citation)
                    
            except Exception as e:
                logging.warning(f"Retrieval failed, continuing without context: {e}")
        
        # Create LLM request
        from core.llm.models import LLMRequest, EnvironmentalQueryType, ModelName
        
        llm_request = LLMRequest(
            query=query_input.question,
            session_id=session_id,
            query_type=EnvironmentalQueryType.GENERAL_QA,
            model=ModelName(query_input.model.value),
            context_chunks=context_chunks,
            conversation_history=conversation_history
        )
        
        # Process with LLM orchestrator
        llm_response = await llm_orchestrator.process_request(llm_request)
        answer = llm_response.content
        
        logging.info(f"Session ID: {session_id}, AI Response: {answer}, Citations: {len(citations)}")
        
        return QueryResponse(
            answer=answer, 
            session_id=session_id, 
            model=query_input.model,
            citations=citations
        )
        
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

# Upload document endpoint
@app.post("/upload-doc")
async def upload_and_index_document(file: UploadFile = File(...)):
    """Upload to Blob Storage, process with OCR if needed, and index to Azure AI Search."""
    
    file_id = str(uuid.uuid4())
    
    try:
        # 1. Read file content
        file_content = await file.read()
        
        # 2. Process through complete pipeline (Blob Storage + OCR + Chunking)
        processing_result = await document_processor.process_uploaded_file(
            file_content=file_content,
            filename=file.filename,
            file_id=file_id
        )
        
        if not processing_result["success"]:
            raise Exception("Document processing failed")
        
        # 3. Save chunks to JSONL with proper format
        chunks_file = f"artifacts/{file_id}_chunks.jsonl"
        os.makedirs("artifacts", exist_ok=True)
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in processing_result["chunks"]:
                # Convert Chunk object to dict if needed
                chunk_dict = {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata
                }
                f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")
        
        # 4. Index to Azure AI Search
        indexing_result = None
        if index_manager and retrieval_service:
            try:
                indexing_pipeline = IndexingPipeline(
                    chunks_file=chunks_file,
                    index_name="collahuasi-documents",
                    batch_size=50
                )
                indexing_result = await indexing_pipeline.run_indexing()
            except Exception as e:
                logging.error(f"Indexing failed: {e}")
        
        # 5. Save document metadata to Cosmos DB
        doc_metadata = {
            "id": file_id,
            "filename": file.filename,
            "blob_name": processing_result["blob_name"],
            "blob_url": processing_result["blob_url"],
            "upload_timestamp": datetime.now().isoformat(),
            "processing_status": "completed" if indexing_result else "processed",
            "chunks_count": len(processing_result["chunks"]),
            "needs_ocr": processing_result["needs_ocr"]
        }
        
        try:
            from core.memory.cosmos_store import cosmos_client
            cosmos_client.create_item("documents", doc_metadata)
        except Exception as e:
            logging.warning(f"Could not save to Cosmos DB: {e}")
        
        # 6. Return comprehensive response
        return {
            "message": f"File {file.filename} uploaded and indexed successfully",
            "file_id": file_id,
            "blob_url": processing_result["blob_url"],
            "needs_ocr": processing_result["needs_ocr"],
            "processing_result": {
                "chunks_count": len(processing_result["chunks"]),
                "chunks_file": chunks_file
            },
            "indexing_result": indexing_result
        }
        
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# New search endpoint
@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents using Azure AI Search hybrid search."""
    if not retrieval_service:
        raise HTTPException(status_code=503, detail="Azure AI Search service not available")
    
    try:
        # Create retrieval request
        retrieval_request = RetrievalRequest(
            query=request.query,
            filters=request.filters,
            top_k=request.top_k,
            quality_threshold=request.quality_threshold,
            expand_references=request.expand_references
        )
        
        # Perform search
        result = await retrieval_service.retrieve_documents(retrieval_request)
        
        # Convert results to response format
        search_results = []
        for search_result in result.results:
            search_results.append({
                "chunk_id": search_result.chunk_id,
                "content": search_result.content,
                "score": search_result.score,
                "metadata": search_result.metadata,
                "highlights": search_result.highlights
            })
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=result.total_results,
            processing_time=result.processing_time,
            metadata=result.metadata
        )
        
    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Indexing endpoint
@app.post("/index", response_model=IndexingResponse)
async def index_documents(request: IndexingRequest):
    """Index documents to Azure AI Search."""
    if not index_manager:
        raise HTTPException(status_code=503, detail="Azure AI Search service not available")
    
    try:
        # Create indexing pipeline
        pipeline = IndexingPipeline(
            chunks_file=request.chunks_file,
            index_name="collahuasi-documents",
            batch_size=request.batch_size
        )
        
        # Run indexing
        result = await pipeline.run_indexing()
        
        return IndexingResponse(
            success=result["success"],
            total_chunks=result["total_chunks"],
            total_uploaded=result["total_uploaded"],
            total_failed=result["total_failed"],
            processing_time=result["processing_time_seconds"],
            index_document_count=result["index_document_count"],
            embedding_stats=result["embedding_stats"]
        )
        
    except Exception as e:
        logging.error(f"Indexing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

# Index management endpoints
@app.get("/index/status")
async def get_index_status():
    """Get index status and statistics."""
    if not index_manager:
        raise HTTPException(status_code=503, detail="Azure AI Search service not available")
    
    try:
        info = index_manager.get_index_info()
        stats = index_manager.get_index_statistics()
        
        return {
            "index_exists": index_manager.index_exists(),
            "index_info": info,
            "statistics": stats
        }
        
    except Exception as e:
        logging.error(f"Index status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get index status: {str(e)}")

@app.post("/index/create")
async def create_index():
    """Create the search index."""
    if not index_manager:
        raise HTTPException(status_code=503, detail="Azure AI Search service not available")
    
    try:
        success = index_manager.create_index()
        
        if success:
            return {"message": "Index created successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create index")
            
    except Exception as e:
        logging.error(f"Index creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create index: {str(e)}")

@app.delete("/index")
async def delete_index():
    """Delete the search index."""
    if not index_manager:
        raise HTTPException(status_code=503, detail="Azure AI Search service not available")
    
    try:
        success = index_manager.delete_index()
        
        if success:
            return {"message": "Index deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete index")
            
    except Exception as e:
        logging.error(f"Index deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete index: {str(e)}")

# Service statistics endpoint
@app.get("/stats")
async def get_service_statistics():
    """Get service statistics."""
    stats = {}
    
    if retrieval_service:
        stats["retrieval_service"] = retrieval_service.get_service_statistics()
    
    if index_manager:
        stats["index_manager"] = {
            "index_exists": index_manager.index_exists(),
            "index_name": index_manager.index_name
        }
    
    return stats

# Legacy endpoints (for backward compatibility)
@app.get("/list-docs", response_model=List[DocumentInfo])
def list_documents():
    """List all documents (legacy endpoint)."""
    # Return empty list since we're not using SQLite anymore
    return []

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    """Delete document (legacy endpoint)."""
    # Note: This would need to be updated to work with Azure AI Search
    # For now, return a message indicating it's not implemented
    return {"message": "Document deletion not implemented for Azure AI Search"}

@app.get("/documents")
async def list_documents():
    """List all uploaded documents from Cosmos DB."""
    try:
        from core.memory.cosmos_store import cosmos_client
        
        query = "SELECT * FROM c ORDER BY c.upload_timestamp DESC"
        documents = cosmos_client.query_items("documents", query)
        
        return {
            "documents": list(documents),
            "total": len(list(documents))
        }
    except Exception as e:
        logging.error(f"Error listing documents: {e}")
        return {"documents": [], "total": 0}

# Include new routers
app.include_router(llm_router)
app.include_router(admin_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "retrieval_service": retrieval_service is not None,
            "index_manager": index_manager is not None,
            "llm_orchestrator": llm_orchestrator is not None,
            "memory_manager": memory_manager is not None,
            "azure_integration": True
        }
    }
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
