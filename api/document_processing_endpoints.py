"""
Document Processing Endpoints
============================

Advanced document processing endpoints with hybrid OCR, pattern detection,
quality validation, and intelligent chunking.
"""

import os
import asyncio
import logging
import tempfile
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse

# Import our advanced RAG components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.ingest.processing import (
    DocumentFile, 
    EnhancedHybridChunker,
    AzureDocumentIntelligenceOcrProvider,
    DoclingOcrProvider
)

# Import Pydantic models
from pydantic_models import (
    DocumentUploadRequest,
    DocumentProcessingRequest,
    DocumentProcessingResponse,
    ChunkInfo,
    PatternAnalysisResult,
    QualityMetrics,
    ChunkingConfig,
    ProcessingConfig
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/documents", tags=["Document Processing"])

# Global services (initialized on startup)
chunker = None
azure_ocr = None
docling_ocr = None

@router.on_event("startup")
async def startup_event():
    """Initialize document processing services."""
    global chunker, azure_ocr, docling_ocr
    
    try:
        # Initialize OCR providers
        try:
            azure_ocr = AzureDocumentIntelligenceOcrProvider()
            logger.info("✅ Azure Document Intelligence OCR initialized")
        except Exception as e:
            logger.warning(f"⚠️ Azure OCR not available: {e}")
            azure_ocr = None
        
        try:
            docling_ocr = DoclingOcrProvider()
            logger.info("✅ Docling OCR initialized")
        except Exception as e:
            logger.warning(f"⚠️ Docling OCR not available: {e}")
            docling_ocr = None
        
        # Initialize enhanced chunker
        chunker = EnhancedHybridChunker(
            docling_provider=docling_ocr,
            azure_provider=azure_ocr,
            max_tokens=900,
            overlap_tokens=120,
            ocr_threshold=0.1,
            min_text_length=50,
            
            # Quality validation
            min_chunk_length=50,
            max_chunk_length=2000,
            min_sentence_count=1,
            max_repetition_ratio=0.3,
            max_special_char_ratio=0.3,
            min_quality_threshold=0.5,
            
            # Semantic overlap
            overlap_sentences=2,
            preserve_paragraphs=True,
            
            # Reference preservation
            enable_reference_preservation=True,
            
            # Pattern detection
            enable_pattern_detection=True,
            pattern_similarity_threshold=0.8,
            auto_clean_headers=True,
            noise_threshold=10.0,
            
            verbose=False  # Disable verbose logging in API
        )
        
        logger.info("✅ Document processing services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize document processing services: {e}")
        raise

@router.post("/upload", response_model=Dict[str, Any])
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload a document for processing.
    
    Args:
        file: The document file to upload
        background_tasks: Background tasks for processing
        
    Returns:
        Document upload response with processing status
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are supported"
            )
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Create temporary file
        temp_dir = Path(tempfile.gettempdir()) / "rag_documents"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / f"{document_id}_{file.filename}"
        
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Get file info
        file_size = len(content)
        
        # Create document metadata
        document_metadata = {
            "document_id": document_id,
            "filename": file.filename,
            "file_path": str(temp_file_path),
            "file_size": file_size,
            "upload_timestamp": datetime.now().isoformat(),
            "processing_status": "uploaded",
            "content_type": file.content_type or "application/pdf"
        }
        
        logger.info(f"📄 Document uploaded: {file.filename} ({file_size:,} bytes)")
        
        return {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "file_size": file_size,
            "upload_timestamp": document_metadata["upload_timestamp"],
            "processing_status": "uploaded",
            "message": "Document uploaded successfully. Ready for processing."
        }
        
    except Exception as e:
        logger.error(f"❌ Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/process", response_model=DocumentProcessingResponse)
async def process_document(
    request: DocumentProcessingRequest,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Process a document with advanced chunking and analysis.
    
    Args:
        request: Document processing request with parameters
        background_tasks: Background tasks for processing
        
    Returns:
        Document processing response with chunks and analysis
    """
    try:
        if not chunker:
            raise HTTPException(
                status_code=503, 
                detail="Document processing service not available"
            )
        
        # Find the document file
        temp_dir = Path(tempfile.gettempdir()) / "rag_documents"
        document_files = list(temp_dir.glob(f"{request.document_id}_*"))
        
        if not document_files:
            raise HTTPException(
                status_code=404, 
                detail=f"Document {request.document_id} not found"
            )
        
        document_file_path = document_files[0]
        filename = document_file_path.name.split('_', 1)[1] if '_' in document_file_path.name else document_file_path.name
        
        logger.info(f"🔄 Processing document: {filename}")
        
        # Create DocumentFile
        doc_file = DocumentFile(
            local_path=str(document_file_path),
            blob_name=filename,
            metadata={
                "blob_name": filename,
                "original_ext": ".pdf",
                "size_bytes": document_file_path.stat().st_size,
                "needs_ocr": False,
                "document_id": request.document_id
            },
            needs_ocr=False
        )
        
        # Apply custom processing options if provided
        if request.processing_options:
            # Update chunker parameters based on request
            if "chunking_params" in request.processing_options:
                chunking_params = request.processing_options["chunking_params"]
                # Note: In a real implementation, you'd update chunker parameters here
                pass
        
        # Process document
        start_time = datetime.now()
        chunks = chunker.chunk_document(doc_file)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Document processed: {len(chunks)} chunks generated")
        
        # Convert chunks to ChunkInfo models
        chunk_infos = []
        for i, chunk in enumerate(chunks):
            chunk_info = ChunkInfo(
                chunk_id=chunk.id,
                content=chunk.content,
                metadata=chunk.metadata,
                quality_score=chunk.metadata.get('quality_score'),
                embedding=None,  # Will be generated separately
                page_number=chunk.metadata.get('page_number'),
                chunk_index=i + 1,
                cleaning_applied=chunk.metadata.get('cleaning_applied', False),
                noise_reduction=chunk.metadata.get('noise_reduction', 0)
            )
            chunk_infos.append(chunk_info)
        
        # Extract pattern analysis from first chunk if available
        pattern_analysis = None
        if chunks and 'pattern_analysis' in chunks[0].metadata:
            pattern_data = chunks[0].metadata['pattern_analysis']
            pattern_analysis = PatternAnalysisResult(
                noise_percentage=pattern_data.get('noise_percentage', 0),
                similarity_patterns_count=pattern_data.get('similarity_patterns', 0),
                header_patterns_count=0,  # Would need to extract from analysis
                common_lines_count=pattern_data.get('common_lines', 0),
                cleaning_applied=any(chunk.metadata.get('cleaning_applied', False) for chunk in chunks),
                recommendations=[],  # Would need to extract from analysis
                most_common_line=None
            )
        
        # Extract quality metrics
        quality_metrics = None
        if chunks and 'processing_metadata' in chunks[0].metadata:
            processing_metadata = chunks[0].metadata['processing_metadata']
            quality_report = processing_metadata.get('quality_report', {})
            
            quality_metrics = QualityMetrics(
                total_chunks=len(chunks),
                valid_chunks=len(chunks),  # All chunks passed quality filter
                invalid_chunks=0,
                average_quality_score=quality_report.get('average_quality', 0),
                quality_distribution=quality_report.get('quality_distribution', {}),
                most_common_issues=quality_report.get('most_common_issues', []),
                processing_time=processing_time
            )
        
        # Get processing statistics
        processing_stats = chunker.get_processing_statistics()
        
        return DocumentProcessingResponse(
            document_id=request.document_id,
            success=True,
            chunks=chunk_infos,
            pattern_analysis=pattern_analysis,
            quality_metrics=quality_metrics,
            processing_stats=processing_stats,
            error_message=None
        )
        
    except Exception as e:
        logger.error(f"❌ Document processing failed: {e}")
        return DocumentProcessingResponse(
            document_id=request.document_id,
            success=False,
            chunks=[],
            pattern_analysis=None,
            quality_metrics=None,
            processing_stats={},
            error_message=str(e)
        )

@router.get("/{document_id}/chunks", response_model=List[ChunkInfo])
async def get_document_chunks(document_id: str):
    """
    Get chunks for a specific document.
    
    Args:
        document_id: The document ID
        
    Returns:
        List of chunks for the document
    """
    try:
        # In a real implementation, you'd retrieve chunks from a database
        # For now, we'll return a placeholder response
        return []
        
    except Exception as e:
        logger.error(f"❌ Failed to get chunks for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}/analysis", response_model=PatternAnalysisResult)
async def get_document_analysis(document_id: str):
    """
    Get pattern analysis for a specific document.
    
    Args:
        document_id: The document ID
        
    Returns:
        Pattern analysis results
    """
    try:
        # In a real implementation, you'd retrieve analysis from a database
        # For now, we'll return a placeholder response
        return PatternAnalysisResult(
            noise_percentage=0.0,
            similarity_patterns_count=0,
            header_patterns_count=0,
            common_lines_count=0,
            cleaning_applied=False,
            recommendations=[],
            most_common_line=None
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get analysis for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{document_id}/reprocess", response_model=DocumentProcessingResponse)
async def reprocess_document(
    document_id: str,
    chunking_config: Optional[ChunkingConfig] = None,
    processing_config: Optional[ProcessingConfig] = None
):
    """
    Reprocess a document with different parameters.
    
    Args:
        document_id: The document ID
        chunking_config: New chunking configuration
        processing_config: New processing configuration
        
    Returns:
        Document processing response with new chunks
    """
    try:
        # Create a new processing request with updated parameters
        request = DocumentProcessingRequest(
            document_id=document_id,
            processing_options={
                "chunking_params": chunking_config.dict() if chunking_config else {},
                "processing_params": processing_config.dict() if processing_config else {}
            },
            enable_pattern_detection=processing_config.enable_pattern_detection if processing_config else True,
            enable_quality_validation=processing_config.enable_quality_validation if processing_config else True,
            enable_reference_preservation=processing_config.enable_reference_preservation if processing_config else True
        )
        
        # Process the document with new parameters
        return await process_document(request)
        
    except Exception as e:
        logger.error(f"❌ Document reprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and its associated chunks.
    
    Args:
        document_id: The document ID
        
    Returns:
        Deletion confirmation
    """
    try:
        # Find and delete the document file
        temp_dir = Path(tempfile.gettempdir()) / "rag_documents"
        document_files = list(temp_dir.glob(f"{document_id}_*"))
        
        deleted_files = []
        for file_path in document_files:
            file_path.unlink()
            deleted_files.append(file_path.name)
        
        # In a real implementation, you'd also delete chunks from the database
        
        logger.info(f"🗑️ Document deleted: {document_id}")
        
        return {
            "success": True,
            "document_id": document_id,
            "deleted_files": deleted_files,
            "message": f"Document {document_id} and associated data deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_processing_status():
    """
    Get the status of document processing services.
    
    Returns:
        Service status information
    """
    try:
        status = {
            "chunker_available": chunker is not None,
            "azure_ocr_available": azure_ocr is not None,
            "docling_ocr_available": docling_ocr is not None,
            "services_initialized": all([chunker, azure_ocr or docling_ocr]),
            "timestamp": datetime.now().isoformat()
        }
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Failed to get processing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
