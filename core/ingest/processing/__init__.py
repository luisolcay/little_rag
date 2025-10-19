"""
Document processing module with enhanced chunking capabilities.

This module provides a comprehensive document processing pipeline with:
- Intelligent hybrid chunking with OCR fallback
- Quality validation and filtering
- Semantic overlap preservation
- Reference detection and linking
- Multiple text extraction methods
"""

# Core models
from .models import Chunk, ChunkerError, HybridChunkerError, EnhancedChunkerError

# Chunkers
from .chunkers import BaseChunker, HybridChunker, EnhancedHybridChunker

# Text processing
from .text_processing import (
    BaseExtractor, AutoExtractor, PdfExtractor, DocxExtractor, HtmlExtractor, TxtExtractor,
    BaseSplitter, TokenAwareSentenceSplitter, SemanticOverlapSplitter,
    normalize_text,
    BaseOcrProvider, AzureDocumentIntelligenceOcrProvider, DoclingOcrProvider,
)

# Analysis
from .analysis import ChunkQualityValidator, ReferencePreserver

# Utils
from .utils import deterministic_chunk_id

__all__ = [
    # Core models
    "Chunk", "ChunkerError", "HybridChunkerError", "EnhancedChunkerError",
    
    # Chunkers
    "BaseChunker", "HybridChunker", "EnhancedHybridChunker",
    
    # Text processing
    "BaseExtractor", "AutoExtractor", "PdfExtractor", "DocxExtractor", "HtmlExtractor", "TxtExtractor",
    "BaseSplitter", "TokenAwareSentenceSplitter", "SemanticOverlapSplitter",
    "normalize_text",
    "BaseOcrProvider", "AzureDocumentIntelligenceOcrProvider", "DoclingOcrProvider",
    
    # Analysis
    "ChunkQualityValidator", "ReferencePreserver",
    
    # Utils
    "deterministic_chunk_id",
]