"""
Processing module for document ingestion and chunking.
"""

from .models import DocumentFile, Chunk
from .text_processing.ocr import (
    BaseOcrProvider, 
    AzureDocumentIntelligenceOcrProvider, 
    DoclingOcrProvider,
    TesseractOcrProvider,
    HybridOcrProvider
)
from .text_processing.extractors import (
    BaseExtractor,
    PyMuPDFExtractor,
    TesseractExtractor,
    HybridExtractor,
    ExcelExtractor
)
from .text_processing.splitters import (
    BaseSplitter,
    TokenBasedSplitter,
    SemanticOverlapSplitter,
    AdaptiveSplitter
)
from .chunkers.base import BaseChunker, HybridChunker
from .chunkers.enhanced import EnhancedHybridChunker
from .analysis.quality_validator import ChunkQualityValidator
from .analysis.reference_preserver import ReferencePreserver
from .pattern_detector import RepetitivePatternDetector
from .header_filter import (
    BaseHeaderFilter,
    HeaderFilter, 
    CorporateHeaderFilter, 
    AcademicHeaderFilter, 
    LegalHeaderFilter, 
    DocumentTypeDetector, 
    get_header_filter_for_document
)

__all__ = [
    # Models
    'DocumentFile',
    'Chunk',
    
    # OCR Providers
    'BaseOcrProvider',
    'AzureDocumentIntelligenceOcrProvider',
    'DoclingOcrProvider',
    'TesseractOcrProvider',
    'HybridOcrProvider',
    
    # Extractors
    'BaseExtractor',
    'PyMuPDFExtractor',
    'TesseractExtractor',
    'HybridExtractor',
    'ExcelExtractor',
    
    # Splitters
    'BaseSplitter',
    'TokenBasedSplitter',
    'SemanticOverlapSplitter',
    'AdaptiveSplitter',
    
    # Chunkers
    'BaseChunker',
    'HybridChunker',
    'EnhancedHybridChunker',
    
    # Analysis
    'ChunkQualityValidator',
    'ReferencePreserver',
    
    # Pattern Detection
    'RepetitivePatternDetector',
    
    # Header Filtering
    'BaseHeaderFilter',
    'HeaderFilter',
    'CorporateHeaderFilter',
    'AcademicHeaderFilter',
    'LegalHeaderFilter',
    'DocumentTypeDetector',
    'get_header_filter_for_document'
]
