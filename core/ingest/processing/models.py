"""
Core data models for document processing and chunking.

This module contains the fundamental data structures used throughout
the processing pipeline, including the Chunk class and related exceptions.
"""

from typing import Dict, Any
from .utils.ids import deterministic_chunk_id


class ChunkerError(Exception):
    """Raised when something goes wrong during the chunking process."""
    pass


class HybridChunkerError(Exception):
    """Raised when something goes wrong during the hybrid chunking process."""
    pass


class EnhancedChunkerError(Exception):
    """Raised when something goes wrong during the enhanced chunking process."""
    pass


class Chunk:
    """
    Represents a text chunk ready for embedding/indexing.
    
    This is the unified Chunk class used throughout the processing pipeline.
    It contains the chunk content, metadata, and automatically generates
    a deterministic ID based on the metadata.
    """
    
    def __init__(self, content: str, metadata: Dict[str, Any]):
        """
        Initialize a new chunk.
        
        Args:
            content: The text content of the chunk
            metadata: Dictionary containing metadata about the chunk
        """
        self.content = content
        self.metadata = metadata
        self.id = deterministic_chunk_id(metadata)

    def to_langchain_document(self) -> Dict[str, Any]:
        """
        Returns a LangChain-style document dict.
        
        Returns:
            Dictionary with 'page_content' and 'metadata' keys
        """
        return {"page_content": self.content, "metadata": self.metadata}

    def __repr__(self):
        """String representation of the chunk."""
        return f"Chunk(id={self.id}, metadata={self.metadata})"
    
    def __str__(self):
        """String representation showing content preview."""
        content_preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"Chunk(id={self.id}, content='{content_preview}')"
    
    def get_quality_score(self) -> float:
        """
        Get the quality score from metadata if available.
        
        Returns:
            Quality score (0.0-1.0) or 0.0 if not available
        """
        if 'quality_metrics' in self.metadata:
            return self.metadata['quality_metrics'].get('quality_score', 0.0)
        return 0.0
    
    def has_references(self) -> bool:
        """
        Check if this chunk contains references.
        
        Returns:
            True if chunk has references, False otherwise
        """
        return self.metadata.get('has_references', False)
    
    def get_page_number(self) -> int:
        """
        Get the page number from metadata.
        
        Returns:
            Page number or 0 if not available
        """
        return self.metadata.get('page_number', 0)
    
    def get_document_id(self) -> str:
        """
        Get the document ID from metadata.
        
        Returns:
            Document ID or empty string if not available
        """
        return self.metadata.get('document_id', '')
    
    def get_blob_name(self) -> str:
        """
        Get the blob name from metadata.
        
        Returns:
            Blob name or empty string if not available
        """
        return self.metadata.get('document_blob', '')
