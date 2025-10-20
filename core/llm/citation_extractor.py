"""
Citation extraction utilities for LLM responses.
"""

import logging
from typing import List, Dict, Any
from .models import Citation

logger = logging.getLogger(__name__)

class CitationExtractor:
    """Extract citations from LLM responses."""
    
    @staticmethod
    def extract_citations_from_chunks(chunks: List[Dict[str, Any]], response_content: str) -> List[Citation]:
        """Extract relevant citations from retrieved chunks."""
        citations = []
        
        for chunk in chunks:
            try:
                # Calculate relevance based on content overlap
                chunk_content = chunk.get("content", "")
                relevance_score = CitationExtractor._calculate_relevance(response_content, chunk_content)
                
                if relevance_score > 0.1:  # Minimum relevance threshold
                    citation = Citation(
                        chunk_id=chunk.get("chunk_id", ""),
                        document_name=chunk.get("metadata", {}).get("document_blob", "Unknown"),
                        page_number=chunk.get("metadata", {}).get("page_number"),
                        content_snippet=chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
                        relevance_score=relevance_score,
                        metadata=chunk.get("metadata", {})
                    )
                    citations.append(citation)
                    
            except Exception as e:
                logger.warning(f"Failed to create citation from chunk: {e}")
                continue
        
        return sorted(citations, key=lambda x: x.relevance_score, reverse=True)
    
    @staticmethod
    def _calculate_relevance(response_content: str, chunk_content: str) -> float:
        """Calculate relevance score between response and chunk content."""
        try:
            # Simple word overlap calculation
            response_words = set(response_content.lower().split())
            chunk_words = set(chunk_content.lower().split())
            
            if not response_words or not chunk_words:
                return 0.0
            
            overlap = len(response_words.intersection(chunk_words))
            total_words = len(response_words.union(chunk_words))
            
            return overlap / total_words if total_words > 0 else 0.0
            
        except Exception:
            return 0.0

# Global instance
citation_extractor = CitationExtractor()
