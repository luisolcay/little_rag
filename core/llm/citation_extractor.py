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
    def extract_citations_from_chunks(chunks: List[Dict[str, Any]], response_content: str, max_citations: int = 3, query: str = None) -> List[Citation]:
        """Extract relevant citations from retrieved chunks."""
        citations = []
        
        for chunk in chunks:
            try:
                # Calculate relevance based on content overlap
                chunk_content = chunk.get("content", "")
                
                # Calculate relevance to both response and query (if provided)
                response_relevance = CitationExtractor._calculate_relevance(response_content, chunk_content)
                
                # If query is provided, also consider query relevance
                if query:
                    query_relevance = CitationExtractor._calculate_relevance(query, chunk_content)
                    # Weighted average: 60% response relevance, 40% query relevance
                    relevance_score = (response_relevance * 0.6) + (query_relevance * 0.4)
                else:
                    relevance_score = response_relevance
                
                # Log relevance scores for debugging
                logger.info(f"Chunk relevance score: {relevance_score:.3f}")
                
                # Lower threshold to 0.05 to catch more relevant chunks
                if relevance_score > 0.05:
                    # Get first 3-4 sentences or ~500 characters as summary
                    sentences = chunk_content.split('. ')
                    if len(sentences) > 3:
                        summary = '. '.join(sentences[:3]) + '.'
                    else:
                        summary = chunk_content[:500] + ('...' if len(chunk_content) > 500 else '')
                    
                    citation = Citation(
                        chunk_id=chunk.get("chunk_id", ""),
                        document_name=chunk.get("metadata", {}).get("document_blob", "Unknown"),
                        page_number=chunk.get("metadata", {}).get("page_number"),
                        content_snippet=summary,  # First few sentences as summary
                        relevance_score=relevance_score,
                        metadata=chunk.get("metadata", {})
                    )
                    citations.append(citation)
                    
            except Exception as e:
                logger.warning(f"Failed to create citation from chunk: {e}")
                continue
        
        # Sort by relevance and return only top N
        sorted_citations = sorted(citations, key=lambda x: x.relevance_score, reverse=True)
        logger.info(f"Returning {len(sorted_citations[:max_citations])} citations out of {len(sorted_citations)} total")
        return sorted_citations[:max_citations]
    
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
