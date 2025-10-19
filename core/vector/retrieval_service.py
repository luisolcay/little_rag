"""
Advanced retrieval service for hybrid search with metadata filtering.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .embeddings.azure_embedding_service import AzureEmbeddingService
from .azure_search.search_client import AzureSearchClient, HybridSearchRequest, SearchResult

@dataclass
class RetrievalRequest:
    """Request for document retrieval."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 10
    vector_k: int = 50
    quality_threshold: float = 0.7
    semantic_config: Optional[str] = "semantic-config"
    expand_references: bool = True
    max_context_length: int = 4000

@dataclass
class RetrievalResult:
    """Result of document retrieval."""
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
    query_vector: List[float]
    expanded_context: Optional[str] = None
    metadata: Dict[str, Any] = None

class AdvancedRetrievalService:
    """
    Advanced retrieval service with hybrid search capabilities.
    
    Features:
    - Hybrid search (vector + keyword + semantic)
    - Metadata-aware filtering
    - Reference expansion
    - Context aggregation
    - Quality-based ranking
    """
    
    def __init__(
        self,
        index_name: str = "collahuasi-documents",
        embedding_model: str = "text-embedding-3-large"
    ):
        """
        Initialize the retrieval service.
        
        Args:
            index_name: Name of the Azure AI Search index
            embedding_model: Model for generating query embeddings
        """
        self.index_name = index_name
        self.embedding_model = embedding_model
        
        # Initialize services
        self.embedding_service = AzureEmbeddingService()
        self.search_client = AzureSearchClient(index_name=index_name)
        
        print(f"[RETRIEVAL] Initialized service for index: {index_name}")
    
    def _build_filter_expression(self, filters: Dict[str, Any]) -> Optional[str]:
        """
        Build OData filter expression from filters dictionary.
        
        Args:
            filters: Dictionary of filter criteria
            
        Returns:
            OData filter expression string
        """
        if not filters:
            return None
        
        filter_parts = []
        
        # Quality threshold
        if "quality_threshold" in filters:
            threshold = filters["quality_threshold"]
            filter_parts.append(f"quality_score ge {threshold}")
        
        # Document filter
        if "document_blob" in filters:
            doc_blob = filters["document_blob"]
            filter_parts.append(f"document_blob eq '{doc_blob}'")
        
        # Document ID filter
        if "document_id" in filters:
            doc_id = filters["document_id"]
            filter_parts.append(f"document_id eq '{doc_id}'")
        
        # Page range filter
        if "page_range" in filters:
            min_page, max_page = filters["page_range"]
            filter_parts.append(f"page_number ge {min_page} and page_number le {max_page}")
        
        # References filter
        if "has_references_only" in filters and filters["has_references_only"]:
            filter_parts.append("has_references eq true")
        
        # OCR filter
        if "needs_ocr" in filters:
            needs_ocr = filters["needs_ocr"]
            filter_parts.append(f"metadata/needs_ocr eq {str(needs_ocr).lower()}")
        
        # Reference count filter
        if "min_reference_count" in filters:
            min_refs = filters["min_reference_count"]
            filter_parts.append(f"metadata/reference_count ge {min_refs}")
        
        return " and ".join(filter_parts) if filter_parts else None
    
    async def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for the query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        embedding_results = await self.embedding_service.generate_embeddings([query])
        return embedding_results[0].embedding if embedding_results else []
    
    def _expand_context_with_references(
        self, 
        results: List[SearchResult], 
        max_length: int = 4000
    ) -> Tuple[List[SearchResult], Optional[str]]:
        """
        Expand context by retrieving referenced chunks.
        
        Args:
            results: Initial search results
            max_length: Maximum context length
            
        Returns:
            Tuple of (expanded_results, aggregated_context)
        """
        expanded_results = results.copy()
        context_parts = []
        current_length = 0
        
        for result in results:
            # Add main result to context
            context_parts.append(f"[Document: {result.metadata['document_blob']}, Page: {result.metadata['page_number']}]")
            context_parts.append(result.content)
            current_length += len(result.content)
            
            # Check if we have references to expand
            if result.metadata.get("has_references", False):
                metadata = result.metadata.get("metadata", {})
                processing_stats = metadata.get("processing_stats", "{}")
                
                try:
                    import json
                    stats = json.loads(processing_stats) if isinstance(processing_stats, str) else processing_stats
                    references = stats.get("references", [])
                    
                    # Retrieve referenced chunks (simplified - in production, you'd query by chunk IDs)
                    for ref in references[:3]:  # Limit to 3 references per result
                        if current_length >= max_length:
                            break
                        
                        # In a real implementation, you'd query the search index for these chunk IDs
                        # For now, we'll just note that references exist
                        context_parts.append(f"[Reference: {ref.get('reference_value', 'Unknown')}]")
                        
                except Exception as e:
                    print(f"[RETRIEVAL] Error processing references: {e}")
            
            if current_length >= max_length:
                break
        
        aggregated_context = "\n\n".join(context_parts) if context_parts else None
        return expanded_results, aggregated_context
    
    async def retrieve_documents(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Retrieve documents using hybrid search.
        
        Args:
            request: RetrievalRequest with search parameters
            
        Returns:
            RetrievalResult with search results and metadata
        """
        start_time = datetime.now()
        
        try:
            # 1. Generate query embedding
            print(f"[RETRIEVAL] Generating embedding for query: '{request.query}'")
            query_vector = await self._generate_query_embedding(request.query)
            
            if not query_vector:
                raise Exception("Failed to generate query embedding")
            
            # 2. Build filter expression
            filter_expression = self._build_filter_expression(request.filters or {})
            
            # 3. Perform hybrid search
            print(f"[RETRIEVAL] Performing hybrid search...")
            hybrid_request = HybridSearchRequest(
                query_text=request.query,
                query_vector=query_vector,
                filters=filter_expression,
                top=request.top_k,
                vector_k=request.vector_k,
                semantic_config=request.semantic_config
            )
            
            search_results = self.search_client.hybrid_search(hybrid_request)
            
            # 4. Expand context with references if requested
            expanded_results = search_results
            expanded_context = None
            
            if request.expand_references and search_results:
                print(f"[RETRIEVAL] Expanding context with references...")
                expanded_results, expanded_context = self._expand_context_with_references(
                    search_results, 
                    request.max_context_length
                )
            
            # 5. Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 6. Create result
            result = RetrievalResult(
                query=request.query,
                results=expanded_results,
                total_results=len(expanded_results),
                processing_time=processing_time,
                query_vector=query_vector,
                expanded_context=expanded_context,
                metadata={
                    "filter_expression": filter_expression,
                    "quality_threshold": request.quality_threshold,
                    "vector_k": request.vector_k,
                    "semantic_config": request.semantic_config,
                    "expand_references": request.expand_references
                }
            )
            
            print(f"[RETRIEVAL] Retrieval completed: {len(expanded_results)} results in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"[RETRIEVAL] Retrieval failed: {str(e)}")
            
            return RetrievalResult(
                query=request.query,
                results=[],
                total_results=0,
                processing_time=processing_time,
                query_vector=[],
                metadata={"error": str(e)}
            )
    
    async def search_by_document(
        self, 
        query: str, 
        document_blob: str, 
        top_k: int = 5
    ) -> RetrievalResult:
        """
        Search within a specific document.
        
        Args:
            query: Search query
            document_blob: Document filename
            top_k: Number of results
            
        Returns:
            RetrievalResult with results from the specific document
        """
        request = RetrievalRequest(
            query=query,
            filters={"document_blob": document_blob},
            top_k=top_k
        )
        
        return await self.retrieve_documents(request)
    
    async def search_by_quality(
        self, 
        query: str, 
        min_quality: float = 0.8, 
        top_k: int = 10
    ) -> RetrievalResult:
        """
        Search with quality filtering.
        
        Args:
            query: Search query
            min_quality: Minimum quality score
            top_k: Number of results
            
        Returns:
            RetrievalResult with high-quality results
        """
        request = RetrievalRequest(
            query=query,
            filters={"quality_threshold": min_quality},
            top_k=top_k
        )
        
        return await self.retrieve_documents(request)
    
    async def search_with_references(
        self, 
        query: str, 
        top_k: int = 10
    ) -> RetrievalResult:
        """
        Search for chunks that have references.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            RetrievalResult with chunks that have references
        """
        request = RetrievalRequest(
            query=query,
            filters={"has_references_only": True},
            top_k=top_k,
            expand_references=True
        )
        
        return await self.retrieve_documents(request)
    
    async def search_by_page_range(
        self, 
        query: str, 
        min_page: int, 
        max_page: int, 
        top_k: int = 10
    ) -> RetrievalResult:
        """
        Search within a specific page range.
        
        Args:
            query: Search query
            min_page: Minimum page number
            max_page: Maximum page number
            top_k: Number of results
            
        Returns:
            RetrievalResult with results from the page range
        """
        request = RetrievalRequest(
            query=query,
            filters={"page_range": (min_page, max_page)},
            top_k=top_k
        )
        
        return await self.retrieve_documents(request)
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get retrieval service statistics."""
        embedding_stats = self.embedding_service.get_statistics()
        
        return {
            "index_name": self.index_name,
            "embedding_model": self.embedding_model,
            "embedding_stats": embedding_stats,
            "search_client_initialized": self.search_client is not None
        }

# Convenience functions
async def search_documents(
    query: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    index_name: str = "collahuasi-documents"
) -> RetrievalResult:
    """
    Convenience function for document search.
    
    Args:
        query: Search query
        filters: Optional filter criteria
        top_k: Number of results
        index_name: Name of the search index
        
    Returns:
        RetrievalResult with search results
    """
    service = AdvancedRetrievalService(index_name=index_name)
    
    request = RetrievalRequest(
        query=query,
        filters=filters,
        top_k=top_k
    )
    
    return await service.retrieve_documents(request)
