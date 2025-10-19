"""
Azure AI Search client for vector search operations.
"""

import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from dotenv import load_dotenv
import json
from datetime import datetime

try:
    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizedQuery
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ResourceNotFoundError
except ImportError:
    print("[WARN] azure-search-documents not installed. Install with: pip install azure-search-documents")
    SearchClient = None
    VectorizedQuery = None
    AzureKeyCredential = None
    ResourceNotFoundError = Exception

load_dotenv()

@dataclass
class SearchResult:
    """Search result from Azure AI Search."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    highlights: Optional[List[str]] = None

@dataclass
class HybridSearchRequest:
    """Request for hybrid search."""
    query_text: str
    query_vector: List[float]
    filters: Optional[str] = None
    top: int = 10
    vector_k: int = 50
    semantic_config: Optional[str] = "semantic-config"

class AzureSearchClient:
    """
    Client for Azure AI Search operations.
    
    Features:
    - Hybrid search (vector + keyword + semantic)
    - Metadata filtering
    - Batch operations
    - Error handling
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: str = "collahuasi-documents"
    ):
        """
        Initialize Azure Search Client.
        
        Args:
            endpoint: Azure AI Search endpoint URL
            api_key: Azure AI Search API key
            index_name: Name of the search index
        """
        if not SearchClient:
            raise ImportError("azure-search-documents package is required")
        
        self.endpoint = endpoint or os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_SEARCH_API_KEY_ADMIN")
        self.index_name = index_name
        
        if not all([self.endpoint, self.api_key]):
            raise ValueError(
                "Missing required Azure AI Search configuration. "
                "Set AZURE_AI_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY_ADMIN"
            )
        
        # Clean endpoint URL
        self.endpoint = self.endpoint.rstrip('/')
        
        # Initialize search client
        self.credential = AzureKeyCredential(self.api_key)
        self.client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        
        print(f"[SEARCH] Initialized client for index: {self.index_name}")
    
    def hybrid_search(
        self, 
        request: HybridSearchRequest
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector similarity and keyword matching.
        
        Args:
            request: HybridSearchRequest with query details
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Create vector query
            vector_query = VectorizedQuery(
                vector=request.query_vector,
                k_nearest_neighbors=request.vector_k,
                fields="content_vector"
            )
            
            # Perform search
            search_results = self.client.search(
                search_text=request.query_text,
                vector_queries=[vector_query],
                filter=request.filters,
                top=request.top,
                select=["chunk_id", "content", "page_number", "quality_score", 
                       "document_id", "document_blob", "has_references", "metadata"]
            )
            
            # Convert results
            results = []
            for result in search_results:
                if result is None:
                    continue
                    
                search_result = SearchResult(
                    chunk_id=result.get("chunk_id", ""),
                    content=result.get("content", ""),
                    score=result.get("@search.score", 0.0),
                    metadata={
                        "page_number": result.get("page_number"),
                        "quality_score": result.get("quality_score"),
                        "document_id": result.get("document_id"),
                        "document_blob": result.get("document_blob"),
                        "has_references": result.get("has_references"),
                        "metadata": result.get("metadata", {})
                    },
                    highlights=result.get("@search.highlights", {}).get("content", []) if result.get("@search.highlights") else None
                )
                results.append(search_result)
            
            print(f"[SEARCH] Hybrid search completed: {len(results)} results")
            return results
            
        except Exception as e:
            print(f"[SEARCH] Hybrid search failed: {str(e)}")
            raise
    
    def vector_search(
        self, 
        query_vector: List[float], 
        filters: Optional[str] = None,
        top: int = 10,
        k: int = 50
    ) -> List[SearchResult]:
        """
        Perform vector-only search.
        
        Args:
            query_vector: Query embedding vector
            filters: OData filter expression
            top: Number of results to return
            k: Number of nearest neighbors to retrieve
            
        Returns:
            List of SearchResult objects
        """
        try:
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=k,
                fields="content_vector"
            )
            
            search_results = self.client.search(
                vector_queries=[vector_query],
                filter=filters,
                top=top,
                select=["chunk_id", "content", "page_number", "quality_score", 
                       "document_id", "document_blob", "has_references", "metadata"]
            )
            
            results = []
            for result in search_results:
                if result is None:
                    continue
                    
                search_result = SearchResult(
                    chunk_id=result.get("chunk_id", ""),
                    content=result.get("content", ""),
                    score=result.get("@search.score", 0.0),
                    metadata={
                        "page_number": result.get("page_number"),
                        "quality_score": result.get("quality_score"),
                        "document_id": result.get("document_id"),
                        "document_blob": result.get("document_blob"),
                        "has_references": result.get("has_references"),
                        "metadata": result.get("metadata", {})
                    }
                )
                results.append(search_result)
            
            print(f"[SEARCH] Vector search completed: {len(results)} results")
            return results
            
        except Exception as e:
            print(f"[SEARCH] Vector search failed: {str(e)}")
            raise
    
    def keyword_search(
        self, 
        query_text: str, 
        filters: Optional[str] = None,
        top: int = 10
    ) -> List[SearchResult]:
        """
        Perform keyword-only search.
        
        Args:
            query_text: Search query text
            filters: OData filter expression
            top: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            search_results = self.client.search(
                search_text=query_text,
                filter=filters,
                top=top,
                select=["chunk_id", "content", "page_number", "quality_score", 
                       "document_id", "document_blob", "has_references", "metadata"]
            )
            
            results = []
            for result in search_results:
                if result is None:
                    continue
                    
                search_result = SearchResult(
                    chunk_id=result.get("chunk_id", ""),
                    content=result.get("content", ""),
                    score=result.get("@search.score", 0.0),
                    metadata={
                        "page_number": result.get("page_number"),
                        "quality_score": result.get("quality_score"),
                        "document_id": result.get("document_id"),
                        "document_blob": result.get("document_blob"),
                        "has_references": result.get("has_references"),
                        "metadata": result.get("metadata", {})
                    },
                    highlights=result.get("@search.highlights", {}).get("content", []) if result.get("@search.highlights") else None
                )
                results.append(search_result)
            
            print(f"[SEARCH] Keyword search completed: {len(results)} results")
            return results
            
        except Exception as e:
            print(f"[SEARCH] Keyword search failed: {str(e)}")
            raise
    
    def upload_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Upload documents to the search index.
        
        Args:
            documents: List of documents to upload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.client.upload_documents(documents)
            
            # Check for errors
            failed_docs = [doc for doc in result if not doc.succeeded]
            if failed_docs:
                print(f"[SEARCH] Upload completed with {len(failed_docs)} errors")
                for doc in failed_docs:
                    print(f"[SEARCH] Error: {doc.error_message}")
                return False
            
            print(f"[SEARCH] Upload successful: {len(documents)} documents")
            return True
            
        except Exception as e:
            print(f"[SEARCH] Upload failed: {str(e)}")
            return False
    
    def delete_documents(self, document_keys: List[str]) -> bool:
        """
        Delete documents from the search index.
        
        Args:
            document_keys: List of document keys to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            documents = [{"chunk_id": key} for key in document_keys]
            result = self.client.delete_documents(documents)
            
            failed_docs = [doc for doc in result if not doc.succeeded]
            if failed_docs:
                print(f"[SEARCH] Delete completed with {len(failed_docs)} errors")
                return False
            
            print(f"[SEARCH] Delete successful: {len(document_keys)} documents")
            return True
            
        except Exception as e:
            print(f"[SEARCH] Delete failed: {str(e)}")
            return False
    
    def get_document_count(self) -> int:
        """Get total number of documents in the index."""
        try:
            result = self.client.get_document_count()
            return result
        except Exception as e:
            print(f"[SEARCH] Failed to get document count: {str(e)}")
            return 0
    
    def search_with_filters(
        self,
        query_text: str,
        query_vector: List[float],
        quality_threshold: float = 0.7,
        document_filter: Optional[str] = None,
        page_range: Optional[tuple] = None,
        has_references_only: bool = False,
        top: int = 10
    ) -> List[SearchResult]:
        """
        Search with advanced filtering options.
        
        Args:
            query_text: Search query text
            query_vector: Query embedding vector
            quality_threshold: Minimum quality score
            document_filter: Specific document to search
            page_range: Tuple of (min_page, max_page)
            has_references_only: Only return chunks with references
            top: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        # Build filter expression
        filters = []
        
        if quality_threshold > 0:
            filters.append(f"quality_score ge {quality_threshold}")
        
        if document_filter:
            filters.append(f"document_blob eq '{document_filter}'")
        
        if page_range:
            min_page, max_page = page_range
            filters.append(f"page_number ge {min_page} and page_number le {max_page}")
        
        if has_references_only:
            filters.append("has_references eq true")
        
        filter_expression = " and ".join(filters) if filters else None
        
        # Create hybrid search request
        request = HybridSearchRequest(
            query_text=query_text,
            query_vector=query_vector,
            filters=filter_expression,
            top=top
        )
        
        return self.hybrid_search(request)
