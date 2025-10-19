"""
Vector search and embeddings module for RAG system.
"""

from .embeddings.azure_embedding_service import AzureEmbeddingService
from .azure_search.search_client import AzureSearchClient
from .azure_search.index_manager import IndexManager
from .indexing.pipeline import IndexingPipeline

__all__ = [
    "AzureEmbeddingService",
    "AzureSearchClient", 
    "IndexManager",
    "IndexingPipeline"
]
