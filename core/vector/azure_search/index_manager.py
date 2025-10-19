"""
Azure AI Search index management utilities.
"""

import os
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

try:
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchIndex, SearchField, SearchFieldDataType, SimpleField, 
        SearchableField, VectorSearch, HnswAlgorithmConfiguration,
        VectorSearchProfile, SemanticConfiguration, SemanticPrioritizedFields,
        SemanticField, SemanticSearch
    )
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ResourceNotFoundError
except ImportError:
    print("[WARN] azure-search-documents not installed. Install with: pip install azure-search-documents")
    SearchIndexClient = None
    SearchIndex = None
    AzureKeyCredential = None
    ResourceNotFoundError = Exception

load_dotenv()

class IndexManager:
    """
    Manager for Azure AI Search index operations.
    
    Features:
    - Index creation and configuration
    - Schema management
    - Vector search setup
    - Semantic search configuration
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: str = "collahuasi-documents"
    ):
        """
        Initialize Index Manager.
        
        Args:
            endpoint: Azure AI Search endpoint URL
            api_key: Azure AI Search API key
            index_name: Name of the search index
        """
        if not SearchIndexClient:
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
        
        # Initialize index client
        self.credential = AzureKeyCredential(self.api_key)
        self.client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
        
        print(f"[INDEX] Initialized manager for index: {self.index_name}")
    
    def create_index_schema(self) -> SearchIndex:
        """
        Create the search index schema for Collahuasi documents.
        
        Returns:
            SearchIndex object with complete configuration
        """
        # Define fields
        fields = [
            # Primary key
            SimpleField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                key=True,
                searchable=False,
                filterable=False,
                sortable=False,
                facetable=False
            ),
            
            # Content field
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="es.microsoft"  # Spanish analyzer
            ),
            
            # Vector field
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=3072,  # text-embedding-3-large dimensions
                vector_search_profile_name="hnsw-profile"
            ),
            
            # Document metadata
            SimpleField(
                name="document_id",
                type=SearchFieldDataType.String,
                searchable=False,
                filterable=True,
                facetable=True,
                sortable=False
            ),
            
            SimpleField(
                name="document_blob",
                type=SearchFieldDataType.String,
                searchable=False,
                filterable=True,
                facetable=True,
                sortable=False
            ),
            
            SimpleField(
                name="page_number",
                type=SearchFieldDataType.Int32,
                searchable=False,
                filterable=True,
                sortable=True,
                facetable=False
            ),
            
            SimpleField(
                name="quality_score",
                type=SearchFieldDataType.Double,
                searchable=False,
                filterable=True,
                sortable=True,
                facetable=False
            ),
            
            SimpleField(
                name="has_references",
                type=SearchFieldDataType.Boolean,
                searchable=False,
                filterable=True,
                facetable=False,
                sortable=False
            ),
            
            # Complex metadata field
            SearchField(
                name="metadata",
                type=SearchFieldDataType.ComplexType,
                fields=[
                    SimpleField(
                        name="chunk_index",
                        type=SearchFieldDataType.Int32,
                        searchable=False,
                        filterable=True,
                        sortable=True,
                        facetable=False
                    ),
                    SimpleField(
                        name="needs_ocr",
                        type=SearchFieldDataType.Boolean,
                        searchable=False,
                        filterable=True,
                        facetable=False,
                        sortable=False
                    ),
                    SimpleField(
                        name="reference_count",
                        type=SearchFieldDataType.Int32,
                        searchable=False,
                        filterable=True,
                        sortable=True,
                        facetable=False
                    ),
                    SimpleField(
                        name="ingested_at",
                        type=SearchFieldDataType.Int64,
                        searchable=False,
                        filterable=True,
                        sortable=True,
                        facetable=False
                    ),
                    SimpleField(
                        name="processing_stats",
                        type=SearchFieldDataType.String,
                        searchable=False,
                        filterable=False,
                        sortable=False,
                        facetable=False
                    )
                ]
            )
        ]
        
        # Vector search configuration
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-algorithm",
                    kind="hnsw",
                    parameters={
                        "metric": "cosine",
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="hnsw-profile",
                    algorithm_configuration_name="hnsw-algorithm"
                )
            ]
        )
        
        # Semantic search configuration
        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="semantic-config",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="document_blob"),
                        content_fields=[
                            SemanticField(field_name="content")
                        ],
                        keywords_fields=[
                            SemanticField(field_name="document_id")
                        ]
                    )
                )
            ]
        )
        
        # Create index
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        return index
    
    def create_index(self) -> bool:
        """
        Create the search index.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            index = self.create_index_schema()
            
            # Check if index already exists
            try:
                existing_index = self.client.get_index(self.index_name)
                print(f"[INDEX] Index '{self.index_name}' already exists")
                return True
            except ResourceNotFoundError:
                pass  # Index doesn't exist, proceed with creation
            
            # Create the index
            result = self.client.create_index(index)
            print(f"[INDEX] Index '{self.index_name}' created successfully")
            return True
            
        except Exception as e:
            print(f"[INDEX] Failed to create index: {str(e)}")
            return False
    
    def delete_index(self) -> bool:
        """
        Delete the search index.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_index(self.index_name)
            print(f"[INDEX] Index '{self.index_name}' deleted successfully")
            return True
            
        except ResourceNotFoundError:
            print(f"[INDEX] Index '{self.index_name}' does not exist")
            return True
        except Exception as e:
            print(f"[INDEX] Failed to delete index: {str(e)}")
            return False
    
    def index_exists(self) -> bool:
        """
        Check if the index exists.
        
        Returns:
            True if index exists, False otherwise
        """
        try:
            self.client.get_index(self.index_name)
            return True
        except ResourceNotFoundError:
            return False
        except Exception as e:
            print(f"[INDEX] Error checking index existence: {str(e)}")
            return False
    
    def get_index_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the index.
        
        Returns:
            Dictionary with index information or None if error
        """
        try:
            index = self.client.get_index(self.index_name)
            
            info = {
                "name": index.name,
                "field_count": len(index.fields),
                "vector_search_enabled": index.vector_search is not None,
                "semantic_search_enabled": index.semantic_search is not None,
                "fields": [
                    {
                        "name": field.name,
                        "type": str(field.type),
                        "searchable": getattr(field, 'searchable', False),
                        "filterable": getattr(field, 'filterable', False),
                        "sortable": getattr(field, 'sortable', False),
                        "facetable": getattr(field, 'facetable', False)
                    }
                    for field in index.fields
                ]
            }
            
            return info
            
        except ResourceNotFoundError:
            print(f"[INDEX] Index '{self.index_name}' does not exist")
            return None
        except Exception as e:
            print(f"[INDEX] Failed to get index info: {str(e)}")
            return None
    
    def update_index(self) -> bool:
        """
        Update the index schema (if needed).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # For now, we'll recreate the index
            # In production, you might want to implement proper schema evolution
            print(f"[INDEX] Updating index '{self.index_name}'...")
            
            # Delete and recreate
            self.delete_index()
            return self.create_index()
            
        except Exception as e:
            print(f"[INDEX] Failed to update index: {str(e)}")
            return False
    
    def get_index_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with statistics or None if error
        """
        try:
            stats = self.client.get_search_index_statistics(self.index_name)
            
            return {
                "document_count": stats.document_count,
                "storage_size": stats.storage_size,
                "vector_index_size": stats.vector_index_size
            }
            
        except Exception as e:
            print(f"[INDEX] Failed to get index statistics: {str(e)}")
            return None

# Convenience functions
def create_collahuasi_index(index_name: str = "collahuasi-documents") -> bool:
    """Create the Collahuasi documents index."""
    manager = IndexManager(index_name=index_name)
    return manager.create_index()

def delete_collahuasi_index(index_name: str = "collahuasi-documents") -> bool:
    """Delete the Collahuasi documents index."""
    manager = IndexManager(index_name=index_name)
    return manager.delete_index()

def index_exists(index_name: str = "collahuasi-documents") -> bool:
    """Check if the Collahuasi documents index exists."""
    manager = IndexManager(index_name=index_name)
    return manager.index_exists()
