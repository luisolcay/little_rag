"""
Azure AI Search integration module.
"""

from .search_client import AzureSearchClient
from .index_manager import IndexManager

__all__ = [
    "AzureSearchClient",
    "IndexManager"
]
