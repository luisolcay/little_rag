"""
Hierarchical retrieval strategy for context expansion.
"""

import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass

from ..vector.retrieval_service import AdvancedRetrievalService, RetrievalRequest, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class HierarchicalRetrievalConfig:
    """Configuration for hierarchical retrieval."""
    expand_levels: int = 2
    max_context_tokens: int = 4000
    breadth_first: bool = True
    max_results_per_level: int = 5
    context_expansion_threshold: float = 0.7

class HierarchicalRetrievalStrategy:
    """Hierarchical retrieval with context expansion."""
    
    def __init__(self, retrieval_service: AdvancedRetrievalService):
        self.retrieval_service = retrieval_service
        self.name = "hierarchical"
    
    async def retrieve(self, query: str, config: HierarchicalRetrievalConfig) -> RetrievalResult:
        """Retrieve documents with hierarchical context expansion."""
        try:
            # Start with initial retrieval
            initial_result = await self._initial_retrieval(query, config)
            
            if not initial_result.results:
                return initial_result
            
            # Expand context hierarchically
            expanded_result = await self._expand_context_hierarchically(
                initial_result, query, config
            )
            
            logger.info(f"Hierarchical retrieval completed: {len(expanded_result.results)} results")
            return expanded_result
            
        except Exception as e:
            logger.error(f"Hierarchical retrieval failed: {e}")
            return RetrievalResult(
                query=query,
                results=[],
                total_results=0,
                processing_time=0.0,
                query_vector=[],
                metadata={"error": str(e)}
            )
    
    async def _initial_retrieval(self, query: str, config: HierarchicalRetrievalConfig) -> RetrievalResult:
        """Perform initial retrieval."""
        request = RetrievalRequest(
            query=query,
            top_k=config.max_results_per_level,
            quality_threshold=0.6
        )
        
        return await self.retrieval_service.retrieve_documents(request)
    
    async def _expand_context_hierarchically(self, initial_result: RetrievalResult, 
                                           query: str, config: HierarchicalRetrievalConfig) -> RetrievalResult:
        """Expand context hierarchically."""
        all_results = initial_result.results.copy()
        processed_chunks = set()
        
        # Mark initial results as processed
        for result in initial_result.results:
            processed_chunks.add(result.chunk_id)
        
        # Expand for each level
        for level in range(1, config.expand_levels + 1):
            level_results = await self._expand_level(all_results, query, config, level)
            
            # Add new results
            for result in level_results:
                if result.chunk_id not in processed_chunks:
                    all_results.append(result)
                    processed_chunks.add(result.chunk_id)
            
            # Check if we have enough context
            if self._has_sufficient_context(all_results, config):
                break
        
        # Sort and limit results
        all_results.sort(key=lambda x: x.score, reverse=True)
        final_results = all_results[:config.max_results_per_level * config.expand_levels]
        
        # Update metadata
        initial_result.results = final_results
        initial_result.total_results = len(final_results)
        initial_result.metadata = initial_result.metadata or {}
        initial_result.metadata["hierarchical_expansion"] = {
            "levels_expanded": config.expand_levels,
            "total_chunks_found": len(all_results),
            "unique_chunks": len(processed_chunks),
            "strategy": "breadth_first" if config.breadth_first else "depth_first"
        }
        
        return initial_result
    
    async def _expand_level(self, current_results: List, query: str, 
                           config: HierarchicalRetrievalConfig, level: int) -> List:
        """Expand context at a specific level."""
        level_results = []
        
        # Get documents from current level
        documents = self._get_documents_from_results(current_results)
        
        for doc_name in documents:
            try:
                # Search within the same document for related content
                doc_results = await self._search_within_document(query, doc_name, config)
                level_results.extend(doc_results)
                
                # Search for related documents
                related_results = await self._search_related_documents(query, doc_name, config)
                level_results.extend(related_results)
                
            except Exception as e:
                logger.warning(f"Failed to expand level {level} for document {doc_name}: {e}")
                continue
        
        return level_results
    
    def _get_documents_from_results(self, results: List) -> Set[str]:
        """Extract unique document names from results."""
        documents = set()
        for result in results:
            doc_name = result.metadata.get("document_blob", "")
            if doc_name:
                documents.add(doc_name)
        return documents
    
    async def _search_within_document(self, query: str, document_name: str, 
                                    config: HierarchicalRetrievalConfig) -> List:
        """Search for related content within the same document."""
        request = RetrievalRequest(
            query=query,
            filters={"document_blob": document_name},
            top_k=config.max_results_per_level,
            quality_threshold=0.5
        )
        
        result = await self.retrieval_service.retrieve_documents(request)
        
        # Mark as same-document expansion
        for search_result in result.results:
            search_result.metadata = search_result.metadata or {}
            search_result.metadata["expansion_type"] = "same_document"
        
        return result.results
    
    async def _search_related_documents(self, query: str, reference_doc: str, 
                                      config: HierarchicalRetrievalConfig) -> List:
        """Search for related documents."""
        # This would use document similarity or cross-references
        # For now, use a simplified approach
        
        request = RetrievalRequest(
            query=query,
            top_k=config.max_results_per_level,
            quality_threshold=0.4
        )
        
        result = await self.retrieval_service.retrieve_documents(request)
        
        # Filter out the reference document
        related_results = []
        for search_result in result.results:
            if search_result.metadata.get("document_blob") != reference_doc:
                search_result.metadata = search_result.metadata or {}
                search_result.metadata["expansion_type"] = "related_document"
                related_results.append(search_result)
        
        return related_results
    
    def _has_sufficient_context(self, results: List, config: HierarchicalRetrievalConfig) -> bool:
        """Check if we have sufficient context."""
        # Simple token count estimation
        total_tokens = sum(len(result.content) // 4 for result in results)
        return total_tokens >= config.max_context_tokens
    
    async def get_document_context(self, query: str, document_name: str, 
                                 config: HierarchicalRetrievalConfig) -> RetrievalResult:
        """Get comprehensive context for a specific document."""
        # Get all chunks from the document
        request = RetrievalRequest(
            query=query,
            filters={"document_blob": document_name},
            top_k=50,  # Get more chunks for comprehensive context
            quality_threshold=0.3
        )
        
        result = await self.retrieval_service.retrieve_documents(request)
        
        # Organize by sections/chapters if available
        organized_results = self._organize_by_sections(result.results)
        
        result.results = organized_results
        result.metadata = result.metadata or {}
        result.metadata["document_context"] = {
            "document_name": document_name,
            "total_chunks": len(organized_results),
            "sections_found": len(set(r.metadata.get("section", "unknown") for r in organized_results))
        }
        
        return result
    
    def _organize_by_sections(self, results: List) -> List:
        """Organize results by document sections."""
        # Group by section if metadata available
        sections = {}
        for result in results:
            section = result.metadata.get("section", "unknown")
            if section not in sections:
                sections[section] = []
            sections[section].append(result)
        
        # Sort sections and flatten
        organized = []
        for section in sorted(sections.keys()):
            section_results = sorted(sections[section], key=lambda x: x.metadata.get("page_number", 0))
            organized.extend(section_results)
        
        return organized
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            "name": self.name,
            "description": "Hierarchical retrieval with context expansion",
            "capabilities": [
                "Multi-level context expansion",
                "Document-wide context gathering",
                "Section-based organization",
                "Related document discovery"
            ],
            "best_for": [
                "Complex technical analysis",
                "Comprehensive document review",
                "Context-heavy queries",
                "Multi-document research"
            ]
        }

