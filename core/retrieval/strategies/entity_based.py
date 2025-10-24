"""
Entity-based retrieval strategy for entity-focused document search.
"""

import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass

from ..vector.retrieval_service import AdvancedRetrievalService, RetrievalRequest, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class EntityRetrievalConfig:
    """Configuration for entity-based retrieval."""
    entities: List[str]
    entity_boost: float = 1.5
    cross_reference: bool = True
    max_results_per_entity: int = 5
    entity_expansion: bool = True

class EntityRetrievalStrategy:
    """Entity-focused retrieval strategy."""
    
    def __init__(self, retrieval_service: AdvancedRetrievalService):
        self.retrieval_service = retrieval_service
        self.name = "entity_based"
        
        # Entity categories for environmental consulting
        self.entity_categories = {
            "mining_sites": ["Collahuasi", "Escondida", "Los Pelambres", "Chuquicamata"],
            "regions": ["Atacama", "Antofagasta", "Tarapacá", "Coquimbo"],
            "environmental_elements": ["water", "air", "soil", "biodiversity", "ecosystem"],
            "regulatory_bodies": ["SEA", "SMA", "DGA", "CCHEN", "SISS"],
            "project_types": ["mining", "processing", "waste management", "water treatment"]
        }
    
    async def retrieve(self, query: str, config: EntityRetrievalConfig) -> RetrievalResult:
        """Retrieve documents based on entity analysis."""
        try:
            # Extract and categorize entities
            categorized_entities = self._categorize_entities(config.entities)
            
            # Retrieve documents for each entity category
            all_results = []
            entity_results = {}
            
            for category, entities in categorized_entities.items():
                if entities:
                    category_results = await self._retrieve_for_entities(
                        query, entities, category, config
                    )
                    entity_results[category] = category_results
                    all_results.extend(category_results)
            
            # Cross-reference entities if enabled
            if config.cross_reference and len(categorized_entities) > 1:
                cross_ref_results = await self._cross_reference_entities(
                    query, categorized_entities, config
                )
                all_results.extend(cross_ref_results)
            
            # Merge and deduplicate results
            merged_results = self._merge_and_deduplicate(all_results)
            
            # Apply entity-based scoring
            scored_results = self._apply_entity_scoring(merged_results, categorized_entities, config)
            
            logger.info(f"Entity-based retrieval completed: {len(scored_results)} results")
            
            return RetrievalResult(
                query=query,
                results=scored_results,
                total_results=len(scored_results),
                processing_time=0.0,  # Would be calculated in real implementation
                query_vector=[],  # Would be generated
                metadata={
                    "entity_categories": list(categorized_entities.keys()),
                    "entities_found": {k: len(v) for k, v in categorized_entities.items()},
                    "cross_reference_applied": config.cross_reference,
                    "retrieval_strategy": "entity_based"
                }
            )
            
        except Exception as e:
            logger.error(f"Entity-based retrieval failed: {e}")
            return RetrievalResult(
                query=query,
                results=[],
                total_results=0,
                processing_time=0.0,
                query_vector=[],
                metadata={"error": str(e)}
            )
    
    def _categorize_entities(self, entities: List[str]) -> Dict[str, List[str]]:
        """Categorize entities by type."""
        categorized = {category: [] for category in self.entity_categories.keys()}
        
        for entity in entities:
            entity_lower = entity.lower()
            
            # Check each category
            for category, category_entities in self.entity_categories.items():
                for cat_entity in category_entities:
                    if cat_entity.lower() in entity_lower or entity_lower in cat_entity.lower():
                        categorized[category].append(entity)
                        break
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    async def _retrieve_for_entities(self, query: str, entities: List[str], 
                                   category: str, config: EntityRetrievalConfig) -> List:
        """Retrieve documents for specific entities."""
        results = []
        
        for entity in entities:
            try:
                # Create entity-specific query
                entity_query = f"{query} {entity}"
                
                # Build entity-specific filters
                filters = self._build_entity_filters(entity, category)
                
                request = RetrievalRequest(
                    query=entity_query,
                    filters=filters,
                    top_k=config.max_results_per_entity,
                    quality_threshold=0.5
                )
                
                result = await self.retrieval_service.retrieve_documents(request)
                
                # Mark results with entity information
                for search_result in result.results:
                    search_result.metadata = search_result.metadata or {}
                    search_result.metadata["matched_entity"] = entity
                    search_result.metadata["entity_category"] = category
                
                results.extend(result.results)
                
            except Exception as e:
                logger.warning(f"Failed to retrieve for entity {entity}: {e}")
                continue
        
        return results
    
    def _build_entity_filters(self, entity: str, category: str) -> Dict[str, Any]:
        """Build filters based on entity and category."""
        filters = {}
        
        if category == "mining_sites":
            filters["site_name"] = entity
        elif category == "regions":
            filters["region"] = entity
        elif category == "environmental_elements":
            filters["environmental_element"] = entity
        elif category == "regulatory_bodies":
            filters["regulatory_body"] = entity
        
        return filters
    
    async def _cross_reference_entities(self, query: str, categorized_entities: Dict[str, List[str]], 
                                      config: EntityRetrievalConfig) -> List:
        """Cross-reference entities to find related documents."""
        results = []
        
        # Create cross-reference queries
        cross_ref_queries = []
        
        # Combine entities from different categories
        categories = list(categorized_entities.keys())
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat1, cat2 = categories[i], categories[j]
                entities1 = categorized_entities[cat1]
                entities2 = categorized_entities[cat2]
                
                for entity1 in entities1:
                    for entity2 in entities2:
                        cross_ref_query = f"{query} {entity1} {entity2}"
                        cross_ref_queries.append(cross_ref_query)
        
        # Limit cross-reference queries to avoid too many requests
        cross_ref_queries = cross_ref_queries[:10]
        
        for cross_ref_query in cross_ref_queries:
            try:
                request = RetrievalRequest(
                    query=cross_ref_query,
                    top_k=3,  # Fewer results for cross-references
                    quality_threshold=0.4
                )
                
                result = await self.retrieval_service.retrieve_documents(request)
                
                # Mark as cross-reference
                for search_result in result.results:
                    search_result.metadata = search_result.metadata or {}
                    search_result.metadata["cross_reference"] = True
                
                results.extend(result.results)
                
            except Exception as e:
                logger.warning(f"Cross-reference query failed: {e}")
                continue
        
        return results
    
    def _merge_and_deduplicate(self, all_results: List) -> List:
        """Merge and deduplicate results."""
        seen_chunks = set()
        unique_results = []
        
        for result in all_results:
            if result.chunk_id not in seen_chunks:
                unique_results.append(result)
                seen_chunks.add(result.chunk_id)
        
        return unique_results
    
    def _apply_entity_scoring(self, results: List, categorized_entities: Dict[str, List[str]], 
                            config: EntityRetrievalConfig) -> List:
        """Apply entity-based scoring to results."""
        scored_results = []
        
        for result in results:
            # Calculate entity relevance score
            entity_score = self._calculate_entity_relevance(result, categorized_entities)
            
            # Apply entity boost
            boosted_score = result.score * (1 + (entity_score * config.entity_boost))
            
            # Update result
            result.score = min(boosted_score, 1.0)  # Cap at 1.0
            result.metadata = result.metadata or {}
            result.metadata["entity_relevance_score"] = entity_score
            result.metadata["entity_boost_applied"] = config.entity_boost
            
            scored_results.append(result)
        
        # Sort by boosted score
        scored_results.sort(key=lambda x: x.score, reverse=True)
        
        return scored_results
    
    def _calculate_entity_relevance(self, result, categorized_entities: Dict[str, List[str]]) -> float:
        """Calculate entity relevance score for a result."""
        try:
            content = result.content.lower()
            metadata = result.metadata or {}
            
            relevance_score = 0.0
            total_entities = sum(len(entities) for entities in categorized_entities.values())
            
            if total_entities == 0:
                return 0.0
            
            # Check for entity mentions in content
            for category, entities in categorized_entities.items():
                for entity in entities:
                    if entity.lower() in content:
                        relevance_score += 1.0
            
            # Check for entity mentions in metadata
            matched_entity = metadata.get("matched_entity", "")
            if matched_entity:
                relevance_score += 0.5
            
            # Normalize score
            return min(relevance_score / total_entities, 1.0)
            
        except Exception as e:
            logger.warning(f"Failed to calculate entity relevance: {e}")
            return 0.0
    
    async def get_entity_context(self, entity: str, category: str) -> RetrievalResult:
        """Get comprehensive context for a specific entity."""
        # Get all documents related to the entity
        request = RetrievalRequest(
            query=entity,
            filters=self._build_entity_filters(entity, category),
            top_k=20,
            quality_threshold=0.3
        )
        
        result = await self.retrieval_service.retrieve_documents(request)
        
        # Organize by document
        documents = {}
        for search_result in result.results:
            doc_name = search_result.metadata.get("document_blob", "unknown")
            if doc_name not in documents:
                documents[doc_name] = []
            documents[doc_name].append(search_result)
        
        # Sort by document and page
        organized_results = []
        for doc_name in sorted(documents.keys()):
            doc_results = sorted(documents[doc_name], 
                               key=lambda x: x.metadata.get("page_number", 0))
            organized_results.extend(doc_results)
        
        result.results = organized_results
        result.metadata = result.metadata or {}
        result.metadata["entity_context"] = {
            "entity": entity,
            "category": category,
            "documents_found": len(documents),
            "total_chunks": len(organized_results)
        }
        
        return result
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            "name": self.name,
            "description": "Entity-focused retrieval with cross-referencing",
            "capabilities": [
                "Entity categorization",
                "Cross-entity referencing",
                "Entity-based scoring",
                "Comprehensive entity context"
            ],
            "best_for": [
                "Site-specific queries",
                "Project-focused research",
                "Entity relationship analysis",
                "Location-based searches"
            ],
            "entity_categories": list(self.entity_categories.keys())
        }

