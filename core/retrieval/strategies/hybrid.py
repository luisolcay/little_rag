"""
Hybrid retrieval strategy combining multiple approaches with result fusion.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..vector.retrieval_service import AdvancedRetrievalService, RetrievalRequest, RetrievalResult
from .temporal import TemporalRetrievalStrategy, TemporalRetrievalConfig
from .hierarchical import HierarchicalRetrievalStrategy, HierarchicalRetrievalConfig
from .entity_based import EntityRetrievalStrategy, EntityRetrievalConfig

logger = logging.getLogger(__name__)

@dataclass
class HybridRetrievalConfig:
    """Configuration for hybrid retrieval."""
    temporal_weight: float = 0.3
    entity_weight: float = 0.4
    hierarchical_weight: float = 0.3
    fusion_method: str = "weighted_average"  # "weighted_average", "reciprocal_rank", "borda_count"
    max_results: int = 15
    min_strategy_results: int = 3

class HybridRetrievalStrategy:
    """Hybrid retrieval combining multiple strategies with intelligent fusion."""
    
    def __init__(self, retrieval_service: AdvancedRetrievalService):
        self.retrieval_service = retrieval_service
        self.name = "hybrid"
        
        # Initialize sub-strategies
        self.temporal_strategy = TemporalRetrievalStrategy(retrieval_service)
        self.hierarchical_strategy = HierarchicalRetrievalStrategy(retrieval_service)
        self.entity_strategy = EntityRetrievalStrategy(retrieval_service)
    
    async def retrieve(self, query: str, config: HybridRetrievalConfig, 
                      entities: Optional[List[str]] = None) -> RetrievalResult:
        """Retrieve documents using hybrid strategy with result fusion."""
        try:
            start_time = datetime.now()
            
            # Run strategies in parallel
            strategy_tasks = []
            
            # Temporal strategy
            temporal_config = TemporalRetrievalConfig(
                priority_recent=True,
                max_results=config.max_results
            )
            strategy_tasks.append(
                self.temporal_strategy.retrieve(query, temporal_config)
            )
            
            # Hierarchical strategy
            hierarchical_config = HierarchicalRetrievalConfig(
                expand_levels=2,
                max_context_tokens=4000,
                max_results_per_level=config.max_results // 2
            )
            strategy_tasks.append(
                self.hierarchical_strategy.retrieve(query, hierarchical_config)
            )
            
            # Entity-based strategy (if entities provided)
            if entities:
                entity_config = EntityRetrievalConfig(
                    entities=entities,
                    entity_boost=1.5,
                    cross_reference=True,
                    max_results_per_entity=config.max_results // 3
                )
                strategy_tasks.append(
                    self.entity_strategy.retrieve(query, entity_config)
                )
            
            # Wait for all strategies to complete
            strategy_results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
            
            # Process results
            valid_results = []
            strategy_names = ["temporal", "hierarchical"]
            if entities:
                strategy_names.append("entity_based")
            
            for i, result in enumerate(strategy_results):
                if isinstance(result, Exception):
                    logger.warning(f"Strategy {strategy_names[i]} failed: {result}")
                    continue
                
                if result.results:
                    valid_results.append((strategy_names[i], result))
            
            if not valid_results:
                logger.warning("No valid results from any strategy")
                return RetrievalResult(
                    query=query,
                    results=[],
                    total_results=0,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    query_vector=[],
                    metadata={"error": "All strategies failed"}
                )
            
            # Fuse results
            fused_result = self._fuse_results(valid_results, config)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            fused_result.processing_time = processing_time
            
            logger.info(f"Hybrid retrieval completed: {len(fused_result.results)} results in {processing_time:.3f}s")
            
            return fused_result
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return RetrievalResult(
                query=query,
                results=[],
                total_results=0,
                processing_time=0.0,
                query_vector=[],
                metadata={"error": str(e)}
            )
    
    def _fuse_results(self, strategy_results: List[Tuple[str, RetrievalResult]], 
                     config: HybridRetrievalConfig) -> RetrievalResult:
        """Fuse results from multiple strategies."""
        if config.fusion_method == "weighted_average":
            return self._weighted_average_fusion(strategy_results, config)
        elif config.fusion_method == "reciprocal_rank":
            return self._reciprocal_rank_fusion(strategy_results, config)
        elif config.fusion_method == "borda_count":
            return self._borda_count_fusion(strategy_results, config)
        else:
            logger.warning(f"Unknown fusion method: {config.fusion_method}, using weighted average")
            return self._weighted_average_fusion(strategy_results, config)
    
    def _weighted_average_fusion(self, strategy_results: List[Tuple[str, RetrievalResult]], 
                               config: HybridRetrievalConfig) -> RetrievalResult:
        """Fuse results using weighted average scoring."""
        chunk_scores = {}
        chunk_results = {}
        
        # Collect scores from all strategies
        for strategy_name, result in strategy_results:
            weight = self._get_strategy_weight(strategy_name, config)
            
            for i, search_result in enumerate(result.results):
                chunk_id = search_result.chunk_id
                
                # Calculate weighted score
                weighted_score = search_result.score * weight
                
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = 0.0
                    chunk_results[chunk_id] = search_result
                
                chunk_scores[chunk_id] += weighted_score
                
                # Merge metadata
                if chunk_id in chunk_results:
                    chunk_results[chunk_id].metadata = chunk_results[chunk_id].metadata or {}
                    strategy_metadata = search_result.metadata or {}
                    chunk_results[chunk_id].metadata[f"{strategy_name}_score"] = search_result.score
                    chunk_results[chunk_id].metadata[f"{strategy_name}_rank"] = i + 1
        
        # Sort by fused score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create fused results
        fused_results = []
        for chunk_id, fused_score in sorted_chunks[:config.max_results]:
            result = chunk_results[chunk_id]
            result.score = fused_score
            result.metadata = result.metadata or {}
            result.metadata["fused_score"] = fused_score
            result.metadata["fusion_method"] = "weighted_average"
            fused_results.append(result)
        
        # Create result
        return RetrievalResult(
            query=strategy_results[0][1].query,
            results=fused_results,
            total_results=len(fused_results),
            processing_time=0.0,  # Will be set by caller
            query_vector=[],
            metadata={
                "fusion_method": "weighted_average",
                "strategies_used": [name for name, _ in strategy_results],
                "strategy_weights": {
                    name: self._get_strategy_weight(name, config) 
                    for name, _ in strategy_results
                },
                "retrieval_strategy": "hybrid"
            }
        )
    
    def _reciprocal_rank_fusion(self, strategy_results: List[Tuple[str, RetrievalResult]], 
                              config: HybridRetrievalConfig) -> RetrievalResult:
        """Fuse results using reciprocal rank fusion."""
        chunk_scores = {}
        chunk_results = {}
        
        for strategy_name, result in strategy_results:
            for i, search_result in enumerate(result.results):
                chunk_id = search_result.chunk_id
                rank = i + 1
                reciprocal_score = 1.0 / rank
                
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = 0.0
                    chunk_results[chunk_id] = search_result
                
                chunk_scores[chunk_id] += reciprocal_score
                
                # Merge metadata
                if chunk_id in chunk_results:
                    chunk_results[chunk_id].metadata = chunk_results[chunk_id].metadata or {}
                    chunk_results[chunk_id].metadata[f"{strategy_name}_rank"] = rank
        
        # Sort by reciprocal rank score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create fused results
        fused_results = []
        for chunk_id, rr_score in sorted_chunks[:config.max_results]:
            result = chunk_results[chunk_id]
            result.score = rr_score
            result.metadata = result.metadata or {}
            result.metadata["reciprocal_rank_score"] = rr_score
            result.metadata["fusion_method"] = "reciprocal_rank"
            fused_results.append(result)
        
        return RetrievalResult(
            query=strategy_results[0][1].query,
            results=fused_results,
            total_results=len(fused_results),
            processing_time=0.0,
            query_vector=[],
            metadata={
                "fusion_method": "reciprocal_rank",
                "strategies_used": [name for name, _ in strategy_results],
                "retrieval_strategy": "hybrid"
            }
        )
    
    def _borda_count_fusion(self, strategy_results: List[Tuple[str, RetrievalResult]], 
                          config: HybridRetrievalConfig) -> RetrievalResult:
        """Fuse results using Borda count method."""
        chunk_scores = {}
        chunk_results = {}
        
        for strategy_name, result in strategy_results:
            max_rank = len(result.results)
            
            for i, search_result in enumerate(result.results):
                chunk_id = search_result.chunk_id
                borda_score = max_rank - i
                
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = 0
                    chunk_results[chunk_id] = search_result
                
                chunk_scores[chunk_id] += borda_score
                
                # Merge metadata
                if chunk_id in chunk_results:
                    chunk_results[chunk_id].metadata = chunk_results[chunk_id].metadata or {}
                    chunk_results[chunk_id].metadata[f"{strategy_name}_borda_score"] = borda_score
        
        # Sort by Borda count
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create fused results
        fused_results = []
        for chunk_id, borda_score in sorted_chunks[:config.max_results]:
            result = chunk_results[chunk_id]
            result.score = borda_score / sum(len(r.results) for _, r in strategy_results)  # Normalize
            result.metadata = result.metadata or {}
            result.metadata["borda_count"] = borda_score
            result.metadata["fusion_method"] = "borda_count"
            fused_results.append(result)
        
        return RetrievalResult(
            query=strategy_results[0][1].query,
            results=fused_results,
            total_results=len(fused_results),
            processing_time=0.0,
            query_vector=[],
            metadata={
                "fusion_method": "borda_count",
                "strategies_used": [name for name, _ in strategy_results],
                "retrieval_strategy": "hybrid"
            }
        )
    
    def _get_strategy_weight(self, strategy_name: str, config: HybridRetrievalConfig) -> float:
        """Get weight for a specific strategy."""
        weights = {
            "temporal": config.temporal_weight,
            "hierarchical": config.hierarchical_weight,
            "entity_based": config.entity_weight
        }
        return weights.get(strategy_name, 0.1)
    
    async def adaptive_retrieve(self, query: str, entities: Optional[List[str]] = None) -> RetrievalResult:
        """Adaptive retrieval that adjusts strategy weights based on query characteristics."""
        # Analyze query to determine optimal weights
        if entities and len(entities) > 2:
            # Entity-heavy query
            config = HybridRetrievalConfig(
                temporal_weight=0.2,
                entity_weight=0.6,
                hierarchical_weight=0.2
            )
        elif "recent" in query.lower() or "latest" in query.lower():
            # Temporal query
            config = HybridRetrievalConfig(
                temporal_weight=0.6,
                entity_weight=0.2,
                hierarchical_weight=0.2
            )
        elif "analysis" in query.lower() or "comprehensive" in query.lower():
            # Complex analysis query
            config = HybridRetrievalConfig(
                temporal_weight=0.2,
                entity_weight=0.3,
                hierarchical_weight=0.5
            )
        else:
            # Balanced query
            config = HybridRetrievalConfig()
        
        return await self.retrieve(query, config, entities)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            "name": self.name,
            "description": "Hybrid retrieval combining multiple strategies",
            "capabilities": [
                "Multi-strategy parallel execution",
                "Intelligent result fusion",
                "Adaptive weight adjustment",
                "Comprehensive coverage"
            ],
            "best_for": [
                "Complex queries requiring multiple perspectives",
                "Comprehensive research tasks",
                "High-recall requirements",
                "Balanced precision and recall"
            ],
            "fusion_methods": ["weighted_average", "reciprocal_rank", "borda_count"],
            "sub_strategies": ["temporal", "hierarchical", "entity_based"]
        }

