"""
Mock simple para probar las estrategias de recuperación sin servicios de Azure.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Mock classes para las pruebas
class MockRetrievalService:
    """Mock del servicio de recuperación para pruebas."""
    pass

class MockRetrievalRequest:
    """Mock de la solicitud de recuperación para pruebas."""
    pass

class MockRetrievalResult:
    """Mock del resultado de recuperación para pruebas."""
    pass

@dataclass
class TemporalRetrievalConfig:
    """Configuration for temporal retrieval."""
    priority_recent: bool = True
    date_range: Optional[tuple] = None
    temporal_focus: str = "recent"  # "recent", "historical", "all"
    recency_weight: float = 0.7
    max_results: int = 10

class TemporalRetrievalStrategy:
    """Time-aware retrieval strategy prioritizing recent documents."""
    
    def __init__(self, retrieval_service=None):
        self.retrieval_service = retrieval_service
        self.name = "temporal"
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information."""
        return {
            "name": self.name,
            "description": "Time-aware retrieval prioritizing recent documents",
            "capabilities": [
                "Recent document prioritization",
                "Historical context retrieval",
                "Temporal scoring",
                "Date range filtering"
            ],
            "best_for": [
                "Regulatory updates",
                "Recent compliance changes",
                "Historical comparisons",
                "Time-sensitive queries"
            ]
        }

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
    
    def __init__(self, retrieval_service=None):
        self.retrieval_service = retrieval_service
        self.name = "hierarchical"
    
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
    
    def __init__(self, retrieval_service=None):
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
    
    def __init__(self, retrieval_service=None):
        self.retrieval_service = retrieval_service
        self.name = "hybrid"
        
        # Initialize sub-strategies
        self.temporal_strategy = TemporalRetrievalStrategy(retrieval_service)
        self.hierarchical_strategy = HierarchicalRetrievalStrategy(retrieval_service)
        self.entity_strategy = EntityRetrievalStrategy(retrieval_service)
    
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

