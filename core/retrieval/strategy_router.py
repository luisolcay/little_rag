"""
Intelligent retrieval routing based on query analysis and classification.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"
    ENTITY_BASED = "entity_based"
    HYBRID = "hybrid"
    STANDARD = "standard"

class StrategyRouter:
    """Simple strategy router for retrieval."""
    
    def __init__(self):
        self.strategies = [
            RetrievalStrategy.TEMPORAL,
            RetrievalStrategy.HIERARCHICAL,
            RetrievalStrategy.ENTITY_BASED,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.STANDARD
        ]
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies."""
        return [strategy.value for strategy in self.strategies]
    
    def route_query(self, query: str) -> RetrievalStrategy:
        """Route query to appropriate strategy."""
        # Simple routing logic
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["fecha", "tiempo", "reciente", "último"]):
            return RetrievalStrategy.TEMPORAL
        elif any(word in query_lower for word in ["jerarquía", "nivel", "categoría"]):
            return RetrievalStrategy.HIERARCHICAL
        elif any(word in query_lower for word in ["entidad", "empresa", "persona"]):
            return RetrievalStrategy.ENTITY_BASED
        else:
            return RetrievalStrategy.HYBRID

class QueryComplexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

@dataclass
class QueryAnalysis:
    """Analysis result of a query."""
    query: str
    query_type: str
    complexity: QueryComplexity
    entities: List[str]
    temporal_indicators: List[str]
    regulatory_keywords: List[str]
    technical_keywords: List[str]
    confidence: float
    suggested_strategy: RetrievalStrategy
    reasoning: str

class QueryClassifier:
    """Intelligent query classification for retrieval strategy selection."""
    
    def __init__(self):
        # Environmental consulting keywords
        self.regulatory_keywords = [
            "compliance", "regulation", "permit", "license", "approval",
            "SEA", "SMA", "DGA", "CCHEN", "SISS", "DS", "Ley",
            "environmental impact", "EIA", "DIA", "RCA"
        ]
        
        self.technical_keywords = [
            "analysis", "assessment", "study", "evaluation", "monitoring",
            "measurement", "sampling", "testing", "modeling", "simulation",
            "design", "engineering", "technical", "methodology"
        ]
        
        self.temporal_patterns = [
            r"\b(?:recent|latest|current|new|updated|modified)\b",
            r"\b(?:since|from|after|before|during)\s+\d{4}\b",
            r"\b(?:last|past|previous)\s+(?:year|month|week|decade)\b",
            r"\b(?:historical|archive|old|past)\b",
            r"\b(?:deadline|due|expires?|valid|expired)\b"
        ]
        
        self.entity_patterns = [
            r"\b[A-Z][a-z]+\s+(?:mine|mining|project|site|facility|plant)\b",
            r"\b(?:Collahuasi|Escondida|Los Pelambres|Chuquicamata)\b",
            r"\b(?:Atacama|Antofagasta|Tarapacá|Coquimbo)\b",
            r"\b(?:water|air|soil|biodiversity|ecosystem)\b"
        ]
        
        # Strategy weights based on query characteristics
        self.strategy_weights = {
            RetrievalStrategy.TEMPORAL: {
                "temporal_indicators": 0.8,
                "regulatory_keywords": 0.3,
                "complexity": 0.2
            },
            RetrievalStrategy.HIERARCHICAL: {
                "technical_keywords": 0.6,
                "complexity": 0.7,
                "entity_count": 0.4
            },
            RetrievalStrategy.ENTITY_BASED: {
                "entity_count": 0.8,
                "regulatory_keywords": 0.4,
                "complexity": 0.3
            },
            RetrievalStrategy.HYBRID: {
                "complexity": 0.9,
                "multiple_categories": 0.7,
                "confidence_threshold": 0.6
            }
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query and determine optimal retrieval strategy."""
        try:
            # Extract features
            entities = self._extract_entities(query)
            temporal_indicators = self._extract_temporal_indicators(query)
            regulatory_keywords = self._extract_keywords(query, self.regulatory_keywords)
            technical_keywords = self._extract_keywords(query, self.technical_keywords)
            
            # Determine complexity
            complexity = self._assess_complexity(query, entities, regulatory_keywords, technical_keywords)
            
            # Determine query type
            query_type = self._determine_query_type(regulatory_keywords, technical_keywords, temporal_indicators)
            
            # Select strategy
            strategy, confidence, reasoning = self._select_strategy(
                query, entities, temporal_indicators, regulatory_keywords, 
                technical_keywords, complexity, query_type
            )
            
            return QueryAnalysis(
                query=query,
                query_type=query_type,
                complexity=complexity,
                entities=entities,
                temporal_indicators=temporal_indicators,
                regulatory_keywords=regulatory_keywords,
                technical_keywords=technical_keywords,
                confidence=confidence,
                suggested_strategy=strategy,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Fallback to standard strategy
            return QueryAnalysis(
                query=query,
                query_type="general",
                complexity=QueryComplexity.MODERATE,
                entities=[],
                temporal_indicators=[],
                regulatory_keywords=[],
                technical_keywords=[],
                confidence=0.5,
                suggested_strategy=RetrievalStrategy.STANDARD,
                reasoning="Fallback due to analysis error"
            )
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        entities = []
        
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        # Extract capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend([word for word in capitalized_words if len(word) > 3])
        
        return list(set(entities))
    
    def _extract_temporal_indicators(self, query: str) -> List[str]:
        """Extract temporal indicators from query."""
        indicators = []
        
        for pattern in self.temporal_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            indicators.extend(matches)
        
        return list(set(indicators))
    
    def _extract_keywords(self, query: str, keywords: List[str]) -> List[str]:
        """Extract matching keywords from query."""
        found_keywords = []
        query_lower = query.lower()
        
        for keyword in keywords:
            if keyword.lower() in query_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _assess_complexity(self, query: str, entities: List[str], 
                          regulatory_keywords: List[str], 
                          technical_keywords: List[str]) -> QueryComplexity:
        """Assess query complexity."""
        score = 0
        
        # Length factor
        if len(query) > 200:
            score += 2
        elif len(query) > 100:
            score += 1
        
        # Entity count
        score += min(len(entities), 3)
        
        # Keyword categories
        score += len(regulatory_keywords)
        score += len(technical_keywords)
        
        # Question complexity indicators
        complex_indicators = ["analysis", "assessment", "evaluation", "compare", "analyze"]
        for indicator in complex_indicators:
            if indicator in query.lower():
                score += 1
        
        if score <= 2:
            return QueryComplexity.SIMPLE
        elif score <= 5:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX
    
    def _determine_query_type(self, regulatory_keywords: List[str], 
                             technical_keywords: List[str], 
                             temporal_indicators: List[str]) -> str:
        """Determine the type of query."""
        if regulatory_keywords and temporal_indicators:
            return "regulatory_temporal"
        elif regulatory_keywords:
            return "regulatory"
        elif technical_keywords:
            return "technical"
        elif temporal_indicators:
            return "temporal"
        else:
            return "general"
    
    def _select_strategy(self, query: str, entities: List[str], 
                        temporal_indicators: List[str], 
                        regulatory_keywords: List[str], 
                        technical_keywords: List[str], 
                        complexity: QueryComplexity, 
                        query_type: str) -> Tuple[RetrievalStrategy, float, str]:
        """Select optimal retrieval strategy."""
        
        # Calculate scores for each strategy
        scores = {}
        
        # Temporal strategy score
        temporal_score = 0
        if temporal_indicators:
            temporal_score += 0.8
        if "recent" in query.lower() or "latest" in query.lower():
            temporal_score += 0.6
        if regulatory_keywords and temporal_indicators:
            temporal_score += 0.4
        scores[RetrievalStrategy.TEMPORAL] = temporal_score
        
        # Hierarchical strategy score
        hierarchical_score = 0
        if complexity == QueryComplexity.COMPLEX:
            hierarchical_score += 0.7
        if technical_keywords:
            hierarchical_score += 0.6
        if len(entities) > 2:
            hierarchical_score += 0.4
        scores[RetrievalStrategy.HIERARCHICAL] = hierarchical_score
        
        # Entity-based strategy score
        entity_score = 0
        if len(entities) > 0:
            entity_score += 0.8
        if regulatory_keywords:
            entity_score += 0.4
        if complexity == QueryComplexity.MODERATE:
            entity_score += 0.3
        scores[RetrievalStrategy.ENTITY_BASED] = entity_score
        
        # Hybrid strategy score
        hybrid_score = 0
        if complexity == QueryComplexity.COMPLEX:
            hybrid_score += 0.9
        if len([k for k in [regulatory_keywords, technical_keywords, temporal_indicators] if k]) > 1:
            hybrid_score += 0.7
        if len(entities) > 1 and temporal_indicators:
            hybrid_score += 0.6
        scores[RetrievalStrategy.HYBRID] = hybrid_score
        
        # Standard strategy (fallback)
        scores[RetrievalStrategy.STANDARD] = 0.1
        
        # Select strategy with highest score
        best_strategy = max(scores, key=scores.get)
        confidence = scores[best_strategy]
        
        # Generate reasoning
        reasoning_parts = []
        if temporal_indicators:
            reasoning_parts.append(f"Temporal indicators found: {temporal_indicators}")
        if entities:
            reasoning_parts.append(f"Entities identified: {entities}")
        if regulatory_keywords:
            reasoning_parts.append(f"Regulatory keywords: {regulatory_keywords}")
        if technical_keywords:
            reasoning_parts.append(f"Technical keywords: {technical_keywords}")
        reasoning_parts.append(f"Complexity: {complexity.value}")
        reasoning_parts.append(f"Strategy score: {confidence:.2f}")
        
        reasoning = "; ".join(reasoning_parts)
        
        return best_strategy, confidence, reasoning
    
    def get_strategy_parameters(self, strategy: RetrievalStrategy, 
                               analysis: QueryAnalysis) -> Dict[str, Any]:
        """Get strategy-specific parameters based on query analysis."""
        params = {
            "strategy": strategy,
            "query_type": analysis.query_type,
            "confidence": analysis.confidence
        }
        
        if strategy == RetrievalStrategy.TEMPORAL:
            params.update({
                "temporal_focus": "recent" if "recent" in analysis.query.lower() else "all",
                "date_range": self._extract_date_range(analysis.query),
                "priority_recent": True
            })
        
        elif strategy == RetrievalStrategy.HIERARCHICAL:
            params.update({
                "expand_levels": 2 if analysis.complexity == QueryComplexity.COMPLEX else 1,
                "max_context_tokens": 6000 if analysis.complexity == QueryComplexity.COMPLEX else 4000,
                "breadth_first": True
            })
        
        elif strategy == RetrievalStrategy.ENTITY_BASED:
            params.update({
                "entities": analysis.entities,
                "entity_boost": 1.5,
                "cross_reference": True
            })
        
        elif strategy == RetrievalStrategy.HYBRID:
            params.update({
                "temporal_weight": 0.3 if analysis.temporal_indicators else 0.1,
                "entity_weight": 0.4 if analysis.entities else 0.1,
                "regulatory_weight": 0.3 if analysis.regulatory_keywords else 0.1,
                "fusion_method": "weighted_average"
            })
        
        return params
    
    def _extract_date_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """Extract date range from query."""
        # Simple date extraction - in production, use more sophisticated NLP
        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, query)
        
        if len(years) >= 2:
            start_year = min(int(y) for y in years)
            end_year = max(int(y) for y in years)
            return (
                datetime(start_year, 1, 1),
                datetime(end_year, 12, 31)
            )
        elif len(years) == 1:
            year = int(years[0])
            return (
                datetime(year, 1, 1),
                datetime(year, 12, 31)
            )
        
        return None

# Global instance
query_classifier = QueryClassifier()

