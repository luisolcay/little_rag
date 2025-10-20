"""
Common interfaces and utilities for the LLM layer.
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime
from .models import ModelName

class LLMConfig:
    """Configuration for LLM services."""
    
    @staticmethod
    def get_model_costs_per_1k_tokens() -> Dict[str, Dict[str, float]]:
        """Get current model pricing per 1K tokens."""
        return {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006}
        }
    
    @staticmethod
    def get_default_temperature() -> float:
        """Get default temperature from environment."""
        return float(os.getenv("LLM_DEFAULT_TEMPERATURE", 0.1))
    
    @staticmethod
    def get_max_tokens() -> int:
        """Get max tokens from environment."""
        return int(os.getenv("LLM_MAX_TOKENS", 4096))
    
    @staticmethod
    def is_streaming_enabled() -> bool:
        """Check if streaming is enabled."""
        return os.getenv("ENABLE_STREAMING", "true").lower() == "true"

class CostCalculator:
    """Calculate costs for LLM usage."""
    
    @staticmethod
    def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        pricing = LLMConfig.get_model_costs_per_1k_tokens()
        
        if model not in pricing:
            return 0.0
        
        model_pricing = pricing[model]
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    @staticmethod
    def suggest_model(query: str, complexity_threshold: float = 0.7) -> str:
        """Suggest optimal model based on query complexity and cost."""
        # Simple complexity estimation based on query length and keywords
        complexity_score = 0
        
        # Length factor
        if len(query) > 200:
            complexity_score += 0.3
        elif len(query) > 100:
            complexity_score += 0.1
        
        # Keyword complexity
        complex_keywords = ["analysis", "assessment", "evaluation", "comprehensive", "detailed"]
        for keyword in complex_keywords:
            if keyword in query.lower():
                complexity_score += 0.2
        
        # Technical terms
        technical_terms = ["regulatory", "compliance", "environmental", "technical"]
        for term in technical_terms:
            if term in query.lower():
                complexity_score += 0.1
        
        # Suggest model based on complexity
        if complexity_score >= complexity_threshold:
            return "gpt-4o"  # Use more capable model for complex queries
        else:
            return "gpt-4o-mini"  # Use cheaper model for simple queries
