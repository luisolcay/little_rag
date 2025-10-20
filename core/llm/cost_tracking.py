"""
Cost tracking and optimization utilities for the LLM layer.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from core.memory.cosmos_store import cosmos_client
from .config import CostCalculator

logger = logging.getLogger(__name__)

@dataclass
class CostMetrics:
    """Cost metrics for tracking."""
    session_id: str
    model: str
    tokens_used: int
    cost_usd: float
    timestamp: datetime
    query_type: str
    success: bool

class CostTracker:
    """Track and analyze LLM costs."""
    
    def __init__(self):
        self.daily_budgets = {
            "gpt-4o": 50.0,  # $50/day
            "gpt-4o-mini": 20.0  # $20/day
        }
        self.monthly_budgets = {
            "gpt-4o": 1000.0,  # $1000/month
            "gpt-4o-mini": 500.0  # $500/month
        }
    
    async def track_request_cost(self, session_id: str, model: str, tokens_used: int, 
                               cost_usd: float, query_type: str, success: bool = True):
        """Track cost for a single request."""
        try:
            metric = CostMetrics(
                session_id=session_id,
                model=model,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                timestamp=datetime.now(),
                query_type=query_type,
                success=success
            )
            
            # Store in Cosmos DB
            await cosmos_client.create_item("llm_analytics", {
                "id": f"cost_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "session_id": session_id,
                "model": model,
                "tokens_used": tokens_used,
                "cost_usd": cost_usd,
                "query_type": query_type,
                "success": success,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Cost tracked: ${cost_usd:.6f} for {model} ({tokens_used} tokens)")
            
        except Exception as e:
            logger.error(f"Failed to track cost: {e}")
    
    async def get_daily_costs(self, date: Optional[datetime] = None) -> Dict[str, float]:
        """Get daily costs by model."""
        if not date:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        
        try:
            query = """
            SELECT c.model, SUM(c.cost_usd) as total_cost
            FROM c
            WHERE c.date = @date
            GROUP BY c.model
            """
            
            parameters = [{"name": "@date", "value": date_str}]
            
            results = await cosmos_client.query_items("llm_analytics", query, parameters)
            
            daily_costs = {}
            for result in results:
                daily_costs[result["model"]] = result["total_cost"]
            
            return daily_costs
            
        except Exception as e:
            logger.error(f"Failed to get daily costs: {e}")
            return {}
    
    async def get_monthly_costs(self, year: int, month: int) -> Dict[str, float]:
        """Get monthly costs by model."""
        try:
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)
            
            query = """
            SELECT c.model, SUM(c.cost_usd) as total_cost
            FROM c
            WHERE c.timestamp >= @start_date AND c.timestamp < @end_date
            GROUP BY c.model
            """
            
            parameters = [
                {"name": "@start_date", "value": start_date.isoformat()},
                {"name": "@end_date", "value": end_date.isoformat()}
            ]
            
            results = await cosmos_client.query_items("llm_analytics", query, parameters)
            
            monthly_costs = {}
            for result in results:
                monthly_costs[result["model"]] = result["total_cost"]
            
            return monthly_costs
            
        except Exception as e:
            logger.error(f"Failed to get monthly costs: {e}")
            return {}
    
    async def check_budget_alerts(self) -> List[Dict[str, Any]]:
        """Check for budget alerts."""
        alerts = []
        
        try:
            # Check daily budgets
            daily_costs = await self.get_daily_costs()
            
            for model, cost in daily_costs.items():
                daily_budget = self.daily_budgets.get(model, 0)
                if daily_budget > 0 and cost >= daily_budget * 0.8:  # 80% threshold
                    alerts.append({
                        "type": "daily_budget_warning",
                        "model": model,
                        "current_cost": cost,
                        "budget": daily_budget,
                        "percentage": (cost / daily_budget) * 100,
                        "message": f"Daily budget for {model} is {(cost / daily_budget) * 100:.1f}% used"
                    })
                
                if cost >= daily_budget:
                    alerts.append({
                        "type": "daily_budget_exceeded",
                        "model": model,
                        "current_cost": cost,
                        "budget": daily_budget,
                        "message": f"Daily budget for {model} has been exceeded!"
                    })
            
            # Check monthly budgets
            now = datetime.now()
            monthly_costs = await self.get_monthly_costs(now.year, now.month)
            
            for model, cost in monthly_costs.items():
                monthly_budget = self.monthly_budgets.get(model, 0)
                if monthly_budget > 0 and cost >= monthly_budget * 0.9:  # 90% threshold
                    alerts.append({
                        "type": "monthly_budget_warning",
                        "model": model,
                        "current_cost": cost,
                        "budget": monthly_budget,
                        "percentage": (cost / monthly_budget) * 100,
                        "message": f"Monthly budget for {model} is {(cost / monthly_budget) * 100:.1f}% used"
                    })
                
                if cost >= monthly_budget:
                    alerts.append({
                        "type": "monthly_budget_exceeded",
                        "model": model,
                        "current_cost": cost,
                        "budget": monthly_budget,
                        "message": f"Monthly budget for {model} has been exceeded!"
                    })
            
        except Exception as e:
            logger.error(f"Failed to check budget alerts: {e}")
        
        return alerts
    
    async def get_cost_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get cost summary for the last N days."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            query = """
            SELECT c.model, c.query_type, 
                   SUM(c.cost_usd) as total_cost,
                   SUM(c.tokens_used) as total_tokens,
                   COUNT(1) as request_count,
                   AVG(c.cost_usd) as avg_cost_per_request
            FROM c
            WHERE c.timestamp >= @start_date AND c.timestamp <= @end_date
            GROUP BY c.model, c.query_type
            """
            
            parameters = [
                {"name": "@start_date", "value": start_date.isoformat()},
                {"name": "@end_date", "value": end_date.isoformat()}
            ]
            
            results = await cosmos_client.query_items("llm_analytics", query, parameters)
            
            # Aggregate by model
            model_summary = {}
            total_cost = 0
            total_tokens = 0
            total_requests = 0
            
            for result in results:
                model = result["model"]
                if model not in model_summary:
                    model_summary[model] = {
                        "total_cost": 0,
                        "total_tokens": 0,
                        "request_count": 0,
                        "query_types": {}
                    }
                
                model_summary[model]["total_cost"] += result["total_cost"]
                model_summary[model]["total_tokens"] += result["total_tokens"]
                model_summary[model]["request_count"] += result["request_count"]
                model_summary[model]["query_types"][result["query_type"]] = {
                    "cost": result["total_cost"],
                    "tokens": result["total_tokens"],
                    "requests": result["request_count"],
                    "avg_cost_per_request": result["avg_cost_per_request"]
                }
                
                total_cost += result["total_cost"]
                total_tokens += result["total_tokens"]
                total_requests += result["request_count"]
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days
                },
                "summary": {
                    "total_cost": total_cost,
                    "total_tokens": total_tokens,
                    "total_requests": total_requests,
                    "avg_cost_per_request": total_cost / max(total_requests, 1),
                    "avg_tokens_per_request": total_tokens / max(total_requests, 1)
                },
                "by_model": model_summary,
                "budgets": {
                    "daily": self.daily_budgets,
                    "monthly": self.monthly_budgets
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get cost summary: {e}")
            return {}

class CostOptimizer:
    """Optimize costs through intelligent model selection and caching."""
    
    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        return CostCalculator.estimate_cost(model, input_tokens, output_tokens)
    
    def suggest_model(self, query: str, complexity_threshold: float = 0.7) -> str:
        """Suggest optimal model based on query complexity and cost."""
        return CostCalculator.suggest_model(query, complexity_threshold)
    
    async def optimize_session_costs(self, session_id: str) -> Dict[str, Any]:
        """Optimize costs for a specific session."""
        try:
            # Get session costs
            query = """
            SELECT c.model, SUM(c.cost_usd) as total_cost, COUNT(1) as request_count
            FROM c
            WHERE c.session_id = @session_id
            GROUP BY c.model
            """
            
            parameters = [{"name": "@session_id", "value": session_id}]
            
            results = await cosmos_client.query_items("llm_analytics", query, parameters)
            
            session_costs = {}
            total_session_cost = 0
            
            for result in results:
                session_costs[result["model"]] = {
                    "cost": result["total_cost"],
                    "requests": result["request_count"]
                }
                total_session_cost += result["total_cost"]
            
            # Generate optimization suggestions
            suggestions = []
            
            if "gpt-4o" in session_costs and "gpt-4o-mini" not in session_costs:
                suggestions.append({
                    "type": "model_optimization",
                    "message": "Consider using gpt-4o-mini for simpler queries to reduce costs",
                    "potential_savings": session_costs["gpt-4o"]["cost"] * 0.3
                })
            
            if total_session_cost > 5.0:  # High-cost session
                suggestions.append({
                    "type": "session_cost_warning",
                    "message": f"Session cost is high: ${total_session_cost:.2f}",
                    "recommendation": "Consider breaking complex queries into smaller parts"
                })
            
            return {
                "session_id": session_id,
                "total_cost": total_session_cost,
                "model_usage": session_costs,
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize session costs: {e}")
            return {}

    def get_current_costs(self) -> Dict[str, Any]:
        """Get current cost tracking statistics."""
        return {
            "daily_budgets": self.daily_budgets,
            "monthly_budgets": self.monthly_budgets,
            "tracking_enabled": True
        }

# Global instances
cost_tracker = CostTracker()
cost_optimizer = CostOptimizer(cost_tracker)

