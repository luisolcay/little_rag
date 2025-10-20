"""
Versioned prompt management system with templating and metrics tracking.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import asyncio
import time
from functools import lru_cache

from jinja2 import Template, Environment, FileSystemLoader, TemplateSyntaxError
from ..memory.cosmos_store import cosmos_client

logger = logging.getLogger(__name__)

class PromptManager:
    """Versioned prompt management with templating and metrics."""
    
    def __init__(self):
        self.prompts_dir = Path("core/llm/prompts")
        self.active_versions_file = self.prompts_dir / "active_versions.json"
        self.jinja_env = Environment(loader=FileSystemLoader(str(self.prompts_dir)))
        
        # Enhanced cache with LRU and TTL
        self.prompt_cache = {}
        self.version_cache = {}
        self.template_cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.last_cache_cleanup = time.time()
        
        # Performance metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "template_validations": 0,
            "template_errors": 0,
            "total_requests": 0
        }
        
        # Initialize active versions
        self.active_versions = self._load_active_versions()
    
    def _load_active_versions(self) -> Dict[str, str]:
        """Load active prompt versions."""
        try:
            if self.active_versions_file.exists():
                with open(self.active_versions_file, 'r') as f:
                    return json.load(f)
            else:
                # Default versions
                default_versions = {
                    "environmental_consultant": "v1.0.0",
                    "contextualization": "v1.0.0"
                }
                self._save_active_versions(default_versions)
                return default_versions
        except Exception as e:
            logger.error(f"Failed to load active versions: {e}")
            return {}
    
    def _save_active_versions(self, versions: Dict[str, str]):
        """Save active prompt versions."""
        try:
            with open(self.active_versions_file, 'w') as f:
                json.dump(versions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save active versions: {e}")
    
    def _get_prompt_path(self, prompt_type: str, version: str) -> Path:
        """Get path to prompt template."""
        return self.prompts_dir / prompt_type / version
    
    def _load_prompt_metadata(self, prompt_type: str, version: str) -> Dict[str, Any]:
        """Load prompt metadata."""
        metadata_file = self._get_prompt_path(prompt_type, version) / "metadata.json"
        
        try:
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "version": version,
                    "prompt_type": prompt_type,
                    "author": "system",
                    "created_at": datetime.now().isoformat(),
                    "changelog": "Initial version"
                }
        except Exception as e:
            logger.error(f"Failed to load metadata for {prompt_type}/{version}: {e}")
            return {}
    
    def _load_prompt_template(self, prompt_type: str, version: str, template_name: str) -> Optional[str]:
        """Load prompt template content with validation."""
        template_file = self._get_prompt_path(prompt_type, version) / f"{template_name}.txt"
        
        try:
            if template_file.exists():
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Validate template syntax
                self._validate_template(content, template_name)
                return content
            else:
                logger.warning(f"Template not found: {template_file}")
                return None
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
            return None
    
    def _validate_template(self, template_content: str, template_name: str) -> bool:
        """Validate Jinja2 template syntax."""
        try:
            self.metrics["template_validations"] += 1
            Template(template_content)
            return True
        except TemplateSyntaxError as e:
            self.metrics["template_errors"] += 1
            logger.error(f"Template syntax error in {template_name}: {e}")
            raise
        except Exception as e:
            self.metrics["template_errors"] += 1
            logger.error(f"Template validation error in {template_name}: {e}")
            raise
    
    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        if current_time - self.last_cache_cleanup < 300:  # Clean every 5 minutes
            return
        
        expired_keys = []
        for key, (content, timestamp) in self.prompt_cache.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.prompt_cache[key]
        
        self.last_cache_cleanup = current_time
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_prompt(self, prompt_type: str, template_name: str, variables: Optional[Dict[str, Any]] = None, version: Optional[str] = None) -> Optional[str]:
        """Get prompt with templating and enhanced caching."""
        try:
            self.metrics["total_requests"] += 1
            
            # Cleanup cache periodically
            self._cleanup_cache()
            
            # Use active version if not specified
            if not version:
                version = self.active_versions.get(prompt_type)
                if not version:
                    logger.error(f"No active version for prompt type: {prompt_type}")
                    return None
            
            # Create cache key with variables hash for better cache utilization
            variables_hash = hash(str(sorted((variables or {}).items())))
            cache_key = f"{prompt_type}:{version}:{template_name}:{variables_hash}"
            
            # Check enhanced cache first
            if cache_key in self.prompt_cache:
                content, timestamp = self.prompt_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    self.metrics["cache_hits"] += 1
                    return content
                else:
                    # Remove expired entry
                    del self.prompt_cache[cache_key]
            
            self.metrics["cache_misses"] += 1
            
            # Load template content
            template_content = self._load_prompt_template(prompt_type, version, template_name)
            if not template_content:
                return None
            
            # Render template with variables
            template = Template(template_content)
            rendered = template.render(**(variables or {}))
            
            # Cache the rendered result
            self.prompt_cache[cache_key] = (rendered, time.time())
            
            return rendered
            
        except Exception as e:
            logger.error(f"Failed to get prompt {prompt_type}/{template_name}: {e}")
            return None
    
    def get_system_prompt(self, query_type: Optional[str] = None, variables: Optional[Dict[str, Any]] = None) -> str:
        """Get system prompt for environmental consultant."""
        return self.get_prompt(
            "environmental_consultant",
            "system_prompt",
            variables={
                "query_type": query_type,
                **(variables or {})
            }
        ) or self._get_default_system_prompt()
    
    def get_compliance_prompt(self, variables: Optional[Dict[str, Any]] = None) -> str:
        """Get compliance analysis prompt."""
        return self.get_prompt(
            "environmental_consultant",
            "compliance_analysis",
            variables
        ) or self._get_default_compliance_prompt()
    
    def get_risk_assessment_prompt(self, variables: Optional[Dict[str, Any]] = None) -> str:
        """Get risk assessment prompt."""
        return self.get_prompt(
            "environmental_consultant",
            "risk_assessment",
            variables
        ) or self._get_default_risk_assessment_prompt()
    
    def get_contextualization_prompt(self, variables: Optional[Dict[str, Any]] = None) -> str:
        """Get query contextualization prompt."""
        return self.get_prompt(
            "contextualization",
            "query_rewrite",
            variables
        ) or self._get_default_contextualization_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """Default system prompt."""
        return """You are an expert environmental consultant specializing in Chilean mining operations and environmental regulations. You provide accurate, evidence-based analysis with proper citations.

Key areas of expertise:
- Chilean environmental regulations (SEA, SMA, DGA)
- Mining environmental impact assessments
- Water resource management
- Air quality and emissions
- Waste management and disposal
- Biodiversity and ecosystem protection
- Community relations and social impact

Always provide:
1. Evidence-based analysis with citations
2. Clear, actionable recommendations
3. Risk assessments with mitigation strategies
4. Compliance requirements and deadlines
5. Relevant regulatory references

Be precise, professional, and focused on practical solutions for environmental challenges in mining operations."""
    
    def _get_default_compliance_prompt(self) -> str:
        """Default compliance analysis prompt."""
        return """Analyze the following query for regulatory compliance requirements:

Query: {{ query }}

Provide a structured compliance analysis including:
1. Applicable regulations and standards
2. Compliance status assessment
3. Key findings and gaps
4. Risk level and implications
5. Specific recommendations
6. Required actions and timelines

Focus on Chilean environmental regulations and mining-specific requirements."""
    
    def _get_default_risk_assessment_prompt(self) -> str:
        """Default risk assessment prompt."""
        return """Conduct a comprehensive risk assessment for the following environmental query:

Query: {{ query }}

Provide a structured risk assessment including:
1. Risk category and classification
2. Probability and impact assessment
3. Risk score calculation
4. Key risk factors identified
5. Mitigation strategies
6. Monitoring recommendations
7. Risk level (low/medium/high/critical)

Consider environmental, regulatory, operational, and reputational risks."""
    
    def _get_default_contextualization_prompt(self) -> str:
        """Default contextualization prompt."""
        return """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat History:
{% for message in chat_history %}
{{ message.role }}: {{ message.content }}
{% endfor %}

Latest Question: {{ input }}

Standalone Question:"""
    
    def list_versions(self, prompt_type: str) -> List[Dict[str, Any]]:
        """List all versions for a prompt type."""
        try:
            prompt_dir = self.prompts_dir / prompt_type
            if not prompt_dir.exists():
                return []
            
            versions = []
            for version_dir in prompt_dir.iterdir():
                if version_dir.is_dir():
                    metadata = self._load_prompt_metadata(prompt_type, version_dir.name)
                    metadata["is_active"] = self.active_versions.get(prompt_type) == version_dir.name
                    versions.append(metadata)
            
            return sorted(versions, key=lambda x: x["created_at"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list versions for {prompt_type}: {e}")
            return []
    
    def activate_version(self, prompt_type: str, version: str) -> bool:
        """Activate a specific prompt version."""
        try:
            # Verify version exists
            version_path = self._get_prompt_path(prompt_type, version)
            if not version_path.exists():
                logger.error(f"Version {version} not found for {prompt_type}")
                return False
            
            # Update active versions
            self.active_versions[prompt_type] = version
            self._save_active_versions(self.active_versions)
            
            # Clear cache for this prompt type
            self._clear_cache_for_type(prompt_type)
            
            logger.info(f"Activated version {version} for {prompt_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate version {version} for {prompt_type}: {e}")
            return False
    
    def _clear_cache_for_type(self, prompt_type: str):
        """Clear cache for a specific prompt type."""
        keys_to_remove = [key for key in self.prompt_cache.keys() if key.startswith(f"{prompt_type}:")]
        for key in keys_to_remove:
            del self.prompt_cache[key]
    
    async def track_prompt_metrics(self, prompt_type: str, version: str, metrics: Dict[str, Any]):
        """Track prompt performance metrics."""
        try:
            metric_data = {
                "id": f"metrics_{prompt_type}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "prompt_type": prompt_type,
                "prompt_version": version,
                "timestamp": datetime.now().isoformat(),
                **metrics
            }
            
            await cosmos_client.create_item("prompt_metrics", metric_data)
            
        except Exception as e:
            logger.error(f"Failed to track prompt metrics: {e}")
    
    def get_prompt_performance(self, prompt_type: str, version: str, days: int = 30) -> Dict[str, Any]:
        """Get prompt performance metrics."""
        try:
            # Calculate cache hit rate
            total_requests = self.metrics["total_requests"]
            cache_hit_rate = (self.metrics["cache_hits"] / max(total_requests, 1)) * 100
            
            # Get template validation stats
            validation_success_rate = 0
            if self.metrics["template_validations"] > 0:
                validation_success_rate = ((self.metrics["template_validations"] - self.metrics["template_errors"]) / 
                                        self.metrics["template_validations"]) * 100
            
            return {
                "prompt_type": prompt_type,
                "version": version,
                "period_days": days,
                "cache_performance": {
                    "hit_rate": cache_hit_rate,
                    "total_requests": total_requests,
                    "cache_hits": self.metrics["cache_hits"],
                    "cache_misses": self.metrics["cache_misses"]
                },
                "template_validation": {
                    "total_validations": self.metrics["template_validations"],
                    "errors": self.metrics["template_errors"],
                    "success_rate": validation_success_rate
                },
                "cache_size": len(self.prompt_cache),
                "last_cleanup": self.last_cache_cleanup
            }
            
        except Exception as e:
            logger.error(f"Failed to get prompt performance: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "metrics": self.metrics.copy(),
            "cache_size": len(self.prompt_cache),
            "cache_ttl": self.cache_ttl,
            "last_cleanup": self.last_cache_cleanup
        }

# Global instance
prompt_manager = PromptManager()

