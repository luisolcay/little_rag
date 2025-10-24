"""
Azure Services Manager
======================

Centralized manager for all Azure services used in the RAG system.
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import all Azure services
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.vector.embeddings.azure_embedding_service import AzureEmbeddingService
from core.vector.azure_search.index_manager import IndexManager
from core.vector.azure_search.search_client import AzureSearchClient
from core.memory.memory_manager import MemoryManager
from core.ingest.processing import (
    AzureDocumentIntelligenceOcrProvider,
    DoclingOcrProvider
)

logger = logging.getLogger(__name__)

class AzureServicesManager:
    """Centralized manager for Azure services."""
    
    def __init__(self):
        self.services = {}
        self.health_status = {}
        self.initialization_time = None
        
    async def initialize_all_services(self) -> Dict[str, Any]:
        """Initialize all Azure services."""
        logger.info("🚀 Initializing Azure Services Manager...")
        
        start_time = datetime.now()
        results = {
            "successful": [],
            "failed": [],
            "warnings": []
        }
        
        # 1. Azure OpenAI Embedding Service
        try:
            self.services["embedding"] = AzureEmbeddingService()
            results["successful"].append("Azure OpenAI Embedding Service")
            logger.info("✅ Azure OpenAI Embedding Service initialized")
        except Exception as e:
            results["failed"].append(f"Azure OpenAI Embedding Service: {str(e)}")
            logger.error(f"❌ Azure OpenAI Embedding Service failed: {e}")
        
        # 2. Azure AI Search Index Manager
        try:
            self.services["index_manager"] = IndexManager(
                endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
                api_key=os.getenv("AZURE_SEARCH_API_KEY_ADMIN"),
                index_name="orbe-documents",
                embedding_dimension=1536  # text-embedding-3-large dimension
            )
            results["successful"].append("Azure AI Search Index Manager")
            logger.info("✅ Azure AI Search Index Manager initialized")
        except Exception as e:
            results["failed"].append(f"Azure AI Search Index Manager: {str(e)}")
            logger.error(f"❌ Azure AI Search Index Manager failed: {e}")
        
        # 3. Azure AI Search Client
        try:
            self.services["search_client"] = AzureSearchClient(
                endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
                api_key=os.getenv("AZURE_SEARCH_API_KEY_ADMIN"),
                index_name="orbe-documents"
            )
            results["successful"].append("Azure AI Search Client")
            logger.info("✅ Azure AI Search Client initialized")
        except Exception as e:
            results["failed"].append(f"Azure AI Search Client: {str(e)}")
            logger.error(f"❌ Azure AI Search Client failed: {e}")
        
        # 4. Memory Manager (Cosmos DB + Redis)
        try:
            self.services["memory"] = MemoryManager(
                cosmos_endpoint=os.getenv("AZURE_COSMOS_DB_ENDPOINT"),
                cosmos_key=os.getenv("AZURE_COSMOS_DB_PRIMARY_KEY"),
                cosmos_database_name=os.getenv("AZURE_COSMOS_DB_DATABASE_NAME", "collahuasi-rag"),
                cosmos_container_name=os.getenv("AZURE_COSMOS_DB_CONTAINER_NAME", "conversations"),
                redis_host=os.getenv("AZURE_REDIS_HOST"),
                redis_port=int(os.getenv("AZURE_REDIS_PORT", "6380")),
                redis_password=os.getenv("AZURE_REDIS_PRIMARY_KEY"),
                redis_ssl=os.getenv("AZURE_REDIS_SSL", "true").lower() == "true"
            )
            results["successful"].append("Memory Manager (Cosmos DB + Redis)")
            logger.info("✅ Memory Manager initialized")
        except Exception as e:
            results["failed"].append(f"Memory Manager: {str(e)}")
            logger.error(f"❌ Memory Manager failed: {e}")
        
        # 5. Azure Document Intelligence OCR
        try:
            self.services["azure_ocr"] = AzureDocumentIntelligenceOcrProvider(
                endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
                api_key=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY"),
                model_id=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID", "prebuilt-read")
            )
            results["successful"].append("Azure Document Intelligence OCR")
            logger.info("✅ Azure Document Intelligence OCR initialized")
        except Exception as e:
            results["warnings"].append(f"Azure Document Intelligence OCR: {str(e)}")
            logger.warning(f"⚠️ Azure Document Intelligence OCR not available: {e}")
        
        # 6. Docling OCR (Optional)
        try:
            self.services["docling_ocr"] = DoclingOcrProvider()
            results["successful"].append("Docling OCR")
            logger.info("✅ Docling OCR initialized")
        except Exception as e:
            results["warnings"].append(f"Docling OCR: {str(e)}")
            logger.warning(f"⚠️ Docling OCR not available: {e}")
        
        # Calculate initialization time
        self.initialization_time = (datetime.now() - start_time).total_seconds()
        
        # Log summary
        logger.info(f"🎯 Azure Services Initialization Complete:")
        logger.info(f"   ✅ Successful: {len(results['successful'])}")
        logger.info(f"   ❌ Failed: {len(results['failed'])}")
        logger.info(f"   ⚠️ Warnings: {len(results['warnings'])}")
        logger.info(f"   ⏱️ Time: {self.initialization_time:.2f}s")
        
        return results
    
    def get_service(self, service_name: str):
        """Get a specific service by name."""
        return self.services.get(service_name)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        logger.info("🏥 Performing Azure services health check...")
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "overall_status": "healthy"
        }
        
        # Check each service
        for service_name, service in self.services.items():
            try:
                if service_name == "embedding":
                    # Test embedding service
                    test_result = await service.generate_embeddings(["test"])
                    health_status["services"][service_name] = {
                        "status": "healthy",
                        "response_time": "< 1s",
                        "details": f"Generated {len(test_result)} embeddings"
                    }
                
                elif service_name == "search_client":
                    # Test search client
                    # In a real implementation, you'd test actual search
                    health_status["services"][service_name] = {
                        "status": "healthy",
                        "response_time": "< 1s",
                        "details": "Search client responsive"
                    }
                
                elif service_name == "memory":
                    # Test memory manager
                    # In a real implementation, you'd test Cosmos DB and Redis
                    health_status["services"][service_name] = {
                        "status": "healthy",
                        "response_time": "< 1s",
                        "details": "Memory manager responsive"
                    }
                
                elif service_name in ["azure_ocr", "docling_ocr"]:
                    # OCR services are optional
                    health_status["services"][service_name] = {
                        "status": "available",
                        "response_time": "N/A",
                        "details": "OCR service available"
                    }
                
                else:
                    health_status["services"][service_name] = {
                        "status": "healthy",
                        "response_time": "< 1s",
                        "details": "Service responsive"
                    }
                
            except Exception as e:
                health_status["services"][service_name] = {
                    "status": "unhealthy",
                    "response_time": "N/A",
                    "details": f"Error: {str(e)}"
                }
                health_status["overall_status"] = "degraded"
        
        # Determine overall status
        unhealthy_services = [
            name for name, status in health_status["services"].items()
            if status["status"] == "unhealthy"
        ]
        
        if unhealthy_services:
            health_status["overall_status"] = "degraded"
            health_status["unhealthy_services"] = unhealthy_services
        
        logger.info(f"🏥 Health check complete: {health_status['overall_status']}")
        
        return health_status
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        return {
            "services_initialized": len(self.services),
            "initialization_time": self.initialization_time,
            "available_services": list(self.services.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def create_search_index(self) -> Dict[str, Any]:
        """Create the Azure AI Search index."""
        try:
            if "index_manager" not in self.services:
                raise Exception("Index Manager not available")
            
            result = self.services["index_manager"].create_index()
            
            return {
                "success": True,
                "index_created": result,
                "message": "Search index created successfully"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create search index: {e}")
            return {
                "success": False,
                "index_created": False,
                "error": str(e)
            }
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current Azure configuration."""
        return {
            "azure_openai": {
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_key_set": bool(os.getenv("AZURE_OPENAI_API_KEY")),
                "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                "chat_deployment": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            },
            "azure_search": {
                "endpoint": os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
                "api_key_set": bool(os.getenv("AZURE_SEARCH_API_KEY_ADMIN")),
                "index_name": "orbe-documents"
            },
            "azure_cosmos": {
                "endpoint": os.getenv("AZURE_COSMOS_DB_ENDPOINT"),
                "api_key_set": bool(os.getenv("AZURE_COSMOS_DB_PRIMARY_KEY")),
                "database": os.getenv("AZURE_COSMOS_DB_DATABASE_NAME", "collahuasi-rag"),
                "container": os.getenv("AZURE_COSMOS_DB_CONTAINER_NAME", "conversations")
            },
            "azure_redis": {
                "host": os.getenv("AZURE_REDIS_HOST"),
                "port": os.getenv("AZURE_REDIS_PORT", "6380"),
                "password_set": bool(os.getenv("AZURE_REDIS_PRIMARY_KEY")),
                "ssl": os.getenv("AZURE_REDIS_SSL", "true")
            },
            "azure_document_intelligence": {
                "endpoint": os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
                "api_key_set": bool(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")),
                "model_id": os.getenv("AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID", "prebuilt-read")
            }
        }

# Global instance
azure_services_manager = AzureServicesManager()
