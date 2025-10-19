"""
Azure OpenAI Embedding Service for generating vector embeddings.
"""

import os
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import json
import time
from datetime import datetime

load_dotenv()

@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]
    token_count: int
    model: str
    timestamp: datetime

@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding generation."""
    embeddings: List[EmbeddingResult]
    total_tokens: int
    processing_time: float
    success_count: int
    error_count: int
    errors: List[str]

class AzureEmbeddingService:
    """
    Service for generating embeddings using Azure OpenAI.
    
    Features:
    - Batch processing for efficiency
    - Retry logic with exponential backoff
    - Token counting and cost estimation
    - Progress tracking
    - Error handling and recovery
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: str = "2024-10-21",
        model: str = "text-embedding-3-large",
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Azure Embedding Service.
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            deployment: Deployment name for embeddings
            api_version: API version
            model: Model name (for token counting)
            batch_size: Number of texts per batch
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries
        """
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.api_version = api_version
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not all([self.endpoint, self.api_key, self.deployment]):
            raise ValueError(
                "Missing required Azure OpenAI configuration. "
                "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
            )
        
        # Clean endpoint URL
        self.endpoint = self.endpoint.rstrip('/')
        
        # Initialize tokenizer for counting
        self._tokenizer = None
        self._init_tokenizer()
        
        # Statistics
        self.total_tokens_used = 0
        self.total_requests = 0
        self.total_errors = 0
    
    def _init_tokenizer(self):
        """Initialize tokenizer for token counting."""
        try:
            import tiktoken
            self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        except ImportError:
            print("[WARN] tiktoken not available, using character-based token estimation")
            self._tokenizer = None
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        else:
            # Rough estimation: ~4 characters per token
            return len(text) // 4
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost for token usage."""
        # text-embedding-3-large: $0.13 per 1M tokens
        return (tokens / 1_000_000) * 0.13
    
    async def _make_request(self, texts: List[str]) -> Dict[str, Any]:
        """Make API request to Azure OpenAI."""
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/embeddings"
        params = {"api-version": self.api_version}
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        payload = {
            "input": texts,
            "model": self.model
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, params=params, headers=headers, json=payload)
            
            if response.status_code != 200:
                error_msg = f"API request failed: {response.status_code} - {response.text}"
                raise Exception(error_msg)
            
            return response.json()
    
    async def _generate_batch_embeddings(
        self, 
        texts: List[str], 
        batch_id: int = 0
    ) -> BatchEmbeddingResult:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_id: Batch identifier for logging
            
        Returns:
            BatchEmbeddingResult with embeddings and statistics
        """
        start_time = time.time()
        embeddings = []
        errors = []
        total_tokens = 0
        
        print(f"[EMBEDDING] Processing batch {batch_id}: {len(texts)} texts")
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._make_request(texts)
                
                if "data" not in response:
                    raise Exception(f"Invalid response format: {response}")
                
                # Process successful response
                for i, item in enumerate(response["data"]):
                    if "embedding" not in item:
                        errors.append(f"Missing embedding for text {i}")
                        continue
                    
                    text = texts[i]
                    embedding = item["embedding"]
                    token_count = self._count_tokens(text)
                    
                    result = EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        token_count=token_count,
                        model=self.model,
                        timestamp=datetime.now()
                    )
                    
                    embeddings.append(result)
                    total_tokens += token_count
                
                # Update statistics
                self.total_tokens_used += total_tokens
                self.total_requests += 1
                
                processing_time = time.time() - start_time
                cost = self._estimate_cost(total_tokens)
                
                print(f"[EMBEDDING] Batch {batch_id} completed: {len(embeddings)} embeddings, "
                      f"{total_tokens:,} tokens, ${cost:.4f}, {processing_time:.2f}s")
                
                return BatchEmbeddingResult(
                    embeddings=embeddings,
                    total_tokens=total_tokens,
                    processing_time=processing_time,
                    success_count=len(embeddings),
                    error_count=len(errors),
                    errors=errors
                )
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {str(e)}"
                print(f"[EMBEDDING] {error_msg}")
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"[EMBEDDING] Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    self.total_errors += 1
                    errors.append(error_msg)
                    return BatchEmbeddingResult(
                        embeddings=[],
                        total_tokens=0,
                        processing_time=time.time() - start_time,
                        success_count=0,
                        error_count=len(texts),
                        errors=[error_msg]
                    )
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for a list of texts with batch processing.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress information
            
        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []
        
        print(f"[EMBEDDING] Starting embedding generation for {len(texts)} texts")
        print(f"[EMBEDDING] Batch size: {self.batch_size}, Model: {self.model}")
        
        all_embeddings = []
        total_errors = 0
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_id = i // self.batch_size + 1
            
            batch_result = await self._generate_batch_embeddings(batch_texts, batch_id)
            
            all_embeddings.extend(batch_result.embeddings)
            total_errors += batch_result.error_count
            
            if show_progress:
                progress = min(100, ((i + len(batch_texts)) / len(texts)) * 100)
                print(f"[EMBEDDING] Progress: {progress:.1f}% ({i + len(batch_texts)}/{len(texts)})")
        
        # Final statistics
        total_cost = self._estimate_cost(self.total_tokens_used)
        success_rate = (len(all_embeddings) / len(texts)) * 100 if texts else 0
        
        print(f"[EMBEDDING] Generation completed:")
        print(f"[EMBEDDING] - Total embeddings: {len(all_embeddings)}")
        print(f"[EMBEDDING] - Success rate: {success_rate:.1f}%")
        print(f"[EMBEDDING] - Total tokens: {self.total_tokens_used:,}")
        print(f"[EMBEDDING] - Estimated cost: ${total_cost:.4f}")
        print(f"[EMBEDDING] - Total errors: {total_errors}")
        
        return all_embeddings
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "estimated_cost": self._estimate_cost(self.total_tokens_used),
            "model": self.model,
            "batch_size": self.batch_size
        }
    
    def reset_statistics(self):
        """Reset service statistics."""
        self.total_tokens_used = 0
        self.total_requests = 0
        self.total_errors = 0

# Convenience function for single text embedding
async def embed_text(text: str) -> List[float]:
    """Generate embedding for a single text."""
    service = AzureEmbeddingService()
    results = await service.generate_embeddings([text])
    return results[0].embedding if results else []

# Convenience function for batch embedding
async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts."""
    service = AzureEmbeddingService()
    results = await service.generate_embeddings(texts)
    return [result.embedding for result in results]
