"""
Indexing pipeline for processing chunks and uploading to Azure AI Search.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import uuid

from ..embeddings.azure_embedding_service import AzureEmbeddingService, EmbeddingResult
from ..azure_search.search_client import AzureSearchClient
from ..azure_search.index_manager import IndexManager

class IndexingPipeline:
    """
    Pipeline for indexing document chunks to Azure AI Search.
    
    Features:
    - Read chunks from JSONL files
    - Generate embeddings in batches
    - Transform data for Azure AI Search schema
    - Upload documents to search index
    - Progress tracking and error handling
    """
    
    def __init__(
        self,
        chunks_file: str = "artifacts/chunks.jsonl",
        index_name: str = "orbe-documents",
        batch_size: int = 100,
        show_progress: bool = True
    ):
        """
        Initialize the indexing pipeline.
        
        Args:
            chunks_file: Path to chunks JSONL file
            index_name: Name of the Azure AI Search index
            batch_size: Batch size for embedding generation
            show_progress: Whether to show progress information
        """
        self.chunks_file = Path(chunks_file)
        self.index_name = index_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        # Initialize services
        self.embedding_service = AzureEmbeddingService(batch_size=batch_size)
        self.search_client = AzureSearchClient(index_name=index_name)
        self.index_manager = IndexManager(index_name=index_name)
        
        # Statistics
        self.total_chunks = 0
        self.processed_chunks = 0
        self.failed_chunks = 0
        self.start_time = None
        self.end_time = None
    
    def load_chunks(self) -> List[Dict[str, Any]]:
        """
        Load chunks from JSONL file.
        
        Returns:
            List of chunk dictionaries
        """
        if not self.chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_file}")
        
        chunks = []
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line.strip())
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"[PIPELINE] Error parsing line {line_num}: {e}")
                    continue
        
        self.total_chunks = len(chunks)
        print(f"[PIPELINE] Loaded {self.total_chunks} chunks from {self.chunks_file}")
        
        return chunks
    
    def transform_chunk_for_index(self, chunk: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
        """
        Transform chunk data to Azure AI Search schema.
        
        Args:
            chunk: Original chunk data
            embedding: Generated embedding vector
            
        Returns:
            Transformed document for index
        """
        # Extract metadata
        metadata = chunk.get("metadata", {})
        
        # Create document for Azure AI Search
        document = {
            "chunk_id": chunk.get("chunk_id", str(uuid.uuid4())),
            "content": chunk.get("content", ""),
            "content_vector": embedding,
            "document_id": metadata.get("document_id", ""),
            "document_blob": metadata.get("document_blob", ""),
            "page_number": metadata.get("page_number", 0),
            "quality_score": metadata.get("quality_score", 0.0),
            "has_references": metadata.get("has_references", False),
            "metadata": {
                "chunk_index": metadata.get("chunk_index", 0),
                "needs_ocr": metadata.get("needs_ocr", False),
                "reference_count": metadata.get("reference_count", 0),
                "ingested_at": int(datetime.now().timestamp()),
                "processing_stats": json.dumps(metadata.get("processing_stats", {}))
            }
        }
        
        return document
    
    async def generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], EmbeddingResult]]:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of tuples (chunk, embedding_result)
        """
        # Extract texts for embedding
        texts = [chunk.get("content", "") for chunk in chunks]
        
        # Generate embeddings
        embedding_results = await self.embedding_service.generate_embeddings(
            texts, 
            show_progress=self.show_progress
        )
        
        # Pair chunks with their embeddings
        chunk_embedding_pairs = []
        for i, (chunk, embedding_result) in enumerate(zip(chunks, embedding_results)):
            chunk_embedding_pairs.append((chunk, embedding_result))
        
        return chunk_embedding_pairs
    
    def upload_documents_batch(self, documents: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Upload a batch of documents to the search index.
        
        Args:
            documents: List of documents to upload
            
        Returns:
            Tuple of (success_count, error_count)
        """
        success = self.search_client.upload_documents(documents)
        
        if success:
            return len(documents), 0
        else:
            return 0, len(documents)
    
    async def run_indexing(self) -> Dict[str, Any]:
        """
        Run the complete indexing pipeline.
        
        Returns:
            Dictionary with indexing results and statistics
        """
        self.start_time = datetime.now()
        print(f"[PIPELINE] Starting indexing pipeline at {self.start_time}")
        print(f"[PIPELINE] Index: {self.index_name}, Batch size: {self.batch_size}")
        
        try:
            # 1. Ensure index exists
            if not self.index_manager.index_exists():
                print("[PIPELINE] Creating search index...")
                if not self.index_manager.create_index():
                    raise Exception("Failed to create search index")
            else:
                print("[PIPELINE] Search index already exists")
            
            # 2. Load chunks
            chunks = self.load_chunks()
            if not chunks:
                raise Exception("No chunks found to index")
            
            # 3. Generate embeddings
            print("[PIPELINE] Generating embeddings...")
            chunk_embedding_pairs = await self.generate_embeddings_for_chunks(chunks)
            
            # 4. Transform and upload documents
            print("[PIPELINE] Uploading documents to search index...")
            
            all_documents = []
            for chunk, embedding_result in chunk_embedding_pairs:
                document = self.transform_chunk_for_index(chunk, embedding_result.embedding)
                all_documents.append(document)
            
            # Upload in batches
            total_uploaded = 0
            total_failed = 0
            
            for i in range(0, len(all_documents), self.batch_size):
                batch_documents = all_documents[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                
                print(f"[PIPELINE] Uploading batch {batch_num}: {len(batch_documents)} documents")
                
                success_count, error_count = self.upload_documents_batch(batch_documents)
                total_uploaded += success_count
                total_failed += error_count
                
                if self.show_progress:
                    progress = min(100, ((i + len(batch_documents)) / len(all_documents)) * 100)
                    print(f"[PIPELINE] Upload progress: {progress:.1f}% ({i + len(batch_documents)}/{len(all_documents)})")
            
            # 5. Final statistics
            self.end_time = datetime.now()
            processing_time = (self.end_time - self.start_time).total_seconds()
            
            # Get embedding statistics
            embedding_stats = self.embedding_service.get_statistics()
            
            # Get index statistics
            index_stats = self.search_client.get_document_count()
            
            results = {
                "success": True,
                "total_chunks": len(chunks),
                "total_uploaded": total_uploaded,
                "total_failed": total_failed,
                "processing_time_seconds": processing_time,
                "index_name": self.index_name,
                "index_document_count": index_stats,
                "embedding_stats": embedding_stats,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat()
            }
            
            print(f"[PIPELINE] Indexing completed successfully!")
            print(f"[PIPELINE] - Total chunks: {results['total_chunks']}")
            print(f"[PIPELINE] - Uploaded: {results['total_uploaded']}")
            print(f"[PIPELINE] - Failed: {results['total_failed']}")
            print(f"[PIPELINE] - Processing time: {processing_time:.2f}s")
            print(f"[PIPELINE] - Index documents: {results['index_document_count']}")
            print(f"[PIPELINE] - Embedding cost: ${embedding_stats['estimated_cost']:.4f}")
            
            return results
            
        except Exception as e:
            self.end_time = datetime.now()
            processing_time = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
            
            print(f"[PIPELINE] Indexing failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "total_chunks": self.total_chunks,
                "processed_chunks": self.processed_chunks,
                "failed_chunks": self.failed_chunks,
                "processing_time_seconds": processing_time,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "failed_chunks": self.failed_chunks,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "is_running": self.start_time is not None and self.end_time is None
        }

# Convenience function
async def index_chunks(
    chunks_file: str = "artifacts/chunks.jsonl",
    index_name: str = "orbe-documents",
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Convenience function to index chunks.
    
    Args:
        chunks_file: Path to chunks JSONL file
        index_name: Name of the search index
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with indexing results
    """
    pipeline = IndexingPipeline(
        chunks_file=chunks_file,
        index_name=index_name,
        batch_size=batch_size
    )
    
    return await pipeline.run_indexing()
