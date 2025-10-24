"""
Script to process and index PDF documents to orbe-documents index
=================================================================

This script processes PDF documents and indexes them to Azure AI Search.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

from core.ingest.processing import (
    DocumentFile, 
    EnhancedHybridChunker,
    AzureDocumentIntelligenceOcrProvider,
    DoclingOcrProvider
)
from core.vector.embeddings.azure_embedding_service import AzureEmbeddingService
from core.vector.azure_search.search_client import AzureSearchClient
from core.vector.azure_search.index_manager import IndexManager

async def process_and_index_pdfs(pdf_files: List[str], index_name: str = "orbe-documents"):
    """
    Process PDF files and index them to Azure AI Search.
    
    Args:
        pdf_files: List of PDF file paths
        index_name: Name of the Azure AI Search index
    """
    print("🚀 Starting PDF processing and indexing")
    print("=" * 60)
    print(f"Index: {index_name}")
    print(f"Documents: {len(pdf_files)}")
    print("=" * 60)
    
    # Step 1: Initialize services
    print("\n📋 Step 1: Initializing services...")
    
    # Initialize OCR providers
    azure_ocr = None
    docling_ocr = None
    
    try:
        azure_ocr = AzureDocumentIntelligenceOcrProvider()
        print("✅ Azure Document Intelligence OCR initialized")
    except Exception as e:
        print(f"⚠️ Azure OCR not available: {e}")
    
    try:
        docling_ocr = DoclingOcrProvider()
        print("✅ Docling OCR initialized")
    except Exception as e:
        print(f"⚠️ Docling OCR not available: {e}")
    
    # Initialize chunker
    chunker = EnhancedHybridChunker(
        docling_provider=docling_ocr,
        azure_provider=azure_ocr,
        max_tokens=900,
        overlap_tokens=120,
        ocr_threshold=0.1,
        min_text_length=50,
        min_chunk_length=50,
        max_chunk_length=2000,
        min_sentence_count=1,
        max_repetition_ratio=0.3,
        max_special_char_ratio=0.3,
        min_quality_threshold=0.5,
        overlap_sentences=2,
        preserve_paragraphs=True,
        enable_reference_preservation=True,
        enable_pattern_detection=True,
        pattern_similarity_threshold=0.8,
        auto_clean_headers=True,
        noise_threshold=10.0,
        verbose=True
    )
    print("✅ Chunker initialized")
    
    # Initialize embedding service
    embedding_service = AzureEmbeddingService()
    print("✅ Embedding service initialized")
    
    # Initialize search client
    search_client = AzureSearchClient(index_name=index_name)
    print("✅ Search client initialized")
    
    # Step 2: Ensure index exists
    print("\n📋 Step 2: Checking index...")
    index_manager = IndexManager(index_name=index_name)
    
    if not index_manager.index_exists():
        print(f"Creating index '{index_name}'...")
        index_manager.create_index()
        print(f"✅ Index '{index_name}' created")
    else:
        print(f"✅ Index '{index_name}' already exists")
    
    # Step 3: Process each PDF
    print("\n📋 Step 3: Processing documents...")
    
    total_chunks = 0
    total_documents = 0
    
    for pdf_file in pdf_files:
        pdf_path = Path(pdf_file)
        
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_file}")
            continue
        
        print(f"\n📄 Processing: {pdf_path.name}")
        print("-" * 40)
        
        try:
            # Create DocumentFile
            doc_file = DocumentFile(
                local_path=str(pdf_path),
                blob_name=pdf_path.name,
                metadata={
                    "blob_name": pdf_path.name,
                    "original_ext": ".pdf",
                    "size_bytes": pdf_path.stat().st_size,
                    "needs_ocr": False,
                },
                needs_ocr=False
            )
            
            # Chunk document
            print("  🔄 Chunking document...")
            chunks = chunker.chunk_document(doc_file)
            print(f"  ✅ Generated {len(chunks)} chunks")
            
            # Generate embeddings
            print("  🔄 Generating embeddings...")
            chunk_contents = [chunk.content for chunk in chunks]
            embedding_results = await embedding_service.generate_embeddings(chunk_contents)
            
            # Prepare documents for indexing
            print("  🔄 Preparing documents for indexing...")
            documents_to_index = []
            
            for i, (chunk, embedding_result) in enumerate(zip(chunks, embedding_results)):
                doc = {
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "content_vector": embedding_result.embedding,
                    "document_id": pdf_path.stem,
                    "document_blob": pdf_path.name,
                    "page_number": chunk.metadata.get('page_number', 0),
                    "quality_score": chunk.metadata.get('quality_score', 0.8),
                    "has_references": chunk.metadata.get('has_references', False),
                    "metadata": {
                        "chunk_index": i,
                        "needs_ocr": False,
                        "reference_count": chunk.metadata.get('reference_count', 0),
                        "ingested_at": int(pdf_path.stat().st_mtime),
                        "processing_stats": str(chunk.metadata.get('processing_stats', {}))
                    }
                }
                documents_to_index.append(doc)
            
            # Upload to Azure Search
            print(f"  🔄 Uploading {len(documents_to_index)} documents to Azure Search...")
            success = search_client.upload_documents(documents_to_index)
            
            if success:
                print(f"  ✅ Successfully indexed {len(documents_to_index)} chunks")
                total_chunks += len(documents_to_index)
                total_documents += 1
            else:
                print(f"  ❌ Failed to index chunks")
            
        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Documents processed: {total_documents}/{len(pdf_files)}")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Index: {index_name}")
    print("=" * 60)
    
    # Get index statistics
    print("\n📊 Index Statistics:")
    stats = index_manager.get_index_statistics()
    if stats:
        print(f"  Total documents in index: {stats.get('document_count', 0)}")
        print(f"  Storage size: {stats.get('storage_size', 0)} bytes")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    # PDF files to process
    pdf_files = [
        "AMB-PR-02 Procedimiento de Control de Documentos.pdf",
        "Manual Ingreso Horas Hombre a Sistema Tracking.pdf",
        "OTE 8333 -ODS N° 4 CODELCO EMI_0.pdf"
    ]
    
    # Run async function
    asyncio.run(process_and_index_pdfs(pdf_files, index_name="orbe-documents"))

