"""
Complete Azure RAG Flow Test with All Services
==============================================

This script tests the complete RAG pipeline using all Azure services:
- Azure Document Intelligence (OCR)
- Azure OpenAI (Embeddings + Chat)
- Azure AI Search (Vector Search)
- Azure Cosmos DB (Memory)
- Azure Redis Cache (Session Management)
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)  # Force reload

from core.ingest.processing import (
    DocumentFile, 
    EnhancedHybridChunker,
    AzureDocumentIntelligenceOcrProvider,
    DoclingOcrProvider
)

class CompleteAzureRAGTest:
    """Complete Azure RAG flow test with all services."""
    
    def __init__(self):
        self.document_path = "TR_GIS_Corporativo.pdf"
        self.chunks = []
        self.embeddings = []
        self.search_results = []
        
        print("🚀 COMPLETE AZURE RAG FLOW TEST")
        print("=" * 60)
        print("Testing all Azure services:")
        print("✅ Azure Document Intelligence (OCR)")
        print("✅ Azure OpenAI (Embeddings + Chat)")
        print("✅ Azure AI Search (Vector Search)")
        print("✅ Azure Cosmos DB (Memory)")
        print("✅ Azure Redis Cache (Session)")
        print("=" * 60)
        
        # Check environment variables
        self._check_environment()
    
    def _check_environment(self):
        """Check if all required environment variables are set."""
        required_vars = [
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_API_KEY',
            'AZURE_AI_SEARCH_ENDPOINT',
            'AZURE_SEARCH_API_KEY_ADMIN',
            'AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT',
            'AZURE_DOCUMENT_INTELLIGENCE_API_KEY',
            'AZURE_COSMOS_DB_ENDPOINT',
            'AZURE_COSMOS_DB_PRIMARY_KEY',
            'AZURE_REDIS_HOST',
            'AZURE_REDIS_PRIMARY_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            value = os.getenv(var)
            if not value or value.strip() == '':
                missing_vars.append(var)
        
        if missing_vars:
            print(f"❌ Missing or empty environment variables: {', '.join(missing_vars)}")
            print("Please check your .env file")
            sys.exit(1)
        
        print("✅ All required environment variables are set")
        
        # Show configuration summary
        print("\n🔧 Azure Services Configuration:")
        print(f"  OpenAI Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
        print(f"  AI Search Endpoint: {os.getenv('AZURE_AI_SEARCH_ENDPOINT')}")
        print(f"  Document Intelligence: {os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT')}")
        print(f"  Cosmos DB: {os.getenv('AZURE_COSMOS_DB_ENDPOINT')}")
        print(f"  Redis Host: {os.getenv('AZURE_REDIS_HOST')}")
    
    async def run_complete_test(self):
        """Run the complete Azure RAG flow test."""
        
        try:
            # Step 1: Document Processing with Azure OCR
            print("\n📄 STEP 1: DOCUMENT PROCESSING WITH AZURE OCR")
            print("-" * 50)
            
            processing_result = await self._test_document_processing()
            if not processing_result['success']:
                print(f"❌ Document processing failed: {processing_result['error']}")
                return
            
            # Step 2: Generate Embeddings with Azure OpenAI
            print("\n🧠 STEP 2: GENERATING EMBEDDINGS WITH AZURE OPENAI")
            print("-" * 50)
            
            embeddings_result = await self._test_embedding_generation()
            if not embeddings_result['success']:
                print(f"❌ Embedding generation failed: {embeddings_result['error']}")
                return
            
            # Step 3: Index Documents in Azure AI Search
            print("\n🔍 STEP 3: INDEXING IN AZURE AI SEARCH")
            print("-" * 50)
            
            indexing_result = await self._test_document_indexing()
            if not indexing_result['success']:
                print(f"❌ Document indexing failed: {indexing_result['error']}")
                return
            
            # Step 4: Test Search Queries
            print("\n🔎 STEP 4: TESTING SEARCH QUERIES")
            print("-" * 50)
            
            search_result = await self._test_search_queries()
            if not search_result['success']:
                print(f"❌ Search testing failed: {search_result['error']}")
                return
            
            # Step 5: Test LLM Response Generation
            print("\n💬 STEP 5: TESTING LLM RESPONSE GENERATION")
            print("-" * 50)
            
            llm_result = await self._test_llm_responses()
            if not llm_result['success']:
                print(f"❌ LLM testing failed: {llm_result['error']}")
                return
            
            # Step 6: Test Memory Management
            print("\n🧠 STEP 6: TESTING MEMORY MANAGEMENT")
            print("-" * 50)
            
            memory_result = await self._test_memory_management()
            if not memory_result['success']:
                print(f"❌ Memory testing failed: {memory_result['error']}")
                return
            
            # Step 7: Generate Final Report
            print("\n📊 STEP 7: GENERATING FINAL REPORT")
            print("-" * 50)
            
            self._generate_final_report({
                'processing': processing_result,
                'embeddings': embeddings_result,
                'indexing': indexing_result,
                'search': search_result,
                'llm': llm_result,
                'memory': memory_result
            })
            
            print("\n🎉 COMPLETE AZURE RAG FLOW TEST SUCCESSFUL!")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    async def _test_document_processing(self) -> Dict[str, Any]:
        """Test document processing with Azure OCR."""
        try:
            # Initialize Azure Document Intelligence OCR
            azure_ocr = AzureDocumentIntelligenceOcrProvider()
            print("✅ Azure Document Intelligence OCR initialized")
            
            # Initialize Docling as fallback
            try:
                docling_ocr = DoclingOcrProvider()
                print("✅ Docling OCR initialized as fallback")
            except Exception as e:
                print(f"⚠️ Docling OCR not available: {e}")
                docling_ocr = None
            
            # Initialize enhanced chunker
            chunker = EnhancedHybridChunker(
                docling_provider=docling_ocr,
                azure_provider=azure_ocr,
                max_tokens=900,
                overlap_tokens=120,
                ocr_threshold=0.1,
                min_text_length=50,
                
                # Quality validation
                min_chunk_length=50,
                max_chunk_length=2000,
                min_sentence_count=1,
                max_repetition_ratio=0.3,
                max_special_char_ratio=0.3,
                min_quality_threshold=0.5,
                
                # Semantic overlap
                overlap_sentences=2,
                preserve_paragraphs=True,
                
                # Reference preservation
                enable_reference_preservation=True,
                
                # Pattern detection
                enable_pattern_detection=True,
                pattern_similarity_threshold=0.8,
                auto_clean_headers=True,
                noise_threshold=10.0,
                
                verbose=True
            )
            
            # Create DocumentFile
            doc_file = DocumentFile(
                local_path=self.document_path,
                blob_name=self.document_path,
                metadata={
                    "blob_name": self.document_path,
                    "original_ext": ".pdf",
                    "size_bytes": os.path.getsize(self.document_path),
                    "needs_ocr": False
                },
                needs_ocr=False
            )
            
            print(f"📄 Processing document: {doc_file.blob_name}")
            print(f"📊 Document size: {doc_file.metadata['size_bytes']:,} bytes")
            
            # Process document
            self.chunks = chunker.chunk_document(doc_file)
            
            print(f"✅ Document processed successfully!")
            print(f"📊 Generated {len(self.chunks)} chunks")
            
            # Show chunk statistics
            total_chars = sum(len(chunk.content) for chunk in self.chunks)
            avg_chars = total_chars / len(self.chunks)
            chunks_with_cleaning = sum(1 for chunk in self.chunks if chunk.metadata.get('cleaning_applied', False))
            
            print(f"📊 Total characters: {total_chars:,}")
            print(f"📊 Average chunk length: {avg_chars:.0f} characters")
            print(f"📊 Chunks with cleaning: {chunks_with_cleaning}")
            
            return {
                "success": True,
                "chunks": self.chunks,
                "processing_stats": chunker.get_processing_statistics()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_embedding_generation(self) -> Dict[str, Any]:
        """Test embedding generation with Azure OpenAI."""
        try:
            from core.vector.embeddings.azure_embedding_service import AzureEmbeddingService
            
            # Initialize Azure embedding service
            embedding_service = AzureEmbeddingService()
            print("✅ Azure OpenAI Embedding Service initialized")
            
            # Generate embeddings for all chunks
            print(f"🧠 Generating embeddings for {len(self.chunks)} chunks...")
            
            embedding_results = await embedding_service.generate_embeddings(
                [chunk.content for chunk in self.chunks]
            )
            
            self.embeddings = [result.embedding for result in embedding_results]
            
            print(f"✅ Generated {len(self.embeddings)} embeddings")
            print(f"📊 Embedding dimension: {len(self.embeddings[0]) if self.embeddings else 0}")
            
            # Add embeddings to chunks
            for i, chunk in enumerate(self.chunks):
                chunk.embedding = self.embeddings[i]
            
            return {
                "success": True,
                "embeddings": self.embeddings,
                "embedding_count": len(self.embeddings)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_document_indexing(self) -> Dict[str, Any]:
        """Test document indexing in Azure AI Search."""
        try:
            from core.vector.azure_search.index_manager import IndexManager
            from core.vector.azure_search.search_client import AzureSearchClient
            
            # Initialize index manager
            index_manager = IndexManager()
            print("✅ Azure AI Search Index Manager initialized")
            
            # Create index
            print("🔍 Creating search index...")
            index_created = index_manager.create_index()
            
            if index_created:
                print("✅ Search index created successfully")
            else:
                print("⚠️ Search index may already exist")
            
            # Initialize search client
            search_client = AzureSearchClient()
            print("✅ Azure AI Search Client initialized")
            
            # Index documents
            print(f"📝 Indexing {len(self.chunks)} documents...")
            
            documents = []
            for chunk in self.chunks:
                doc = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "embedding": chunk.embedding,
                    **chunk.metadata
                }
                documents.append(doc)
            
            # Index documents
            indexing_result = search_client.upload_documents(documents)
            
            print(f"✅ Indexed {len(documents)} documents")
            
            return {
                "success": True,
                "indexed_documents": len(documents),
                "indexing_result": indexing_result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_search_queries(self) -> Dict[str, Any]:
        """Test search queries with Azure AI Search."""
        try:
            from core.vector.azure_search.search_client import AzureSearchClient, HybridSearchRequest
            from core.vector.embeddings.azure_embedding_service import AzureEmbeddingService
            
            # Initialize services
            search_client = AzureSearchClient()
            embedding_service = AzureEmbeddingService()
            
            print("✅ Search and embedding services initialized")
            
            # Test queries
            test_queries = [
                "¿Qué es el GIS Corporativo de CODELCO?",
                "¿Cuáles son las necesidades identificadas para la plataforma GIS?",
                "¿Qué información geoespacial maneja CODELCO?",
                "¿Cuáles son los objetivos del sistema de información geográfica?",
                "¿Qué consultoría se requiere para el GIS?"
            ]
            
            search_results = {}
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n🔍 Query {i}: \"{query}\"")
                
                # Generate query embedding
                query_embedding_results = await embedding_service.generate_embeddings([query])
                query_vector = query_embedding_results[0].embedding
                
                # Create hybrid search request
                hybrid_request = HybridSearchRequest(
                    query_text=query,
                    query_vector=query_vector,
                    top=5
                )
                
                # Perform hybrid search
                hybrid_results = search_client.hybrid_search(hybrid_request)
                
                print(f"  📊 Found {len(hybrid_results)} results")
                
                if hybrid_results:
                    top_result = hybrid_results[0]
                    print(f"  🎯 Top result score: {top_result.score:.3f}")
                    print(f"  📄 Top result preview: {top_result.content[:100]}...")
                
                search_results[query] = {
                    "results_count": len(hybrid_results),
                    "top_score": hybrid_results[0].score if hybrid_results else 0,
                    "top_result_preview": hybrid_results[0].content[:100] if hybrid_results else ""
                }
            
            self.search_results = search_results
            
            return {
                "success": True,
                "search_results": search_results,
                "queries_tested": len(test_queries)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_llm_responses(self) -> Dict[str, Any]:
        """Test LLM response generation with Azure OpenAI."""
        try:
            from core.llm.orchestrator import LLMOrchestrator
            from core.vector.retrieval_service import AdvancedRetrievalService
            
            # Initialize services
            retrieval_service = AdvancedRetrievalService()
            llm_orchestrator = LLMOrchestrator()
            
            print("✅ LLM Orchestrator and Retrieval Service initialized")
            
            # Test queries with LLM responses
            test_queries = [
                "¿Qué es el GIS Corporativo de CODELCO y cuál es su propósito?",
                "¿Cuáles son las principales necesidades identificadas para la plataforma GIS?",
                "Resume los objetivos principales del sistema de información geográfica."
            ]
            
            llm_responses = {}
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n💬 LLM Query {i}: \"{query}\"")
                
                try:
                    # Retrieve relevant chunks
                    retrieval_result = await retrieval_service.retrieve_relevant_chunks(
                        query=query,
                        top_k=3
                    )
                    
                    print(f"  📊 Retrieved {len(retrieval_result.chunks)} relevant chunks")
                    
                    # Generate LLM response
                    llm_response = await llm_orchestrator.generate_response(
                        query=query,
                        retrieved_chunks=retrieval_result.chunks,
                        conversation_history=[]
                    )
                    
                    print(f"  ✅ LLM response generated")
                    print(f"  📄 Response preview: {llm_response.response[:150]}...")
                    
                    llm_responses[query] = {
                        "response": llm_response.response,
                        "retrieved_chunks": len(retrieval_result.chunks),
                        "citations": llm_response.citations,
                        "cost": llm_response.cost_info
                    }
                    
                except Exception as e:
                    print(f"  ❌ LLM query failed: {e}")
                    llm_responses[query] = {
                        "error": str(e)
                    }
            
            return {
                "success": True,
                "llm_responses": llm_responses,
                "queries_tested": len(test_queries)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_memory_management(self) -> Dict[str, Any]:
        """Test memory management with Azure Cosmos DB and Redis."""
        try:
            from core.memory.memory_manager import MemoryManager
            
            # Initialize memory manager
            memory_manager = MemoryManager()
            print("✅ Memory Manager initialized")
            
            # Test conversation management
            print("🧠 Testing conversation management...")
            
            # Create a test conversation
            conversation_id = "test_conversation_azure_rag"
            
            # Add messages to conversation
            test_messages = [
                {"role": "user", "content": "¿Qué es el GIS Corporativo de CODELCO?"},
                {"role": "assistant", "content": "El GIS Corporativo de CODELCO es un sistema de información geográfica..."},
                {"role": "user", "content": "¿Cuáles son sus principales objetivos?"},
                {"role": "assistant", "content": "Los principales objetivos incluyen la gestión integrada del recurso hídrico..."}
            ]
            
            for message in test_messages:
                await memory_manager.add_message(conversation_id, message)
            
            print(f"✅ Added {len(test_messages)} messages to conversation")
            
            # Retrieve conversation history
            history = await memory_manager.get_conversation_history(conversation_id)
            print(f"📊 Retrieved {len(history)} messages from history")
            
            # Test conversation summarization
            if len(history) >= 4:
                summary = await memory_manager.summarize_conversation(conversation_id)
                print(f"📝 Generated conversation summary: {summary[:100]}...")
            
            return {
                "success": True,
                "conversation_id": conversation_id,
                "messages_added": len(test_messages),
                "history_retrieved": len(history),
                "summary_generated": len(history) >= 4
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_final_report(self, results: Dict[str, Any]):
        """Generate final test report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create comprehensive report
        report = {
            "test_timestamp": timestamp,
            "document": self.document_path,
            "test_results": results,
            "summary": {
                "document_processing": results['processing']['success'],
                "embedding_generation": results['embeddings']['success'],
                "document_indexing": results['indexing']['success'],
                "search_queries": results['search']['success'],
                "llm_responses": results['llm']['success'],
                "memory_management": results['memory']['success']
            }
        }
        
        # Save report
        report_file = f"azure_rag_complete_test_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Complete test report saved to: {report_file}")
        
        # Print summary
        print("\n📊 TEST SUMMARY")
        print("=" * 30)
        
        all_success = all(result['success'] for result in results.values())
        
        if all_success:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Document Processing: SUCCESS")
            print("✅ Embedding Generation: SUCCESS")
            print("✅ Document Indexing: SUCCESS")
            print("✅ Search Queries: SUCCESS")
            print("✅ LLM Responses: SUCCESS")
            print("✅ Memory Management: SUCCESS")
        else:
            print("⚠️ SOME TESTS FAILED:")
            for test_name, result in results.items():
                status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
                print(f"  {test_name}: {status}")
        
        print(f"\n📄 Document: {self.document_path}")
        print(f"📊 Chunks generated: {len(self.chunks)}")
        print(f"🧠 Embeddings created: {len(self.embeddings)}")
        print(f"🔍 Search queries tested: {results['search'].get('queries_tested', 0)}")
        print(f"💬 LLM queries tested: {results['llm'].get('queries_tested', 0)}")

async def main():
    """Main function."""
    test = CompleteAzureRAGTest()
    await test.run_complete_test()

if __name__ == "__main__":
    asyncio.run(main())
