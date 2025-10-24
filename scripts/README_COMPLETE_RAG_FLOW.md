# Complete RAG Flow Integration Test

This directory contains scripts to test the complete RAG (Retrieval-Augmented Generation) pipeline integrating all components:

- **Enhanced Document Processing** with pattern detection and header cleaning
- **Azure AI Search** with hybrid search capabilities  
- **LangChain Orchestrator** with structured outputs
- **Cosmos DB and Redis** for memory management
- **Cost tracking** and monitoring

## 🚀 Quick Start

### 1. Check Environment Configuration

```bash
python scripts/check_env_config.py
```

This will verify all required environment variables are set.

### 2. Run Complete Test

```bash
python scripts/run_complete_test.py
```

This will run the complete RAG flow test with proper environment setup.

## 📋 Required Environment Variables

### Azure OpenAI
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_CHAT_DEPLOYMENT_MINI`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`

### Azure AI Search
- `AZURE_AI_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY_ADMIN`
- `AZURE_SEARCH_API_KEY_QUERY`

### Azure Cosmos DB
- `AZURE_COSMOS_DB_ENDPOINT`
- `AZURE_COSMOS_DB_PRIMARY_KEY`
- `AZURE_COSMOS_DB_SECONDARY_KEY`

### Azure Redis
- `AZURE_REDIS_HOST`
- `AZURE_REDIS_PORT`
- `AZURE_REDIS_PRIMARY_KEY`
- `AZURE_REDIS_SECONDARY_KEY`

### Azure Document Intelligence
- `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`
- `AZURE_DOCUMENT_INTELLIGENCE_API_KEY`

## 📄 Test Document

The test uses `TR_GIS_Corporativo.pdf` which should be placed in the project root directory.

## 🔧 Test Components

### 1. Document Processing
- **Enhanced Hybrid Chunker** with pattern detection
- **Automatic header cleaning** for repetitive content
- **Quality validation** and semantic overlap
- **Reference preservation** across chunks

### 2. Vector Services
- **Azure Embedding Service** for vector generation
- **Azure AI Search** for hybrid search
- **Advanced Retrieval Service** with metadata filtering
- **Indexing Pipeline** for document storage

### 3. LLM Orchestrator
- **LangChain Integration** as primary system
- **Structured Outputs** for environmental queries
- **Citation Extraction** from retrieved documents
- **Cost Tracking** and monitoring

### 4. Memory Management
- **Cosmos DB** for persistent chat history
- **Redis Cache** for session management
- **Conversation Summarization** for long contexts
- **Hybrid Memory** combining buffer and summary

## 📊 Test Results

The test generates comprehensive reports including:

- **Performance Metrics**: Processing times for each component
- **Cost Analysis**: Token usage and costs breakdown
- **Pattern Detection**: Noise reduction and cleaning statistics
- **Memory Statistics**: Session and conversation metrics
- **Error Logging**: Detailed error information
- **Recommendations**: Optimization suggestions

## 📁 Output Files

- `complete_rag_chunks_YYYYMMDD_HHMMSS.json`: Generated chunks with metadata
- `complete_rag_flow_report_YYYYMMDD_HHMMSS.json`: Comprehensive test report
- `pattern_analysis_*.json`: Pattern detection analysis reports

## 🎯 Test Queries

The test includes predefined queries to validate:

1. **General Consultation**: "¿Cuáles son los objetivos del sistema GIS corporativo?"
2. **Process Analysis**: "¿Qué procesos de negocio se soportan actualmente?"
3. **Implementation**: "¿Cuáles son las fases de implementación del proyecto?"
4. **Technical Design**: "¿Qué incluye el diseño técnico del GIS?"
5. **Evaluation Criteria**: "¿Cuáles son los criterios de evaluación de propuestas?"

## 💬 Conversation Flow

The test also validates conversation flow with:

- **Session Management**: Persistent chat history
- **Context Preservation**: Memory across conversation turns
- **Citation Tracking**: References to source documents
- **Cost Monitoring**: Per-turn cost tracking

## 🔍 Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   ```bash
   python scripts/check_env_config.py
   ```

2. **Missing Test Document**
   - Ensure `TR_GIS_Corporativo.pdf` is in project root

3. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

4. **Azure Service Errors**
   - Verify Azure service endpoints and keys
   - Check service availability and quotas

### Debug Mode

Set `VERBOSE=true` in environment variables for detailed logging.

## 📈 Performance Expectations

- **Document Processing**: 30-60 seconds for 9-page PDF
- **Embedding Generation**: 5-10 seconds for ~10 chunks
- **Indexing**: 10-20 seconds for document upload
- **Retrieval**: 1-3 seconds per query
- **LLM Generation**: 5-15 seconds per response
- **Total Cost**: $0.01-$0.05 per complete test

## 🎉 Success Criteria

A successful test should show:

- ✅ All components initialized without errors
- ✅ Document processed with pattern detection
- ✅ Chunks generated with quality validation
- ✅ Embeddings created successfully
- ✅ Documents indexed in Azure AI Search
- ✅ Retrieval queries return relevant results
- ✅ LLM generates structured responses with citations
- ✅ Conversation flow maintains context and memory
- ✅ Cost tracking shows reasonable usage
- ✅ Comprehensive report generated

## 📚 Related Documentation

- [Azure AI Search Setup](../README_AZURE_SEARCH.md)
- [Environment Configuration](../env.example)
- [Core Components](../core/README.md)
