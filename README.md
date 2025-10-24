# 🎯 Collahuasi RAG System - Complete Azure Integration

## 🚀 Overview

A production-ready RAG (Retrieval-Augmented Generation) system with complete Azure services integration, featuring end-to-end citation system, document processing pipeline, and intelligent retrieval.

### ✅ Azure Services Integrated

- **Azure Blob Storage** - Document storage and management
- **Azure Document Intelligence** - OCR for scanned PDFs
- **Azure OpenAI** - LLM responses and embeddings
- **Azure AI Search** - Hybrid search (vector + keyword + semantic)
- **Azure Cosmos DB** - Conversation memory and document tracking
- **Azure Redis Cache** - Session caching and performance optimization

## 🏗️ Architecture

```
Upload → Azure Blob Storage → FileLoader → OCR (si necesario)
                                              ↓
                                          Chunker
                                              ↓
                                    Azure OpenAI Embeddings
                                              ↓
                                      Azure AI Search Index
                                              ↓
Query → Hybrid Search → Context + Citations → LLM (Azure OpenAI)
                                              ↓
                                    Cosmos DB (memoria)
                                              ↓
                                    Response con Citations
                                              ↓
                                          Frontend
```

## 🎯 Key Features

### 📚 Complete Citation System
- **End-to-end citations**: From retrieval to frontend display
- **Rich metadata**: Document name, page numbers, sections, relevance scores
- **Multiple display options**: Expandable panels, content snippets
- **Relevance scoring**: Automatic calculation based on content overlap

### ⚙️ RAG Configuration
- **Context toggle**: Users can disable document context
- **Model selection**: GPT-4o vs GPT-4o-mini
- **Real-time feedback**: Processing status, OCR usage, chunk counts

### 📄 Document Management
- **Complete pipeline**: Upload → Blob Storage → OCR → Chunking → Indexing
- **Status tracking**: Processing status, chunk counts, OCR usage
- **Document listing**: From Cosmos DB with metadata
- **Azure AI Search status**: Document count, storage size

### 🔄 Robust Error Handling
- **Circuit breaker**: Prevents cascading failures
- **Retry logic**: Exponential backoff for transient errors
- **Graceful fallback**: Memory services continue if one fails
- **Comprehensive logging**: Detailed error tracking

## 🚀 Quick Start

### 1. Environment Setup

Copy and configure your environment variables:

```bash
cp env.example .env
```

Edit `.env` with your Azure service credentials:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=your_search_key

# Azure Cosmos DB (NoSQL API)
AZURE_COSMOS_ENDPOINT=https://your-cosmos.documents.azure.com:443/
AZURE_COSMOS_KEY=your_cosmos_key
AZURE_COSMOS_DATABASE_NAME=rag_database

# Azure Redis Cache
AZURE_REDIS_ENDPOINT=your-redis.redis.cache.windows.net:6380
AZURE_REDIS_KEY=your_redis_key

# Azure Blob Storage
AZURE_BLOB_ENDPOINT=https://yourstorage.blob.core.windows.net
AZURE_BLOB_CONTAINER=orbe

# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=your_doc_intelligence_key
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the System

#### Option A: Automated Startup (Recommended)
```bash
python start_rag_system.py
```

This will:
- Check environment configuration
- Start FastAPI backend on port 8000
- Start Streamlit frontend on port 8501
- Monitor both services

#### Option B: Manual Startup

**Terminal 1 - Backend:**
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd app
streamlit run streamlit_app.py
```

### 4. Access the System

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🧪 Testing

### Quick System Test
```bash
python test_rag_system.py
```

This will test:
- Backend health
- Chat endpoint
- Document upload
- Citation system
- Context retrieval

### Component Tests
```bash
python test_rag_components.py
```

Tests individual components without external dependencies.

## 📖 Usage Guide

### 1. Upload Documents

1. Open the frontend at http://localhost:8501
2. Use the sidebar to upload documents (PDF, DOCX, HTML, TXT)
3. Monitor processing status and chunk creation
4. Verify OCR usage for scanned PDFs

### 2. Configure RAG Settings

- **Use Context**: Toggle to enable/disable document context
- **Model Selection**: Choose between GPT-4o and GPT-4o-mini
- **Monitor Status**: Check Azure AI Search statistics

### 3. Ask Questions

1. Type your question in the chat interface
2. View the response with citations
3. Expand citations to see source details
4. Check relevance scores and content snippets

### 4. Monitor System

- **Document List**: View all uploaded documents
- **Processing Status**: Track upload and indexing progress
- **Search Statistics**: Monitor Azure AI Search performance

## 🔧 API Endpoints

### Core Endpoints

- `POST /chat` - Chat with RAG system
- `POST /upload-doc` - Upload and process documents
- `GET /documents` - List all documents
- `GET /index/status` - Azure AI Search status
- `GET /health` - System health check

### Example API Usage

**Chat Request:**
```json
{
  "question": "What are the environmental compliance requirements?",
  "model": "gpt-4o-mini",
  "use_context": true,
  "session_id": "optional-session-id"
}
```

**Chat Response:**
```json
{
  "answer": "Environmental compliance requirements include...",
  "session_id": "generated-session-id",
  "model": "gpt-4o-mini",
  "citations": [
    {
      "document_name": "compliance_guide.pdf",
      "page_number": 1,
      "relevance_score": 0.85,
      "content_snippet": "Environmental regulations require..."
    }
  ]
}
```

## 🏗️ Project Structure

```
RAG/
├── core/                    # Core RAG components
│   ├── ingest/             # Document processing
│   │   ├── document_processor.py
│   │   ├── blob_utils.py
│   │   └── processing/
│   ├── llm/                # LLM orchestration
│   │   ├── orchestrator.py
│   │   ├── citation_extractor.py
│   │   └── models.py
│   ├── memory/             # Memory management
│   │   ├── memory_manager.py
│   │   ├── cosmos_store.py
│   │   └── redis_cache.py
│   └── vector/              # Vector search
│       ├── retrieval_service.py
│       └── indexing/
├── api/                    # FastAPI backend
│   ├── main.py
│   ├── pydantic_models.py
│   ├── llm_endpoints.py
│   └── admin_endpoints.py
├── app/                    # Streamlit frontend
│   ├── streamlit_app.py
│   ├── chat_interface.py
│   ├── sidebar.py
│   └── api_utils.py
├── start_rag_system.py     # Automated startup
├── test_rag_system.py      # System tests
└── test_rag_components.py  # Component tests
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **Azure Credentials**: Verify all environment variables are set correctly
3. **Port Conflicts**: Check that ports 8000 and 8501 are available
4. **Blob Storage Permissions**: Ensure your account has write access to the container

### Debug Mode

Enable debug logging by setting:
```env
LOG_LEVEL=DEBUG
```

### Service Status

Check service health:
```bash
curl http://localhost:8000/health
```

## 🚀 Production Deployment

### Prerequisites

1. **Azure Services**: All services must be configured and accessible
2. **Environment**: Production environment variables configured
3. **Monitoring**: Application insights and logging configured
4. **Security**: Proper authentication and authorization

### Deployment Steps

1. **Configure Production Environment**:
   ```bash
   export ENVIRONMENT=production
   export LOG_LEVEL=INFO
   ```

2. **Start Services**:
   ```bash
   python start_rag_system.py
   ```

3. **Monitor Performance**:
   - Check Azure service metrics
   - Monitor application logs
   - Verify citation accuracy

## 📊 Performance Metrics

### Expected Performance

- **Document Upload**: 2-5 seconds per document
- **OCR Processing**: 10-30 seconds for scanned PDFs
- **Query Response**: 2-8 seconds with context
- **Citation Extraction**: <1 second

### Monitoring

- **Azure AI Search**: Document count, query latency
- **Cosmos DB**: Request units, storage usage
- **Redis Cache**: Hit rate, memory usage
- **Blob Storage**: Storage usage, request count

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python test_rag_system.py`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check the logs for error details
4. Verify Azure service status

---

**🎉 The RAG system is now ready for production use with complete Azure integration!**
