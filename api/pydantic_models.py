from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

class ModelType(str, Enum):
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"

# === EXISTING MODELS ===
class QueryInput(BaseModel):
    question: str
    session_id: Optional[str] = Field(default=None)
    model: ModelType = Field(default=ModelType.GPT4_O_MINI)
    use_context: bool = Field(default=True)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelType
    citations: Optional[List[Dict[str, Any]]] = []

class DocumentMetadata(BaseModel):
    id: str
    filename: str
    blob_name: str
    blob_url: str
    upload_timestamp: str
    processing_status: str
    chunks_count: int
    needs_ocr: bool

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int

# === NEW ADVANCED RAG MODELS ===

# Document Processing Models
class DocumentUploadRequest(BaseModel):
    """Request for document upload."""
    filename: str = Field(..., min_length=1, max_length=255)
    content_type: str = Field(default="application/pdf")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class DocumentProcessingRequest(BaseModel):
    """Request for document processing with advanced chunking."""
    document_id: str
    processing_options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    chunking_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    enable_pattern_detection: bool = Field(default=True)
    enable_quality_validation: bool = Field(default=True)
    enable_reference_preservation: bool = Field(default=True)

class ChunkInfo(BaseModel):
    """Information about a generated chunk."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    quality_score: Optional[float] = None
    embedding: Optional[List[float]] = None
    page_number: Optional[int] = None
    chunk_index: int
    cleaning_applied: bool = False
    noise_reduction: int = 0

class PatternAnalysisResult(BaseModel):
    """Results from pattern detection analysis."""
    noise_percentage: float
    similarity_patterns_count: int
    header_patterns_count: int
    common_lines_count: int
    cleaning_applied: bool
    recommendations: List[str]
    most_common_line: Optional[str] = None

class QualityMetrics(BaseModel):
    """Quality metrics for chunks."""
    total_chunks: int
    valid_chunks: int
    invalid_chunks: int
    average_quality_score: float
    quality_distribution: Dict[str, int]
    most_common_issues: List[tuple]
    processing_time: float

class DocumentProcessingResponse(BaseModel):
    """Response from document processing."""
    document_id: str
    success: bool
    chunks: List[ChunkInfo]
    pattern_analysis: Optional[PatternAnalysisResult] = None
    quality_metrics: Optional[QualityMetrics] = None
    processing_stats: Dict[str, Any]
    error_message: Optional[str] = None

# Embedding Models
class EmbeddingGenerationRequest(BaseModel):
    """Request for embedding generation."""
    chunk_ids: List[str]
    batch_size: int = Field(default=100, ge=1, le=1000)
    model: str = Field(default="text-embedding-3-large")

class EmbeddingGenerationResponse(BaseModel):
    """Response from embedding generation."""
    success: bool
    embeddings_generated: int
    total_tokens: int
    estimated_cost: float
    processing_time: float
    error_message: Optional[str] = None

class EmbeddingStatus(BaseModel):
    """Status of embedding generation."""
    total_chunks: int
    embedded_chunks: int
    pending_chunks: int
    failed_chunks: int
    progress_percentage: float

# Search Models
class SearchRequest(BaseModel):
    """Request for hybrid search."""
    query: str = Field(..., min_length=1, max_length=2000)
    query_vector: Optional[List[float]] = None
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None
    search_type: str = Field(default="hybrid")  # hybrid, semantic, keyword

class SearchResult(BaseModel):
    """Individual search result."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    relevance_score: float
    document_name: str
    page_number: Optional[int] = None

class SearchResponse(BaseModel):
    """Response from search."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    search_type: str

class SearchSuggestion(BaseModel):
    """Search suggestion."""
    suggestion: str
    confidence: float
    type: str  # query, document, topic

# Memory Models
class MemorySession(BaseModel):
    """Memory session information."""
    session_id: str
    created_at: datetime
    last_activity: datetime
    message_count: int
    total_tokens: int
    summary: Optional[str] = None

class ConversationMessage(BaseModel):
    """Individual conversation message."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class MemorySessionResponse(BaseModel):
    """Response for memory session operations."""
    session_id: str
    success: bool
    messages: List[ConversationMessage]
    summary: Optional[str] = None
    error_message: Optional[str] = None

# LLM Models
class LLMRequest(BaseModel):
    """Enhanced LLM request."""
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    retrieved_chunks: Optional[List[SearchResult]] = None
    conversation_history: Optional[List[ConversationMessage]] = None
    model: ModelType = Field(default=ModelType.GPT4_O_MINI)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=8192)
    enable_streaming: bool = Field(default=False)

class LLMResponse(BaseModel):
    """Enhanced LLM response."""
    response: str
    session_id: str
    model: ModelType
    citations: List[Dict[str, Any]]
    cost_info: Dict[str, Any]
    processing_time: float
    tokens_used: int

# Analytics Models
class ProcessingAnalytics(BaseModel):
    """Analytics for document processing."""
    total_documents: int
    total_chunks: int
    average_chunk_length: float
    average_quality_score: float
    total_processing_time: float
    documents_with_cleaning: int
    total_noise_reduction: int

class SearchAnalytics(BaseModel):
    """Analytics for search operations."""
    total_searches: int
    average_search_time: float
    most_common_queries: List[tuple]
    search_type_distribution: Dict[str, int]
    average_results_per_query: float

class CostAnalytics(BaseModel):
    """Analytics for Azure costs."""
    total_embedding_tokens: int
    total_embedding_cost: float
    total_llm_tokens: int
    total_llm_cost: float
    total_cost: float
    cost_by_service: Dict[str, float]

class SystemHealth(BaseModel):
    """System health status."""
    azure_openai_status: str
    azure_search_status: str
    azure_cosmos_status: str
    azure_redis_status: str
    overall_status: str
    last_check: datetime

# Configuration Models
class ChunkingConfig(BaseModel):
    """Configuration for chunking parameters."""
    max_tokens: int = Field(default=900, ge=100, le=2000)
    overlap_tokens: int = Field(default=120, ge=0, le=500)
    min_chunk_length: int = Field(default=50, ge=10, le=500)
    max_chunk_length: int = Field(default=2000, ge=500, le=5000)
    preserve_paragraphs: bool = Field(default=True)
    overlap_sentences: int = Field(default=2, ge=0, le=10)

class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    enable_pattern_detection: bool = Field(default=True)
    enable_quality_validation: bool = Field(default=True)
    enable_reference_preservation: bool = Field(default=True)
    auto_clean_headers: bool = Field(default=True)
    noise_threshold: float = Field(default=10.0, ge=0.0, le=50.0)
    min_quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class AzureConfig(BaseModel):
    """Configuration for Azure services."""
    openai_endpoint: str
    openai_api_key: str
    search_endpoint: str
    search_api_key: str
    cosmos_endpoint: str
    cosmos_key: str
    redis_host: str
    redis_key: str