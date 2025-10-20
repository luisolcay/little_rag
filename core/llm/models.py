"""
Pydantic models for LLM interactions and environmental consulting schemas.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Union, Literal
from datetime import datetime
from enum import Enum
import uuid

class EnvironmentalQueryType(str, Enum):
    """Types of environmental consulting queries."""
    COMPLIANCE = "compliance"
    RISK_ASSESSMENT = "risk_assessment"
    TECHNICAL_ANALYSIS = "technical_analysis"
    GENERAL_QA = "general_qa"
    REGULATORY_CHECK = "regulatory_check"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    # Structured output types
    STRUCTURED_COMPLIANCE = "structured_compliance"
    STRUCTURED_RISK_ASSESSMENT = "structured_risk_assessment"
    STRUCTURED_CITATION = "structured_citation"

class ModelName(str, Enum):
    """Available LLM models."""
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"

class LLMRequest(BaseModel):
    """Request for LLM processing."""
    query: str = Field(..., description="User query")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_type: Optional[EnvironmentalQueryType] = None
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=8192)
    stream: bool = Field(default=False)
    structured_output: bool = Field(default=False)
    context_chunks: Optional[List[Dict[str, Any]]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

class StreamingChunk(BaseModel):
    """Streaming response chunk."""
    content: str
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None

class LLMResponse(BaseModel):
    """Response from LLM processing."""
    content: str
    session_id: str
    model: ModelName
    query_type: Optional[EnvironmentalQueryType]
    tokens_used: int
    processing_time: float
    confidence_score: Optional[float] = None
    citations: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)

class Citation(BaseModel):
    """Citation reference."""
    chunk_id: str
    document_name: str
    page_number: Optional[int] = None
    content_snippet: str
    relevance_score: float
    metadata: Optional[Dict[str, Any]] = None

class ComplianceReport(BaseModel):
    """Structured compliance report."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    compliance_status: Literal["compliant", "non_compliant", "requires_review", "insufficient_data"]
    applicable_regulations: List[str]
    findings: List[str]
    recommendations: List[str]
    risk_level: Literal["low", "medium", "high", "critical"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    citations: List[Citation]
    generated_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class RiskAssessment(BaseModel):
    """Structured risk assessment."""
    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    risk_category: Literal["environmental", "regulatory", "operational", "financial", "reputation"]
    risk_level: Literal["low", "medium", "high", "critical"]
    probability: float = Field(ge=0.0, le=1.0, description="Risk probability")
    impact: float = Field(ge=0.0, le=1.0, description="Risk impact")
    risk_score: float = Field(ge=0.0, le=1.0, description="Overall risk score")
    risk_factors: List[str]
    mitigation_strategies: List[str]
    monitoring_recommendations: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    citations: List[Citation]
    generated_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class TechnicalAnalysis(BaseModel):
    """Structured technical analysis."""
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    analysis_type: Literal["environmental_impact", "technical_feasibility", "regulatory_compliance", "cost_benefit"]
    key_findings: List[str]
    technical_recommendations: List[str]
    implementation_considerations: List[str]
    data_requirements: List[str]
    limitations: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    citations: List[Citation]
    generated_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class GeneralQA(BaseModel):
    """General Q&A response."""
    response_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    related_topics: List[str]
    follow_up_questions: List[str]
    citations: List[Citation]
    generated_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class StructuredOutput(BaseModel):
    """Union of all structured output types."""
    output_type: Literal["compliance_report", "risk_assessment", "technical_analysis", "general_qa"]
    data: Union[ComplianceReport, RiskAssessment, TechnicalAnalysis, GeneralQA]

class ConversationContext(BaseModel):
    """Conversation context with memory references."""
    session_id: str
    messages: List[Dict[str, str]]
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class PromptVersion(BaseModel):
    """Prompt version metadata."""
    version: str
    prompt_type: str
    template: str
    variables: List[str]
    author: str
    created_at: datetime = Field(default_factory=datetime.now)
    changelog: str
    is_active: bool = False
    performance_metrics: Optional[Dict[str, Any]] = None

class LLMMetrics(BaseModel):
    """LLM performance metrics."""
    session_id: str
    model: ModelName
    query_type: EnvironmentalQueryType
    tokens_used: int
    processing_time: float
    cost_usd: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

