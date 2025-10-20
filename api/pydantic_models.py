from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional, List, Dict, Any

class ModelType(str, Enum):
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"

class QueryInput(BaseModel):
    question: str
    session_id: Optional[str] = Field(default=None)
    model: ModelType = Field(default=ModelType.GPT4_O_MINI)
    use_context: bool = Field(default=True)  # NUEVO

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelType
    citations: Optional[List[Dict[str, Any]]] = []  # NUEVO

class DocumentMetadata(BaseModel):  # NUEVO
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