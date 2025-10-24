"""
Data models for document processing.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

@dataclass
class DocumentFile:
    """Represents a document file ready for processing."""
    
    local_path: str
    blob_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    needs_ocr: bool = False
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}
        
        # Add default metadata if not present
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.now().isoformat()
        
        if 'chunk_length' not in self.metadata:
            self.metadata['chunk_length'] = len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create chunk from dictionary."""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            content=data['content'],
            metadata=data.get('metadata', {})
        )
