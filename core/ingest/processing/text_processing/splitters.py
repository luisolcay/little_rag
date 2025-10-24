"""
Advanced Text Splitting Utilities
=================================

This module provides sophisticated text splitting algorithms
for optimal document chunking with semantic preservation.
"""

import re
import tiktoken
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

class BaseSplitter(ABC):
    """Base class for text splitters."""
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass

class TokenBasedSplitter(BaseSplitter):
    """
    Token-based text splitter with overlap.
    
    Features:
    - Accurate token counting using tiktoken
    - Configurable overlap for context preservation
    - Sentence boundary awareness
    - Performance optimization
    """
    
    def __init__(self, max_tokens: int = 900, overlap_tokens: int = 120, model: str = "gpt-4"):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.model = model
        
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into token-based chunks with overlap.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            # Check if adding this sentence would exceed max tokens
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                current_chunk, current_tokens = self._create_overlap_chunk(current_chunk)
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Enhanced sentence splitting regex
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Skip very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_overlap_chunk(self, previous_chunk: List[str]) -> Tuple[List[str], int]:
        """Create overlap chunk from previous chunk."""
        if not previous_chunk:
            return [], 0
        
        overlap_chunk = []
        overlap_tokens = 0
        
        # Add sentences from the end of previous chunk until we reach overlap_tokens
        for sentence in reversed(previous_chunk):
            sentence_tokens = len(self.encoding.encode(sentence))
            
            if overlap_tokens + sentence_tokens <= self.overlap_tokens:
                overlap_chunk.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_chunk, overlap_tokens

class SemanticOverlapSplitter(BaseSplitter):
    """
    Semantic-aware text splitter with intelligent overlap.
    
    Features:
    - Semantic boundary detection
    - Paragraph preservation
    - Context-aware splitting
    - Quality optimization
    """
    
    def __init__(
        self, 
        max_tokens: int = 900, 
        overlap_sentences: int = 2, 
        preserve_paragraphs: bool = True,
        model: str = "gpt-4"
    ):
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.preserve_paragraphs = preserve_paragraphs
        self.model = model
        
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text with semantic awareness and overlap.
        
        Args:
            text: Input text to split
            
        Returns:
            List of semantically coherent chunks
        """
        if not text.strip():
            return []
        
        # Split into paragraphs if preserve_paragraphs is enabled
        if self.preserve_paragraphs:
            paragraphs = self._split_into_paragraphs(text)
            chunks = self._split_paragraphs_into_chunks(paragraphs)
        else:
            # Direct sentence-based splitting
            sentences = self._split_into_sentences(text)
            chunks = self._split_sentences_into_chunks(sentences)
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 20:  # Skip very short paragraphs
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _split_paragraphs_into_chunks(self, paragraphs: List[str]) -> List[str]:
        """Split paragraphs into chunks while preserving paragraph boundaries."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = len(self.encoding.encode(para))
            
            # If single paragraph exceeds max tokens, split it
            if para_tokens > self.max_tokens:
                # Flush current chunk if it has content
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split the large paragraph
                para_chunks = self._split_large_paragraph(para)
                chunks.extend(para_chunks)
                continue
            
            # Check if adding this paragraph would exceed max tokens
            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                
                # Create overlap chunk
                current_chunk, current_tokens = self._create_paragraph_overlap(current_chunk)
            
            current_chunk.append(para)
            current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _split_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """Split sentences into chunks."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Create overlap chunk
                current_chunk, current_tokens = self._create_sentence_overlap(current_chunk)
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split a large paragraph into smaller chunks."""
        sentences = self._split_into_sentences(paragraph)
        return self._split_sentences_into_chunks(sentences)
    
    def _create_paragraph_overlap(self, previous_chunk: List[str]) -> Tuple[List[str], int]:
        """Create paragraph overlap chunk."""
        if not previous_chunk:
            return [], 0
        
        overlap_chunk = []
        overlap_tokens = 0
        
        # Add paragraphs from the end until we reach overlap threshold
        for para in reversed(previous_chunk):
            para_tokens = len(self.encoding.encode(para))
            
            if overlap_tokens + para_tokens <= self.max_tokens * 0.3:  # 30% overlap
                overlap_chunk.insert(0, para)
                overlap_tokens += para_tokens
            else:
                break
        
        return overlap_chunk, overlap_tokens
    
    def _create_sentence_overlap(self, previous_chunk: List[str]) -> Tuple[List[str], int]:
        """Create sentence overlap chunk."""
        if not previous_chunk:
            return [], 0
        
        overlap_chunk = []
        overlap_tokens = 0
        
        # Add sentences from the end until we reach overlap_sentences
        for sentence in reversed(previous_chunk[-self.overlap_sentences:]):
            sentence_tokens = len(self.encoding.encode(sentence))
            
            if overlap_tokens + sentence_tokens <= self.max_tokens * 0.3:
                overlap_chunk.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_chunk, overlap_tokens

class AdaptiveSplitter(BaseSplitter):
    """
    Adaptive splitter that chooses the best splitting strategy.
    
    Features:
    - Automatic strategy selection
    - Content type detection
    - Quality optimization
    - Performance monitoring
    """
    
    def __init__(self, max_tokens: int = 900, overlap_tokens: int = 120):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        # Initialize different splitters
        self.token_splitter = TokenBasedSplitter(max_tokens, overlap_tokens)
        self.semantic_splitter = SemanticOverlapSplitter(max_tokens, overlap_tokens // 60)
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text using the most appropriate strategy.
        
        Args:
            text: Input text to split
            
        Returns:
            List of optimally split chunks
        """
        if not text.strip():
            return []
        
        # Analyze text characteristics
        text_type = self._analyze_text_type(text)
        
        # Choose appropriate splitter
        if text_type == 'structured':
            # Use semantic splitter for structured content
            return self.semantic_splitter.split_text(text)
        else:
            # Use token-based splitter for general content
            return self.token_splitter.split_text(text)
    
    def _analyze_text_type(self, text: str) -> str:
        """Analyze text to determine its type."""
        # Check for structured content indicators
        structured_indicators = [
            r'\d+\.\s+',  # Numbered lists
            r'•\s+',      # Bullet points
            r'-\s+',      # Dashes
            r'\n\s*\n',   # Multiple paragraphs
            r'Table\s+\d+',  # Table references
            r'Figure\s+\d+', # Figure references
        ]
        
        structured_score = sum(1 for pattern in structured_indicators if re.search(pattern, text))
        
        if structured_score >= 2:
            return 'structured'
        else:
            return 'general'
