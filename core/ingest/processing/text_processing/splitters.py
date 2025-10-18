import re
from typing import List
from .splitter import BaseSplitter


class SemanticOverlapSplitter(BaseSplitter):
    """Splitter with semantic-aware overlap preservation"""
    
    def __init__(self, max_tokens: int = 900, overlap_sentences: int = 2, 
                 preserve_paragraphs: bool = True):
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.preserve_paragraphs = preserve_paragraphs
        self.encoder = None
        
        # Load tokenizer
        try:
            import tiktoken
            self.encoder = tiktoken.encoding_for_model("gpt-4o-mini")
        except ImportError:
            pass
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken or fallback to character count"""
        if self.encoder:
            return len(self.encoder.encode(text))
        return len(text)  # Fallback to characters
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences preserving punctuation"""
        # Enhanced sentence splitting that handles various cases
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Za-zÁÉÍÓÚÜÑ0-9])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _is_paragraph_boundary(self, text: str, position: int) -> bool:
        """Check if position is at a paragraph boundary"""
        if position == 0 or position >= len(text):
            return False
        
        # Look for double newlines or significant whitespace
        before = text[max(0, position-10):position]
        after = text[position:min(len(text), position+10)]
        
        return '\n\n' in before or '\n\n' in after
    
    def _should_preserve_structure(self, text: str) -> bool:
        """Determine if text has structural elements worth preserving"""
        # Check for common structural patterns
        structural_patterns = [
            r'^\d+\.\s+',  # Numbered lists
            r'^[A-Z][a-z]+:',  # Headers
            r'^\*\s+',  # Bullet points
            r'^-\s+',  # Dashes
            r'^\s*[A-Z][A-Z\s]+$',  # ALL CAPS headers
        ]
        
        lines = text.split('\n')
        structural_lines = 0
        
        for line in lines[:10]:  # Check first 10 lines
            for pattern in structural_patterns:
                if re.match(pattern, line.strip()):
                    structural_lines += 1
                    break
        
        return structural_lines >= 2
    
    def split(self, text: str) -> List[str]:
        """Split text with semantic overlap and structure preservation"""
        if not text.strip():
            return []
        
        # Determine splitting strategy based on content
        has_structure = self._should_preserve_structure(text)
        
        if has_structure and self.preserve_paragraphs:
            return self._split_with_paragraph_awareness(text)
        else:
            return self._split_with_sentence_awareness(text)
    
    def _split_with_sentence_awareness(self, text: str) -> List[str]:
        """Split text using sentence boundaries with semantic overlap"""
        sentences = self._split_by_sentences(text)
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)
            
            # Check if adding this sentence exceeds limit
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                # Create current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Create semantic overlap
                overlap_sentences = current_chunk[-self.overlap_sentences:]
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self._count_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _split_with_paragraph_awareness(self, text: str) -> List[str]:
        """Split text preserving paragraph structure"""
        paragraphs = self._split_by_paragraphs(text)
        if not paragraphs:
            return [text]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_tokens = self._count_tokens(paragraph)
            
            # Check if adding this paragraph exceeds limit
            if current_tokens + paragraph_tokens > self.max_tokens and current_chunk:
                # Create current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                # For paragraphs, use smaller overlap
                overlap_paragraphs = current_chunk[-1:] if len(current_chunk) > 1 else []
                current_chunk = overlap_paragraphs + [paragraph]
                current_tokens = sum(self._count_tokens(p) for p in current_chunk)
            else:
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def get_split_statistics(self, text: str) -> dict:
        """Get statistics about the splitting process"""
        sentences = self._split_by_sentences(text)
        paragraphs = self._split_by_paragraphs(text)
        chunks = self.split(text)
        
        return {
            'original_length': len(text),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'chunk_count': len(chunks),
            'average_chunk_length': sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
            'has_structure': self._should_preserve_structure(text),
            'preserve_paragraphs': self.preserve_paragraphs
        }
