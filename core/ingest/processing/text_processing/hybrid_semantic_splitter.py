"""
Hybrid Semantic Text Splitter
==============================

Intelligent chunking system that automatically selects between:
1. LangChain RecursiveCharacterTextSplitter (fast, structural)
2. True semantic chunking with embeddings (precise, topic-based)

Features:
- Auto-detection of document complexity
- Semantic coherence grouping
- Automatic fallback
- Embedding caching
- Batch processing for performance
"""

from typing import List, Dict, Optional
import re
import numpy as np
from abc import ABC

# Import LangChain splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("[WARNING] langchain-text-splitters not available")

# Import sentence transformers for semantic analysis
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("[WARNING] sentence-transformers or scikit-learn not available")

# Import base splitter
from .splitters import BaseSplitter


class ScalableHybridSplitter(BaseSplitter):
    """
    Intelligent hybrid splitter that automatically selects the best strategy.
    
    Decision process:
    1. Analyze document complexity
    2. Choose: LangChain (simple) or Semantic (complex)
    3. Apply chunking with automatic fallback
    """
    
    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 200,
        semantic_threshold: float = 0.6,
        use_caching: bool = True,
        model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    ):
        """
        Initialize hybrid semantic splitter.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks (tokens)
            semantic_threshold: Similarity threshold for grouping (0.0-1.0)
            use_caching: Enable embedding cache
            model_name: Sentence transformer model name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_threshold = semantic_threshold
        self.use_caching = use_caching
        self.model_name = model_name
        
        # Cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {} if use_caching else {}
        
        # Initialize LangChain splitter (structural)
        if LANGCHAIN_AVAILABLE:
            self.langchain_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
                is_separator_regex=False
            )
        else:
            self.langchain_splitter = None
        
        # Initialize semantic model (lazy loading)
        self.semantic_model: Optional[SentenceTransformer] = None
        if SEMANTIC_AVAILABLE:
            try:
                print(f"[HYBRID_SPLITTER] Loading semantic model: {model_name}")
                self.semantic_model = SentenceTransformer(model_name)
                print("[HYBRID_SPLITTER] [OK] Semantic model loaded")
            except Exception as e:
                print(f"[HYBRID_SPLITTER] [ERROR] Failed to load semantic model: {e}")
                self.semantic_model = None
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text intelligently based on content analysis.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Step 1: Decide strategy
        use_semantic = self._should_use_semantic(text)
        
        print(f"[HYBRID_SPLITTER] {'Using SEMANTIC' if use_semantic else 'Using LANGCHAIN'} chunking")
        
        # Step 2: Apply strategy with fallback
        if use_semantic and SEMANTIC_AVAILABLE and self.semantic_model:
            try:
                return self._semantic_split(text)
            except Exception as e:
                print(f"[HYBRID_SPLITTER] [ERROR] Semantic chunking failed: {e}")
                print("[HYBRID_SPLITTER] [FALLBACK] Falling back to LangChain")
                return self._langchain_split(text)
        else:
            return self._langchain_split(text)
    
    def _should_use_semantic(self, text: str) -> bool:
        """
        Heuristics to decide if semantic chunking is worth it.
        
        Returns True if document is:
        - Long (>2000 chars)
        - Dense (high sentence/paragraph ratio)
        - Technical content (specific terms)
        """
        text_length = len(text)
        
        # Very short documents don't need semantic analysis
        if text_length < 2000:
            return False
        
        # Count paragraphs and sentences
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentence_count = len(re.findall(r'[.!?]', text))
        
        # Dense content = many sentences per paragraph
        density = sentence_count / max(len(paragraphs), 1)
        
        # High density suggests semantic analysis needed
        if density > 10:
            return True
        
        # Technical documents → semantic chunking
        technical_terms = [
            'metodología', 'metodology',
            'procedimiento', 'procedure',
            'methodology', 'framework',
            'approach', 'metodología para',
            'proceso de', 'estrategia de',
            'linea base', 'línea base'
        ]
        
        text_lower = text.lower()
        if any(term in text_lower for term in technical_terms):
            return True
        
        return False
    
    def _semantic_split(self, text: str) -> List[str]:
        """
        Apply semantic chunking using embeddings.
        
        Process:
        1. Split into sentences
        2. Generate embeddings
        3. Calculate similarity between adjacent sentences
        4. Group similar sentences (same topic)
        5. Create coherent chunks
        """
        # Check cache first
        if self.use_caching and text in self._embedding_cache:
            print("[HYBRID_SPLITTER] Cache hit!")
            return self._group_from_embeddings(text, self._embedding_cache[text])
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        print(f"[HYBRID_SPLITTER] Processing {len(sentences)} sentences semantically")
        
        # Generate embeddings (batch processing)
        try:
            embeddings = self.semantic_model.encode(
                sentences,
                show_progress_bar=False,
                batch_size=32,
                convert_to_numpy=True
            )
            
            # Cache embeddings
            if self.use_caching:
                self._embedding_cache[text] = embeddings
            
            # Calculate similarity between adjacent sentences
            similarities = self._calculate_similarity(embeddings)
            
            # Group by semantic coherence
            chunks = self._group_by_similarity(sentences, similarities)
            
            print(f"[HYBRID_SPLITTER] Created {len(chunks)} semantic chunks")
            
            return chunks
            
        except Exception as e:
            print(f"[HYBRID_SPLITTER] Error in semantic processing: {e}")
            raise
    
    def _langchain_split(self, text: str) -> List[str]:
        """Use LangChain splitter for fast structural chunking."""
        if not LANGCHAIN_AVAILABLE or not self.langchain_splitter:
            # Fallback: simple split by paragraphs
            return self._simple_split(text)
        
        chunks = self.langchain_splitter.split_text(text)
        print(f"[HYBRID_SPLITTER] Created {len(chunks)} structural chunks (LangChain)")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Enhanced sentence splitting."""
        # Pattern for sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Skip very short
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _calculate_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate similarity between adjacent sentences."""
        if len(embeddings) < 2:
            return np.array([])
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity(
                embeddings[i:i+1],
                embeddings[i+1:i+2]
            )[0][0]
            similarities.append(sim)
        
        return np.array(similarities)
    
    def _group_by_similarity(
        self,
        sentences: List[str],
        similarities: np.ndarray
    ) -> List[str]:
        """
        Group sentences into semantically coherent chunks.
        
        Algorithm:
        1. Start new chunk when similarity drops below threshold
        2. Respect maximum chunk size (tokens)
        3. Include overlap from previous chunk
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk = [sentences[0]]
        current_tokens = len(sentences[0].split())
        
        # Process remaining sentences with similarity info
        for i, (sentence, similarity) in enumerate(zip(sentences[1:], similarities)):
            sentence_tokens = len(sentence.split())
            
            # Decision: should we start a new chunk?
            new_chunk = (
                similarity < self.semantic_threshold or  # Low similarity = new topic
                current_tokens + sentence_tokens > self.chunk_size  # Size limit
            )
            
            if new_chunk and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap (last 2 sentences)
                overlap = self._get_overlap_sentences(current_chunk, max_sentences=2)
                current_chunk = overlap + [sentence]
                current_tokens = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _get_overlap_sentences(self, chunk: List[str], max_sentences: int = 2) -> List[str]:
        """Get last N sentences for overlap."""
        return chunk[-max_sentences:] if len(chunk) >= max_sentences else chunk
    
    def _group_from_embeddings(self, text: str, embeddings: np.ndarray) -> List[str]:
        """Group using cached embeddings."""
        sentences = self._split_into_sentences(text)
        similarities = self._calculate_similarity(embeddings)
        return self._group_by_similarity(sentences, similarities)
    
    def _simple_split(self, text: str) -> List[str]:
        """Simple fallback: split by paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs if paragraphs else [text]
    
    def clear_cache(self):
        """Clear embedding cache."""
        if self.use_caching:
            self._embedding_cache.clear()
            print("[HYBRID_SPLITTER] Cache cleared")

