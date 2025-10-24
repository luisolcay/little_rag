"""
Base chunker with hybrid OCR capabilities.
"""

import os
import time
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod

from ..models import Chunk, DocumentFile
from ..text_processing.extractors import BaseExtractor, HybridExtractor
from ..text_processing.splitters import BaseSplitter, TokenBasedSplitter
from ..text_processing.ocr import BaseOcrProvider, AzureDocumentIntelligenceOcrProvider

class BaseChunker(ABC):
    """Base class for document chunkers."""
    
    @abstractmethod
    def chunk_document(self, doc_file: DocumentFile) -> List[Chunk]:
        """Chunk a document into smaller pieces."""
        pass

class HybridChunker(BaseChunker):
    """Hybrid chunker with intelligent OCR selection."""
    
    def __init__(
        self,
        extractor: BaseExtractor | None = None,
        splitter: BaseSplitter | None = None,
        docling_provider: BaseOcrProvider | None = None,
        azure_provider: BaseOcrProvider | None = None,
        max_tokens: int = 900,
        overlap_tokens: int = 120,
        ocr_threshold: float = 0.1,
        min_text_length: int = 50,
    ):
        self.extractor = extractor or HybridExtractor()
        self.splitter = splitter or TokenBasedSplitter(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        self.docling_provider = docling_provider
        self.azure_provider = azure_provider
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.ocr_threshold = ocr_threshold
        self.min_text_length = min_text_length
    
    def chunk_document(self, doc_file: DocumentFile) -> List[Chunk]:
        """Process a single document with hybrid OCR approach."""
        try:
            print(f"[HYBRID] Processing: {doc_file.blob_name}")
            
            # Extract text by pages using native extraction
            pages = self.extractor.extract(doc_file.local_path)
            print(f"[HYBRID] Native extraction: {len(pages)} pages")
            
            # Intelligent OCR selection (3-tier approach)
            processed_pages = []
            ocr_pages_needed = []
            
            for page_text, page_no in pages:
                analysis = self._analyze_page_text_quality(page_text)
                
                if analysis["needs_ocr"]:
                    print(f"[HYBRID] Page {page_no} needs OCR: {analysis['reason']}")
                    ocr_pages_needed.append(page_no)
                    processed_pages.append((page_text, page_no, analysis))
                else:
                    print(f"[HYBRID] Page {page_no} OK: {analysis['reason']} ({analysis['text_length']} chars)")
                    processed_pages.append((page_text, page_no, analysis))
            
            # Run intelligent OCR selection
            if ocr_pages_needed:
                is_complex = self._is_complex_document(pages)
                
                if is_complex and self.docling_provider:
                    print(f"[HYBRID] Using Docling OCR for complex document")
                    try:
                        ocr_pages = self.docling_provider.extract_text_per_page(doc_file.local_path)
                        ocr_dict = {page_no: text for text, page_no in ocr_pages}
                        
                        # Replace pages that needed OCR
                        for i, (page_text, page_no, analysis) in enumerate(processed_pages):
                            if page_no in ocr_dict:
                                new_text = ocr_dict[page_no]
                                new_analysis = self._analyze_page_text_quality(new_text)
                                processed_pages[i] = (new_text, page_no, new_analysis)
                                print(f"[HYBRID] Docling OCR completed for page {page_no}: {new_analysis['text_length']} chars")
                    except Exception as e:
                        print(f"[HYBRID] Docling failed, falling back to Azure: {e}")
                        if self.azure_provider:
                            try:
                                ocr_pages = self.azure_provider.extract_text_per_page(doc_file.local_path)
                                ocr_dict = {page_no: text for text, page_no in ocr_pages}
                                
                                # Replace pages that needed OCR
                                for i, (page_text, page_no, analysis) in enumerate(processed_pages):
                                    if page_no in ocr_dict:
                                        new_text = ocr_dict[page_no]
                                        new_analysis = self._analyze_page_text_quality(new_text)
                                        processed_pages[i] = (new_text, page_no, new_analysis)
                                        print(f"[HYBRID] Azure OCR completed for page {page_no}: {new_analysis['text_length']} chars")
                            except Exception as e2:
                                print(f"[HYBRID] Azure OCR also failed: {e2}, using native extraction")
                elif self.azure_provider:
                    print(f"[HYBRID] Using Azure OCR for standard document")
                    try:
                        ocr_pages = self.azure_provider.extract_text_per_page(doc_file.local_path)
                        ocr_dict = {page_no: text for text, page_no in ocr_pages}
                        
                        # Replace pages that needed OCR
                        for i, (page_text, page_no, analysis) in enumerate(processed_pages):
                            if page_no in ocr_dict:
                                new_text = ocr_dict[page_no]
                                new_analysis = self._analyze_page_text_quality(new_text)
                                processed_pages[i] = (new_text, page_no, new_analysis)
                                print(f"[HYBRID] Azure OCR completed for page {page_no}: {new_analysis['text_length']} chars")
                    except Exception as e:
                        print(f"[HYBRID] Azure OCR failed: {e}, using native extraction")
            
            # Combine all pages into single text
            all_text = "\n\n".join([text for text, _, _ in processed_pages])
            
            # Split into chunks
            chunks = self.splitter.split_text(all_text)
            
            # Convert to Chunk objects with metadata
            chunk_objects = []
            for i, chunk_text in enumerate(chunks):
                metadata = {
                    **doc_file.metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'document_name': doc_file.blob_name,
                    'processing_method': 'hybrid_chunker',
                    'ocr_pages_needed': ocr_pages_needed,
                    'total_pages': len(pages)
                }
                
                chunk_obj = Chunk(content=chunk_text, metadata=metadata)
                chunk_objects.append(chunk_obj)
            
            print(f"[HYBRID] Generated {len(chunk_objects)} chunks")
            return chunk_objects
            
        except Exception as e:
            print(f"[HYBRID] Error processing {doc_file.blob_name}: {e}")
            raise
    
    def _analyze_page_text_quality(self, text: str) -> Dict[str, Any]:
        """Analyze text quality to determine if OCR is needed."""
        text_length = len(text.strip())
        
        if text_length < self.min_text_length:
            return {
                "needs_ocr": True,
                "reason": f"Text too short ({text_length} chars)",
                "text_length": text_length,
                "confidence": 0.9
            }
        
        # Check for OCR indicators
        ocr_indicators = [
            text.count(' ') / max(len(text), 1) > 0.3,  # Too many spaces
            text.count('\n') / max(len(text), 1) > 0.1,  # Too many line breaks
            len([c for c in text if c.isupper()]) / max(len(text), 1) > 0.5,  # Too many uppercase
        ]
        
        if any(ocr_indicators):
            return {
                "needs_ocr": True,
                "reason": "OCR indicators detected",
                "text_length": text_length,
                "confidence": 0.7
            }
        
        return {
            "needs_ocr": False,
            "reason": "Text quality acceptable",
            "text_length": text_length,
            "confidence": 0.8
        }
    
    def _is_complex_document(self, pages: List[Tuple[str, int]]) -> bool:
        """Detect if document has tables or complex structure."""
        for text, _ in pages[:3]:  # Check first 3 pages
            # Look for table indicators
            if any(indicator in text.lower() for indicator in
                   ['table', 'figura', 'cuadro', 'tabla', '|', '─', '┌', '┐', '└', '┘']):
                return True
        return False
