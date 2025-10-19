"""
Base chunker implementation with hybrid OCR capabilities.

This module consolidates the core chunking functionality from the original
chunker.py and hybrid_chunker.py files, providing a clean base implementation
that can be extended by more specialized chunkers.
"""

import os
import time
from typing import List, Tuple, Dict, Any

from ..models import Chunk, ChunkerError
from ..text_processing.extractors import BaseExtractor, AutoExtractor
from ..text_processing.ocr import BaseOcrProvider
from ..text_processing.splitter import BaseSplitter, TokenAwareSentenceSplitter
from ..text_processing.cleaning import normalize_text
from ...file_loader import DocumentFile


class BaseChunker:
    """
    Base chunker that orchestrates document processing:
    - Extracts text (by file type)
    - Optionally runs OCR if needed
    - Cleans text
    - Splits into smaller chunks
    - Returns a list of Chunk objects
    """
    
    def __init__(
        self,
        extractor: BaseExtractor | None = None,
        splitter: BaseSplitter | None = None,
        ocr_provider: BaseOcrProvider | None = None,
        max_tokens: int = 900,
        overlap_tokens: int = 120,
        ocr_threshold: float = 0.1,
        min_text_length: int = 50,
    ):
        self.extractor = extractor or AutoExtractor()
        self.splitter = splitter or TokenAwareSentenceSplitter(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        self.ocr_provider = ocr_provider
        self.ocr_threshold = ocr_threshold
        self.min_text_length = min_text_length

    def _should_try_ocr(self, doc_file: DocumentFile, pages: List[Tuple[str, int]]) -> bool:
        """Returns True if OCR should be attempted."""
        ext = os.path.splitext(doc_file.local_path)[1].lower()
        all_empty = all(not (t.strip()) for t, _ in pages)
        return bool(self.ocr_provider) and (doc_file.needs_ocr or (ext == ".pdf" and all_empty))

    def _is_complex_document(self, pages: List[Tuple[str, int]]) -> bool:
        """Detect if document has tables or complex structure"""
        for text, _ in pages[:3]:  # Check first 3 pages
            # Look for table indicators
            if any(indicator in text.lower() for indicator in 
                   ['table', 'figura', 'cuadro', 'tabla', '|', '─', '┌', '┐', '└', '┘']):
                return True
        return False

    def _make_document_id(self, doc_file: DocumentFile) -> str:
        """Generate a deterministic document ID."""
        import hashlib, json
        base = {
            "blob_name": doc_file.blob_name,
            "size_bytes": doc_file.metadata.get("size_bytes"),
            "original_ext": doc_file.metadata.get("original_ext"),
        }
        return hashlib.sha1(json.dumps(base, sort_keys=True).encode("utf-8")).hexdigest()

    def chunk_document(self, doc_file: DocumentFile) -> List[Chunk]:
        """Process a single document and return chunks."""
        try:
            # 1️⃣ Extract text by pages
            pages = self.extractor.extract(doc_file.local_path)

            # 2️⃣ Run OCR fallback if needed
            if self._should_try_ocr(doc_file, pages):
                pages = self.ocr_provider.extract_text_per_page(doc_file.local_path)

            # 3️⃣ Normalize + Split
            out: List[Chunk] = []
            global_idx = 0
            for page_text, page_no in pages:
                clean = normalize_text(page_text)
                if not clean:
                    continue
                pieces = self.splitter.split(clean)
                for i, piece in enumerate(pieces):
                    meta = {
                        "document_blob": doc_file.blob_name,
                        "document_id": doc_file.metadata.get("document_id") or self._make_document_id(doc_file),
                        "page_number": page_no,
                        "chunk_index": global_idx,
                        "chunk_index_in_page": i,
                        "original_file_path": doc_file.local_path,
                        "original_ext": doc_file.metadata.get("original_ext"),
                        "size_bytes": doc_file.metadata.get("size_bytes"),
                        "needs_ocr": bool(doc_file.needs_ocr),
                        "ingested_at": int(time.time()),
                    }
                    out.append(Chunk(piece, meta))
                    global_idx += 1

            if not out:
                raise ChunkerError(f"No text extracted from {doc_file.local_path}")

            return out

        except Exception as e:
            raise ChunkerError(f"Chunking failed for {doc_file.blob_name}: {e}")

    def chunk_documents(self, doc_files: List[DocumentFile]) -> List[Chunk]:
        """Process a list of documents."""
        all_chunks: List[Chunk] = []
        for d in doc_files:
            try:
                all_chunks.extend(self.chunk_document(d))
            except ChunkerError as e:
                print(f"[WARN] {e}")
                continue
        return all_chunks


class HybridChunker(BaseChunker):
    """
    Intelligent hybrid chunker that:
    1. First tries native PDF text extraction
    2. Analyzes text quality per page
    3. Falls back to OCR only for pages that need it
    4. Generates page-aware chunks with proper metadata
    """
    
    def __init__(
        self,
        extractor: BaseExtractor | None = None,
        splitter: BaseSplitter | None = None,
        docling_provider: BaseOcrProvider | None = None,  # NEW
        azure_provider: BaseOcrProvider | None = None,     # Renamed from ocr_provider
        max_tokens: int = 900,
        overlap_tokens: int = 120,
        ocr_threshold: float = 0.1,
        min_text_length: int = 50,
    ):
        # Use azure_provider as the main ocr_provider for backward compatibility
        super().__init__(extractor, splitter, azure_provider, max_tokens, overlap_tokens, ocr_threshold, min_text_length)
        self.docling_provider = docling_provider
        self.azure_provider = azure_provider

    def _analyze_page_text_quality(self, text: str) -> Dict[str, Any]:
        """Analyze text quality to determine if OCR is needed."""
        if not text or not text.strip():
            return {
                "needs_ocr": True,
                "reason": "empty_text",
                "text_length": 0,
                "word_count": 0,
                "has_meaningful_content": False
            }
        
        # Basic text analysis
        text_length = len(text.strip())
        word_count = len(text.split())
        
        # Check for common OCR artifacts
        ocr_artifacts = [
            "ERROR! BOOKMARK NOT DEFINED",
            "Error! Bookmark not defined",
            "�",  # Replacement character
            "\x00",  # Null bytes
        ]
        
        has_ocr_artifacts = any(artifact in text for artifact in ocr_artifacts)
        
        # Check if text is too short (likely incomplete extraction)
        is_too_short = text_length < self.min_text_length
        
        # Check for meaningful content (has letters, not just numbers/symbols)
        alpha_chars = sum(1 for c in text if c.isalpha())
        has_meaningful_content = alpha_chars > 20
        
        needs_ocr = has_ocr_artifacts or is_too_short or not has_meaningful_content
        
        reason = "good_text"
        if has_ocr_artifacts:
            reason = "ocr_artifacts"
        elif is_too_short:
            reason = "too_short"
        elif not has_meaningful_content:
            reason = "no_meaningful_content"
        
        return {
            "needs_ocr": needs_ocr,
            "reason": reason,
            "text_length": text_length,
            "word_count": word_count,
            "has_meaningful_content": has_meaningful_content,
            "has_ocr_artifacts": has_ocr_artifacts
        }

    def chunk_document(self, doc_file: DocumentFile) -> List[Chunk]:
        """Process a single document with hybrid OCR approach."""
        try:
            print(f"[HYBRID] Processing: {doc_file.blob_name}")
            
            # 1️⃣ Extract text by pages using native extraction
            pages = self.extractor.extract(doc_file.local_path)
            print(f"[HYBRID] Native extraction: {len(pages)} pages")
            
            # 2️⃣ Intelligent OCR selection (3-tier approach)
            processed_pages = []
            ocr_pages_needed = []
            
            for page_text, page_no in pages:
                analysis = self._analyze_page_text_quality(page_text)
                
                if analysis["needs_ocr"]:
                    print(f"[HYBRID] Page {page_no} needs OCR: {analysis['reason']}")
                    ocr_pages_needed.append(page_no)
                    # Keep original for now, will replace with OCR later
                    processed_pages.append((page_text, page_no, analysis))
                else:
                    print(f"[HYBRID] Page {page_no} OK: {analysis['reason']} ({analysis['text_length']} chars)")
                    processed_pages.append((page_text, page_no, analysis))
            
            # 3️⃣ Run intelligent OCR selection
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
            
            # 4️⃣ Generate chunks with metadata
            out: List[Chunk] = []
            global_idx = 0
            
            for page_text, page_no, analysis in processed_pages:
                clean = normalize_text(page_text)
                if not clean:
                    continue
                    
                pieces = self.splitter.split(clean)
                for i, piece in enumerate(pieces):
                    meta = {
                        "document_blob": doc_file.blob_name,
                        "document_id": doc_file.metadata.get("document_id") or self._make_document_id(doc_file),
                        "page_number": page_no,
                        "chunk_index": global_idx,
                        "chunk_index_in_page": i,
                        "original_file_path": doc_file.local_path,
                        "original_ext": doc_file.metadata.get("original_ext"),
                        "size_bytes": doc_file.metadata.get("size_bytes"),
                        "needs_ocr": analysis["needs_ocr"],
                        "ocr_reason": analysis["reason"],
                        "text_quality": {
                            "text_length": analysis["text_length"],
                            "word_count": analysis["word_count"],
                            "has_meaningful_content": analysis["has_meaningful_content"]
                        },
                        "ingested_at": int(time.time()),
                    }
                    out.append(Chunk(piece, meta))
                    global_idx += 1

            if not out:
                raise ChunkerError(f"No text extracted from {doc_file.local_path}")

            print(f"[HYBRID] Generated {len(out)} chunks from {len(processed_pages)} pages")
            return out

        except Exception as e:
            raise ChunkerError(f"Hybrid chunking failed for {doc_file.blob_name}: {e}")
