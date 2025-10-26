"""
OCR Providers for Document Processing
====================================

This module provides various OCR providers for text extraction
from scanned documents and images.
"""

import os
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

class BaseOcrProvider(ABC):
    """Base class for OCR providers."""
    
    @abstractmethod
    def extract_text(self, local_path: str) -> str:
        """Extract all text from document."""
        pass
    
    @abstractmethod
    def extract_text_per_page(self, local_path: str) -> List[Tuple[str, int]]:
        """Extract text per page."""
        pass

class AzureDocumentIntelligenceOcrProvider(BaseOcrProvider):
    """
    Azure Document Intelligence OCR provider.
    
    Features:
    - High accuracy OCR
    - Structured document processing
    - Multi-page support
    - Advanced layout analysis
    """
    
    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None, model_id: str = "prebuilt-read"):
        self.endpoint = endpoint or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
        self.model_id = model_id
        
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure Document Intelligence endpoint and API key are required")
        
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
            
            self._client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key)
            )
            self.available = True
        except ImportError:
            self.available = False
            print("⚠️ Azure Document Intelligence SDK not available")
        except Exception as e:
            self.available = False
            print(f"⚠️ Azure Document Intelligence initialization failed: {e}")
    
    def extract_text(self, local_path: str) -> str:
        """Extract all text as single string."""
        pages = self.extract_text_per_page(local_path)
        return "\n".join(text for text, _ in pages)
    
    def extract_text_per_page(self, local_path: str) -> List[Tuple[str, int]]:
        """Extract text per page with page numbers."""
        if not self.available:
            raise RuntimeError("Azure Document Intelligence not available")
        
        try:
            with open(local_path, "rb") as fh:
                poller = self._client.begin_analyze_document(
                    model_id=self.model_id, 
                    document=fh
                )
                result = poller.result()
            
            pages_text = []
            
            # Extract text from each page
            for page in getattr(result, "pages", []):
                lines = []
                for line in getattr(page, "lines", []):
                    if hasattr(line, "content") and line.content:
                        lines.append(line.content)
                
                page_text = "\n".join(lines)
                pages_text.append((page_text, page.page_number))
            
            return pages_text if pages_text else [("", 1)]
            
        except Exception as e:
            print(f"[ERROR] Azure Document Intelligence OCR failed: {e}")
            return [("", 1)]

class DoclingOcrProvider(BaseOcrProvider):
    """
    IBM Docling OCR provider for structured document processing.
    
    Features:
    - Preserves document structure
    - Better for complex documents with tables
    - CPU-based processing
    - Semantic preservation
    """
    
    def __init__(self, use_ocr: bool = True):
        try:
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
            self.use_ocr = use_ocr
            self.available = True
        except ImportError as e:
            self.available = False
            print(f"⚠️ Docling not available: {e}")
            print("Install with: pip install docling>=1.0.0 docling-core>=1.0.0")
        except Exception as e:
            self.available = False
            print(f"⚠️ Docling initialization failed: {e}")
    
    def extract_text(self, local_path: str) -> str:
        """Extract all text as single string."""
        pages = self.extract_text_per_page(local_path)
        return "\n".join(text for text, _ in pages)
    
    def extract_text_per_page(self, local_path: str) -> List[Tuple[str, int]]:
        """Extract text per page with structure preservation."""
        if not self.available:
            raise RuntimeError("Docling not available")
        
        try:
            result = self.converter.convert(local_path)
            pages_text = []
            
            # Docling returns structured markdown
            for page_num, page in enumerate(result.pages, start=1):
                # Get markdown representation (preserves tables, headings)
                page_text = page.export_to_markdown()
                pages_text.append((page_text, page_num))
            
            return pages_text if pages_text else [("", 1)]
            
        except Exception as e:
            print(f"[ERROR] Docling OCR failed: {e}")
            return [("", 1)]

class TesseractOcrProvider(BaseOcrProvider):
    """
    Tesseract OCR provider for general-purpose text extraction.
    
    Features:
    - Open source OCR engine
    - Multiple language support
    - Custom preprocessing options
    - Confidence scoring
    """
    
    def __init__(self, language: str = 'spa+eng', config: str = '--psm 6'):
        try:
            import pytesseract
            from PIL import Image
            import fitz  # PyMuPDF for PDF handling
            import io
            import os
            
            # Auto-configure Tesseract path on Windows
            if os.name == 'nt':  # Windows
                if not pytesseract.pytesseract.tesseract_cmd or not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
                    # Try to find Tesseract in common locations
                    possible_paths = [
                        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                        r'C:\tesseract\tesseract.exe',
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            print(f"[OK] Tesseract configured: {path}")
                            break
            
            self.pytesseract = pytesseract
            self.Image = Image
            self.fitz = fitz
            self.io = io
            self.language = language
            self.config = config
            self.available = True
        except ImportError as e:
            self.available = False
            print(f"⚠️ Tesseract dependencies not available: {e}")
            print("Install with: pip install pytesseract pillow PyMuPDF")
        except Exception as e:
            self.available = False
            print(f"⚠️ Tesseract initialization failed: {e}")
    
    def extract_text(self, local_path: str) -> str:
        """Extract all text as single string."""
        pages = self.extract_text_per_page(local_path)
        return "\n".join(text for text, _ in pages)
    
    def extract_text_per_page(self, local_path: str) -> List[Tuple[str, int]]:
        """Extract text per page using Tesseract OCR."""
        if not self.available:
            raise RuntimeError("Tesseract not available")
        
        try:
            pages_text = []
            
            if local_path.lower().endswith('.pdf'):
                # Handle PDF files
                doc = self.fitz.open(local_path)
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Convert page to image
                    mat = self.fitz.Matrix(2.0, 2.0)  # Scale for better OCR
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # OCR on image
                    image = self.Image.open(self.io.BytesIO(img_data))
                    text = self.pytesseract.image_to_string(
                        image, 
                        lang=self.language, 
                        config=self.config
                    )
                    
                    cleaned_text = self._clean_text(text)
                    if cleaned_text.strip():
                        pages_text.append((cleaned_text, page_num + 1))
                
                doc.close()
            else:
                # Handle image files
                image = self.Image.open(local_path)
                text = self.pytesseract.image_to_string(
                    image, 
                    lang=self.language, 
                    config=self.config
                )
                
                cleaned_text = self._clean_text(text)
                if cleaned_text.strip():
                    pages_text.append((cleaned_text, 1))
            
            return pages_text if pages_text else [("", 1)]
            
        except Exception as e:
            print(f"[ERROR] Tesseract OCR failed: {e}")
            return [("", 1)]
    
    def _clean_text(self, text: str) -> str:
        """Clean OCR text output."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:  # Skip very short lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

class HybridOcrProvider(BaseOcrProvider):
    """
    Hybrid OCR provider that combines multiple OCR engines.
    
    Features:
    - Intelligent provider selection
    - Automatic fallback mechanisms
    - Quality assessment
    - Performance optimization
    """
    
    def __init__(self):
        self.providers = []
        
        # Initialize available providers
        try:
            azure_provider = AzureDocumentIntelligenceOcrProvider()
            if azure_provider.available:
                self.providers.append(('azure', azure_provider))
        except Exception:
            pass
        
        try:
            docling_provider = DoclingOcrProvider()
            if docling_provider.available:
                self.providers.append(('docling', docling_provider))
        except Exception:
            pass
        
        try:
            tesseract_provider = TesseractOcrProvider()
            if tesseract_provider.available:
                self.providers.append(('tesseract', tesseract_provider))
        except Exception:
            pass
        
        if not self.providers:
            raise RuntimeError("No OCR providers available")
    
    def extract_text(self, local_path: str) -> str:
        """Extract text using the best available provider."""
        pages = self.extract_text_per_page(local_path)
        return "\n".join(text for text, _ in pages)
    
    def extract_text_per_page(self, local_path: str) -> List[Tuple[str, int]]:
        """Extract text per page using the best available provider."""
        # Try providers in order of preference
        for provider_name, provider in self.providers:
            try:
                print(f"[HYBRID_OCR] Trying {provider_name} OCR...")
                result = provider.extract_text_per_page(local_path)
                
                # Validate result quality
                if self._validate_ocr_quality(result):
                    print(f"[HYBRID_OCR] [OK] {provider_name} OCR successful")
                    return result
                else:
                    print(f"[HYBRID_OCR] ⚠️ {provider_name} OCR quality low, trying next provider...")
                    
            except Exception as e:
                print(f"[HYBRID_OCR] [ERROR] {provider_name} OCR failed: {e}")
                continue
        
        # If all providers failed, return empty result
        print("[HYBRID_OCR] [ERROR] All OCR providers failed")
        return [("", 1)]
    
    def _validate_ocr_quality(self, result: List[Tuple[str, int]]) -> bool:
        """Validate OCR result quality."""
        if not result:
            return False
        
        total_chars = sum(len(text) for text, _ in result)
        
        # Check minimum character threshold
        if total_chars < 50:
            return False
        
        # Check for reasonable text distribution
        non_empty_pages = sum(1 for text, _ in result if text.strip())
        if non_empty_pages == 0:
            return False
        
        # Check for excessive special characters (indicates poor OCR)
        special_char_ratio = sum(text.count('|') + text.count('_') for text, _ in result) / max(total_chars, 1)
        if special_char_ratio > 0.1:
            return False
        
        return True
