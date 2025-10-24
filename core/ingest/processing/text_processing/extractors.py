"""
Text Extraction Utilities for Document Processing
=================================================

This module provides various text extraction methods for different
document types with intelligent fallback mechanisms.
"""

import os
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    """Base class for text extractors."""
    
    @abstractmethod
    def extract(self, file_path: str) -> List[Tuple[str, int]]:
        """
        Extract text from document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of tuples (text, page_number)
        """
        pass

class PyMuPDFExtractor(BaseExtractor):
    """
    High-performance text extractor using PyMuPDF (fitz).
    
    Features:
    - Fast PDF text extraction
    - Page-by-page processing
    - Metadata extraction
    - Error handling and fallback
    """
    
    def __init__(self):
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
            self.available = True
        except ImportError:
            self.available = False
            print("⚠️ PyMuPDF not available. Install with: pip install PyMuPDF")
    
    def extract(self, file_path: str) -> List[Tuple[str, int]]:
        """
        Extract text from PDF using PyMuPDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of tuples (text, page_number)
        """
        if not self.available:
            raise ImportError("PyMuPDF not available")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Open document
            doc = self.fitz.open(file_path)
            pages_text = []
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Clean and validate text
                cleaned_text = self._clean_text(text)
                
                if cleaned_text.strip():  # Only add non-empty pages
                    pages_text.append((cleaned_text, page_num + 1))
            
            doc.close()
            
            if not pages_text:
                # Fallback: try to extract any text
                pages_text = self._fallback_extraction(file_path)
            
            return pages_text
            
        except Exception as e:
            print(f"❌ PyMuPDF extraction failed: {e}")
            # Try fallback extraction
            return self._fallback_extraction(file_path)
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _fallback_extraction(self, file_path: str) -> List[Tuple[str, int]]:
        """Fallback extraction method."""
        try:
            # Try with pdfplumber as fallback
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                pages_text = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        cleaned_text = self._clean_text(text)
                        if cleaned_text.strip():
                            pages_text.append((cleaned_text, page_num))
                
                return pages_text
                
        except ImportError:
            print("⚠️ pdfplumber not available for fallback")
        except Exception as e:
            print(f"❌ Fallback extraction failed: {e}")
        
        # Last resort: return empty with single page
        return [("", 1)]

class TesseractExtractor(BaseExtractor):
    """
    OCR-based text extractor using Tesseract.
    
    Features:
    - OCR for scanned documents
    - Multiple language support
    - Image preprocessing
    - Confidence scoring
    """
    
    def __init__(self, language: str = 'spa+eng'):
        try:
            import pytesseract
            from PIL import Image
            self.pytesseract = pytesseract
            self.Image = Image
            self.language = language
            self.available = True
        except ImportError:
            self.available = False
            print("⚠️ Tesseract not available. Install with: pip install pytesseract pillow")
    
    def extract(self, file_path: str) -> List[Tuple[str, int]]:
        """
        Extract text using OCR.
        
        Args:
            file_path: Path to image or PDF file
            
        Returns:
            List of tuples (text, page_number)
        """
        if not self.available:
            raise ImportError("Tesseract not available")
        
        try:
            # Convert PDF to images if needed
            if file_path.lower().endswith('.pdf'):
                return self._extract_from_pdf(file_path)
            else:
                # Direct image processing
                return self._extract_from_image(file_path)
                
        except Exception as e:
            print(f"❌ Tesseract extraction failed: {e}")
            return [("", 1)]
    
    def _extract_from_pdf(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF using OCR."""
        try:
            import fitz  # PyMuPDF for PDF to image conversion
            
            doc = fitz.open(file_path)
            pages_text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # Scale factor for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # OCR on image
                image = self.Image.open(io.BytesIO(img_data))
                text = self.pytesseract.image_to_string(image, lang=self.language)
                
                cleaned_text = self._clean_text(text)
                if cleaned_text.strip():
                    pages_text.append((cleaned_text, page_num + 1))
            
            doc.close()
            return pages_text
            
        except Exception as e:
            print(f"❌ PDF OCR extraction failed: {e}")
            return [("", 1)]
    
    def _extract_from_image(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from image file."""
        try:
            image = self.Image.open(file_path)
            text = self.pytesseract.image_to_string(image, lang=self.language)
            
            cleaned_text = self._clean_text(text)
            return [(cleaned_text, 1)] if cleaned_text.strip() else [("", 1)]
            
        except Exception as e:
            print(f"❌ Image OCR extraction failed: {e}")
            return [("", 1)]
    
    def _clean_text(self, text: str) -> str:
        """Clean OCR text."""
        if not text:
            return ""
        
        # Remove excessive whitespace and clean up OCR artifacts
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:  # Skip very short lines (likely OCR noise)
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

class HybridExtractor(BaseExtractor):
    """
    Hybrid extractor that combines multiple extraction methods.
    
    Features:
    - Intelligent method selection
    - Automatic fallback mechanisms
    - Quality assessment
    - Performance optimization
    """
    
    def __init__(self):
        self.extractors = []
        
        # Initialize available extractors
        try:
            pymupdf_extractor = PyMuPDFExtractor()
            if pymupdf_extractor.available:
                self.extractors.append(('pymupdf', pymupdf_extractor))
        except Exception:
            pass
        
        try:
            tesseract_extractor = TesseractExtractor()
            if tesseract_extractor.available:
                self.extractors.append(('tesseract', tesseract_extractor))
        except Exception:
            pass
        
        if not self.extractors:
            raise ImportError("No text extractors available")
    
    def extract(self, file_path: str) -> List[Tuple[str, int]]:
        """
        Extract text using the best available method.
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of tuples (text, page_number)
        """
        # Try extractors in order of preference
        for method_name, extractor in self.extractors:
            try:
                print(f"[HYBRID_EXTRACTOR] Trying {method_name} extraction...")
                result = extractor.extract(file_path)
                
                # Validate result quality
                if self._validate_extraction_quality(result):
                    print(f"[HYBRID_EXTRACTOR] ✅ {method_name} extraction successful")
                    return result
                else:
                    print(f"[HYBRID_EXTRACTOR] ⚠️ {method_name} extraction quality low, trying next method...")
                    
            except Exception as e:
                print(f"[HYBRID_EXTRACTOR] ❌ {method_name} extraction failed: {e}")
                continue
        
        # If all methods failed, return empty result
        print("[HYBRID_EXTRACTOR] ❌ All extraction methods failed")
        return [("", 1)]
    
    def _validate_extraction_quality(self, result: List[Tuple[str, int]]) -> bool:
        """Validate the quality of extraction result."""
        if not result:
            return False
        
        total_chars = sum(len(text) for text, _ in result)
        
        # Check minimum character threshold
        if total_chars < 100:
            return False
        
        # Check for reasonable text distribution
        non_empty_pages = sum(1 for text, _ in result if text.strip())
        if non_empty_pages == 0:
            return False
        
        # Check for excessive whitespace (indicates poor extraction)
        whitespace_ratio = sum(text.count(' ') for text, _ in result) / max(total_chars, 1)
        if whitespace_ratio > 0.5:
            return False
        
        return True
