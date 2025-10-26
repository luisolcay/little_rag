"""
Text Extraction Utilities for Document Processing
=================================================

This module provides various text extraction methods for different
document types with intelligent fallback mechanisms.
"""

import os
import io
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

class ExcelExtractor(BaseExtractor):
    """
    Excel file extractor with row-level chunking strategy.
    
    Features:
    - Row-by-row extraction (one row = one chunk)
    - Column headers included in each row for context
    - Sheet-by-sheet processing
    - Optimal for precise search queries
    """
    
    def __init__(self):
        try:
            import pandas as pd
            self.pd = pd
            self.available = True
        except ImportError:
            self.available = False
            print("⚠️ pandas not available. Install with: pip install pandas")
    
    def extract(self, file_path: str) -> List[Tuple[str, int]]:
        """
        Extract Excel with row-level chunking.
        
        Each row becomes a separate text chunk that will be embedded.
        
        Args:
            file_path: Path to Excel file (.xlsx or .xls)
            
        Returns:
            List of tuples (text, row_id)
            Each tuple represents one row as a structured text
        """
        if not self.available:
            raise ImportError("pandas not available")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Load Excel file
            excel_file = self.pd.ExcelFile(file_path, engine='openpyxl')
            sheet_names = excel_file.sheet_names
            
            all_rows = []
            
            for sheet_num, sheet_name in enumerate(sheet_names, 1):
                # Read sheet as DataFrame
                df = self.pd.read_excel(excel_file, sheet_name=sheet_name)
                
                if df.empty:
                    continue
                
                # Get column headers
                headers = df.columns.tolist()
                header_row = " | ".join(str(h) for h in headers)
                
                # Process each row
                for idx, row in df.iterrows():
                    # Convert row values to strings
                    row_values = [str(val) if self.pd.notna(val) else "" for val in row.values]
                    value_row = " | ".join(row_values)
                    
                    # Create row chunk with context
                    row_text = f"Sheet: {sheet_name}\n{header_row}\n{value_row}"
                    
                    # Use a unique identifier for each row across all sheets
                    row_id = (sheet_num - 1) * 10000 + idx
                    all_rows.append((row_text, row_id))
            
            excel_file.close()
            
            if not all_rows:
                return [("", 1)]
            
            return all_rows
            
        except Exception as e:
            print(f"❌ Excel extraction failed: {e}")
            return [("", 1)]

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
        
        try:
            excel_extractor = ExcelExtractor()
            if excel_extractor.available:
                self.extractors.append(('excel', excel_extractor))
        except Exception:
            pass
        
        if not self.extractors:
            raise ImportError("No text extractors available")
    
    def extract(self, file_path: str) -> List[Tuple[str, int]]:
        """
        Extract text using the best available method based on file type.
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of tuples (text, page_number or row_id)
        """
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Route to appropriate extractor
        if file_ext in ['.xlsx', '.xls']:
            # Use Excel extractor
            for method_name, extractor in self.extractors:
                if method_name == 'excel':
                    try:
                        print(f"[HYBRID_EXTRACTOR] Using Excel extraction...")
                        result = extractor.extract(file_path)
                        if result and len(result) > 0:
                            print(f"[HYBRID_EXTRACTOR] ✅ Excel extraction successful: {len(result)} rows")
                            return result
                    except Exception as e:
                        print(f"[HYBRID_EXTRACTOR] ❌ Excel extraction failed: {e}")
            # If no excel extractor found
            print("[HYBRID_EXTRACTOR] ❌ No Excel extractor available")
            return [("", 1)]
        
        elif file_ext == '.pdf':
            # Try PDF extractors in order of preference
            for method_name, extractor in self.extractors:
                if method_name in ['pymupdf', 'tesseract']:
                    try:
                        print(f"[HYBRID_EXTRACTOR] Trying {method_name} extraction...")
                        result = extractor.extract(file_path)
                        
                        if self._validate_extraction_quality(result):
                            print(f"[HYBRID_EXTRACTOR] [OK] {method_name} extraction successful")
                            return result
                        else:
                            print(f"[HYBRID_EXTRACTOR] [WARNING] {method_name} extraction quality low, trying next...")
                    except Exception as e:
                        print(f"[HYBRID_EXTRACTOR] [ERROR] {method_name} extraction failed: {e}")
                        continue
        
        # If all methods failed
        print("[HYBRID_EXTRACTOR] [ERROR] All extraction methods failed")
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
