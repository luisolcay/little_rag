"""
HeaderFilter - Intelligent Header Filter
========================================

This class provides intelligent filtering of repetitive headers
in PDF documents, with support for different document types.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod

class BaseHeaderFilter(ABC):
    """Base class for header filters."""
    
    @abstractmethod
    def clean_content(self, content: str) -> str:
        """Cleans content by removing headers."""
        pass
    
    @abstractmethod
    def is_header_line(self, line: str) -> bool:
        """Determines if a line is part of the header."""
        pass

class HeaderFilter(BaseHeaderFilter):
    """
    General header filter with common patterns.
    """
    
    def __init__(self, 
                 header_threshold: float = 0.15,
                 min_header_length: int = 10,
                 max_header_length: int = 200):
        """
        Initializes the header filter.
        
        Args:
            header_threshold: Percentage of lines from the beginning to consider as header
            min_header_length: Minimum line length to consider as header
            max_header_length: Maximum line length to consider as header
        """
        self.header_threshold = header_threshold
        self.min_header_length = min_header_length
        self.max_header_length = max_header_length
        
        # Common header patterns
        self.header_patterns = [
            r'^[A-Z\s]+$',  # Only uppercase letters and spaces
            r'Página:\s*\d+\s*de\s*\d+',  # Page numbering
            r'Gerencia\s+Corporativa',  # Specific corporate patterns
            r'CORPORACIÓN\s+NACIONAL',  # Specific CODELCO patterns
            r'Contrato\s+Marco',  # Contract patterns
            r'REQUERIMIENTO\s+DE\s+SERVICIO',  # Service request patterns
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+.*\d{4}$',  # Titles with year
        ]
        
        # Document type specific patterns
        self.document_type_patterns = {
            'corporate': [
                r'CORPORACIÓN\s+NACIONAL\s+DEL\s+COBRE',
                r'Gerencia\s+Corporativa',
                r'CODELCO\s*-\s*CHILE',
                r'Contrato\s+Marco',
                r'REQUERIMIENTO\s+DE\s+SERVICIO',
            ],
            'academic': [
                r'Universidad\s+.*',
                r'Facultad\s+.*',
                r'Departamento\s+.*',
                r'Trabajo\s+de\s+.*',
                r'Tesis\s+.*',
                r'Magíster\s+.*',
                r'Doctorado\s+.*',
            ],
            'legal': [
                r'LEY\s+N°\s*\d+',
                r'Artículo\s+\d+',
                r'Decreto\s+.*',
                r'Reglamento\s+.*',
                r'Circular\s+.*',
            ],
            'technical': [
                r'Manual\s+.*',
                r'Especificación\s+.*',
                r'Procedimiento\s+.*',
                r'Protocolo\s+.*',
            ]
        }
    
    def clean_content(self, content: str) -> str:
        """
        Cleans content by removing detected headers.
        
        Args:
            content: Original content
            
        Returns:
            Clean content without headers
        """
        # FIRST: Apply document type specific cleaning
        cleaned_content = self._apply_document_specific_cleaning(content)
        
        # SECOND: Detect and remove remaining header lines
        lines = cleaned_content.split('\n')
        cleaned_lines = []
        
        # Detect header lines
        header_lines = self._detect_header_lines(lines)
        
        # Filter lines that are not headers
        for i, line in enumerate(lines):
            if i not in header_lines:
                cleaned_lines.append(line)
        
        # Join lines and clean extra spaces
        cleaned_content = '\n'.join(cleaned_lines)
        cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_content)  # Clean multiple line breaks
        cleaned_content = cleaned_content.strip()
        
        return cleaned_content
    
    def _apply_document_specific_cleaning(self, content: str) -> str:
        """Applies specific cleaning according to document type."""
        # Detect document type
        detector = DocumentTypeDetector()
        doc_type = detector.detect_document_type(content)
        
        if doc_type == 'corporate':
            # Use more aggressive corporate filter
            corporate_filter = CorporateHeaderFilter()
            return corporate_filter.clean_codelco_document(content)
        elif doc_type == 'academic':
            academic_filter = AcademicHeaderFilter()
            return academic_filter.clean_academic_document(content)
        elif doc_type == 'legal':
            legal_filter = LegalHeaderFilter()
            return legal_filter.clean_legal_document(content)
        
        return content
    
    def _detect_header_lines(self, lines: List[str]) -> List[int]:
        """
        Detects which lines are part of the header.
        
        Args:
            lines: List of content lines
            
        Returns:
            List of indices of lines that are headers
        """
        header_lines = []
        
        # Method 1: By position (first N% of lines)
        header_threshold_lines = int(len(lines) * self.header_threshold)
        
        for i in range(min(header_threshold_lines, len(lines))):
            line = lines[i].strip()
            if self.is_header_line(line):
                header_lines.append(i)
        
        # Method 2: By specific patterns throughout the document
        for i, line in enumerate(lines):
            line = line.strip()
            if self._matches_header_pattern(line):
                header_lines.append(i)
        
        # Method 3: Detect consecutive header blocks
        consecutive_headers = self._detect_consecutive_header_blocks(lines)
        header_lines.extend(consecutive_headers)
        
        return sorted(list(set(header_lines)))  # Remove duplicates and sort
    
    def is_header_line(self, line: str) -> bool:
        """
        Determines if a line is part of the header.
        
        Args:
            line: Line to evaluate
            
        Returns:
            True if it's a header line, False otherwise
        """
        line = line.strip()
        
        # Check length
        if len(line) < self.min_header_length or len(line) > self.max_header_length:
            return False
        
        # Check common patterns
        if self._matches_header_pattern(line):
            return True
        
        # Check header characteristics
        if self._has_header_characteristics(line):
            return True
        
        return False
    
    def _matches_header_pattern(self, line: str) -> bool:
        """Checks if the line matches any header pattern."""
        for pattern in self.header_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _has_header_characteristics(self, line: str) -> bool:
        """Checks if the line has typical header characteristics."""
        # Only uppercase
        if line.isupper() and len(line) > 10:
            return True
        
        # Many spaces (header format)
        if line.count(' ') > len(line) * 0.3:
            return True
        
        # Numbering patterns
        if re.search(r'\d+\s*de\s*\d+', line):
            return True
        
        # Date patterns
        if re.search(r'\d{4}', line) and len(line) < 50:
            return True
        
        return False
    
    def _detect_consecutive_header_blocks(self, lines: List[str]) -> List[int]:
        """Detects consecutive blocks of lines that look like headers."""
        header_blocks = []
        current_block = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if self.is_header_line(line):
                current_block.append(i)
            else:
                # If we have a block of at least 2 consecutive lines, consider it header
                if len(current_block) >= 2:
                    header_blocks.extend(current_block)
                current_block = []
        
        # Check final block
        if len(current_block) >= 2:
            header_blocks.extend(current_block)
        
        return header_blocks

class CorporateHeaderFilter(HeaderFilter):
    """Specific filter for corporate documents."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add specific corporate patterns
        self.header_patterns.extend(self.document_type_patterns['corporate'])
    
    def clean_codelco_document(self, content: str) -> str:
        """Specific cleaning for CODELCO documents."""
        
        # Specific CODELCO patterns - MORE AGGRESSIVE
        codelco_patterns = [
            r'CORPORACIÓN NACIONAL DEL COBRE DE CHILE.*?\n',
            r'Gerencia Corporativa.*?\n',
            r'Página:\s*\d+\s*de\s*\d+.*?\n',
            r'Contrato Marco.*?\n',
            r'REQUERIMIENTO DE SERVICIO.*?\n',
            r'CODELCO\s*-\s*CHILE.*?\n',
            # NEW MORE SPECIFIC PATTERNS
            r'Consultoría para el Diseño Conceptual Sistema GIS Corporativo para la.*?\n',
            r'Gestión Ambiental y del Recurso Hídrico.*?\n',
            r'GMA\s*-\s*CM.*?\n',
            r'REQUERIMIENTO DE SERVICIO N°\d+.*?\n',
        ]
        
        cleaned = content
        for pattern in codelco_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.DOTALL)
        
        # Clean extra spaces and multiple empty lines
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned

class AcademicHeaderFilter(HeaderFilter):
    """Specific filter for academic documents."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.header_patterns.extend(self.document_type_patterns['academic'])
    
    def clean_academic_document(self, content: str) -> str:
        """Specific cleaning for academic documents."""
        
        academic_patterns = [
            r'Universidad\s+.*?\n',
            r'Facultad\s+.*?\n',
            r'Departamento\s+.*?\n',
            r'Trabajo\s+de\s+.*?\n',
            r'Tesis\s+.*?\n',
            r'Magíster\s+.*?\n',
            r'Doctorado\s+.*?\n',
        ]
        
        cleaned = content
        for pattern in academic_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.DOTALL)
        
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned

class LegalHeaderFilter(HeaderFilter):
    """Specific filter for legal documents."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.header_patterns.extend(self.document_type_patterns['legal'])
    
    def clean_legal_document(self, content: str) -> str:
        """Specific cleaning for legal documents."""
        
        legal_patterns = [
            r'LEY\s+N°\s*\d+.*?\n',
            r'Artículo\s+\d+.*?\n',
            r'Decreto\s+.*?\n',
            r'Reglamento\s+.*?\n',
            r'Circular\s+.*?\n',
        ]
        
        cleaned = content
        for pattern in legal_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.DOTALL)
        
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned

class DocumentTypeDetector:
    """Detects document type to apply appropriate filter."""
    
    def __init__(self):
        self.type_indicators = {
            'corporate': [
                'corporación', 'gerencia', 'codelco', 'contrato marco',
                'requerimiento de servicio', 'empresa', 'corporativo'
            ],
            'academic': [
                'universidad', 'facultad', 'departamento', 'tesis',
                'magíster', 'doctorado', 'trabajo de', 'investigación'
            ],
            'legal': [
                'ley', 'artículo', 'decreto', 'reglamento',
                'circular', 'normativa', 'jurídico'
            ],
            'technical': [
                'manual', 'especificación', 'procedimiento',
                'protocolo', 'técnico', 'ingeniería'
            ]
        }
    
    def detect_document_type(self, content: str) -> str:
        """
        Detects document type based on content.
        
        Args:
            content: Document content
            
        Returns:
            Detected document type
        """
        content_lower = content.lower()
        
        type_scores = {}
        for doc_type, indicators in self.type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            type_scores[doc_type] = score
        
        # Return type with highest score
        if type_scores:
            return max(type_scores, key=type_scores.get)
        
        return 'general'

def get_header_filter_for_document(content: str) -> HeaderFilter:
    """
    Gets the appropriate header filter for the document type.
    
    Args:
        content: Document content
        
    Returns:
        Appropriate filter instance
    """
    detector = DocumentTypeDetector()
    doc_type = detector.detect_document_type(content)
    
    filters = {
        'corporate': CorporateHeaderFilter,
        'academic': AcademicHeaderFilter,
        'legal': LegalHeaderFilter,
        'technical': HeaderFilter,
        'general': HeaderFilter
    }
    
    return filters.get(doc_type, HeaderFilter)()
