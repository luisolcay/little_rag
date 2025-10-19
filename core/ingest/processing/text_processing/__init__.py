"""
Text processing utilities for document ingestion.

This module contains extractors, splitters, cleaning utilities, and OCR providers.
"""

from .extractors import (
    BaseExtractor,
    AutoExtractor,
    PdfExtractor,
    DocxExtractor,
    HtmlExtractor,
    TxtExtractor,
)
from .splitter import BaseSplitter, TokenAwareSentenceSplitter
from .splitters import SemanticOverlapSplitter
from .cleaning import normalize_text
from .ocr import BaseOcrProvider, AzureDocumentIntelligenceOcrProvider, DoclingOcrProvider

__all__ = [
    "BaseExtractor", "AutoExtractor", "PdfExtractor", "DocxExtractor", "HtmlExtractor", "TxtExtractor",
    "BaseSplitter", "TokenAwareSentenceSplitter", "SemanticOverlapSplitter",
    "normalize_text",
    "BaseOcrProvider", "AzureDocumentIntelligenceOcrProvider", "DoclingOcrProvider",
]
