import os
from typing import List, Tuple
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential


class BaseOcrProvider:
    """Base OCR interface."""
    def extract_text(self, local_path: str) -> str:
        raise NotImplementedError


class DoclingOcrProvider(BaseOcrProvider):
    """
    OCR using IBM Docling - preserves document structure and semantics.
    Works on CPU, better for complex documents with tables/structure.
    """
    def __init__(self, use_ocr: bool = True):
        try:
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
            self.use_ocr = use_ocr
        except ImportError as e:
            raise ImportError(f"Docling not installed. Run: pip install docling>=1.0.0 docling-core>=1.0.0. Error: {e}")
    
    def extract_text(self, local_path: str) -> str:
        """Legacy method - extracts all text as single string."""
        pages = self.extract_text_per_page(local_path)
        return "\n".join(text for text, _ in pages)

    def extract_text_per_page(self, local_path: str) -> List[Tuple[str, int]]:
        """Extract text per page with structure preservation"""
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
            raise Exception(f"Docling OCR failed for {local_path}: {e}")


class AzureDocumentIntelligenceOcrProvider(BaseOcrProvider):
    """
    OCR using Azure Document Intelligence (Form Recognizer).
    Requires AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and optionally AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID.
    """
    def __init__(self, *, endpoint=None, model_id=None, credential=None, api_key: str | None = None):
        load_dotenv()
        self.endpoint = endpoint or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.model_id = model_id or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID")
        api_key = api_key or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        if not self.endpoint:
            raise ValueError("Missing AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")

        if api_key:
            self._client = DocumentAnalysisClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(api_key)
            )
        else:
            self._client = DocumentAnalysisClient(
                endpoint=self.endpoint,
                credential=credential or DefaultAzureCredential()
            )

    def extract_text(self, local_path: str) -> str:
        """Legacy method - extracts all text as single string."""
        pages = self.extract_text_per_page(local_path)
        return "\n".join(text for text, _ in pages)

    def extract_text_per_page(self, local_path: str) -> List[Tuple[str, int]]:
        """Extract text per page, returns [(text, page_no), ...]"""
        with open(local_path, "rb") as fh:
            poller = self._client.begin_analyze_document(model_id=self.model_id, document=fh)
        result = poller.result()
        
        pages_text = []
        for page in getattr(result, "pages", []):
            lines = []
            for line in getattr(page, "lines", []):
                if line.content:
                    lines.append(line.content)
            pages_text.append(("\n".join(lines), page.page_number))
        
        return pages_text if pages_text else [("", 1)]