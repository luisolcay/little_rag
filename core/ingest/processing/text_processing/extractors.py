from typing import List, Tuple
import os


try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
except Exception:
    PyPDFLoader = None
    Docx2txtLoader = None
    UnstructuredHTMLLoader = None


class BaseExtractor:
    """Base interface for all file extractors."""
    def extract(self, local_path: str) -> List[Tuple[str, int]]:
        raise NotImplementedError


class PdfExtractor(BaseExtractor):
    """Extracts text per page (PyMuPDF → PyPDFLoader → pdfminer fallback)."""
    def extract(self, local_path: str) -> List[Tuple[str, int]]:
        # 1) PyMuPDF
        if fitz is not None:
            try:
                doc = fitz.open(local_path)
                pages = [(page.get_text("text") or "", i + 1) for i, page in enumerate(doc)]
                doc.close()
                return pages
            except Exception:
                pass

        # 2) LangChain PyPDFLoader
        if PyPDFLoader is not None:
            try:
                loader = PyPDFLoader(local_path)
                docs = loader.load()
                out = []
                for d in docs:
                    page0 = int(d.metadata.get("page", 0))
                    out.append((d.page_content or "", page0 + 1))
                return out
            except Exception:
                pass

        # 3) pdfminer fallback
        try:
            import pdfminer.high_level as pdfminer_high
            text = pdfminer_high.extract_text(local_path) or ""
            return [(text, 1)]
        except Exception:
            with open(local_path, "rb") as f:
                blob = f.read().decode("utf-8", errors="ignore")
            return [(blob, 1)]


class DocxExtractor(BaseExtractor):
    def extract(self, local_path: str) -> List[Tuple[str, int]]:
        if Docx2txtLoader is None:
            raise ImportError("Install langchain_community for Docx2txtLoader.")
        docs = Docx2txtLoader(local_path).load()
        full = "\n".join(d.page_content for d in docs)
        return [(full, 1)]


class HtmlExtractor(BaseExtractor):
    def extract(self, local_path: str) -> List[Tuple[str, int]]:
        if UnstructuredHTMLLoader is None:
            raise ImportError("Install unstructured for HTML extraction.")
        docs = UnstructuredHTMLLoader(local_path).load()
        full = "\n".join(d.page_content for d in docs)
        return [(full, 1)]


class TxtExtractor(BaseExtractor):
    def extract(self, local_path: str) -> List[Tuple[str, int]]:
        with open(local_path, encoding="utf-8", errors="ignore") as f:
            return [(f.read(), 1)]


class AutoExtractor(BaseExtractor):
    """Chooses extractor by file extension."""
    def __init__(self):
        self.pdf = PdfExtractor()
        self.docx = DocxExtractor()
        self.html = HtmlExtractor()
        self.txt = TxtExtractor()

    def extract(self, local_path: str) -> List[Tuple[str, int]]:
        ext = os.path.splitext(local_path)[1].lower()
        if ext == ".pdf":
            return self.pdf.extract(local_path)
        if ext == ".docx":
            return self.docx.extract(local_path)
        if ext in (".html", ".htm"):
            return self.html.extract(local_path)
        return self.txt.extract(local_path)