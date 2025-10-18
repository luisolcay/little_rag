import os
import tempfile
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from .blob_utils import (
    BlobStorageError,
    download_blob_to_file,
    list_blobs_in_container
)


class FileLoaderError(Exception):
    """Raised when an error occurs during document ingestion / loading."""
    pass

class DocumentFile:
    """
    Representation of a single document after download/extraction, ready for chunking.
    """
    def __init__(
        self,
        local_path: str,
        blob_name: str,
        metadata: Dict[str, Any],
        needs_ocr: bool = False
    ):
        self.local_path = local_path
        self.blob_name = blob_name
        self.metadata = metadata
        self.needs_ocr = needs_ocr

    def __repr__(self):
        return (f"DocumentFile(blob_name={self.blob_name}, "
                f"local_path={self.local_path}, needs_ocr={self.needs_ocr}, "
                f"metadata={self.metadata})")


class FileLoader:
    """
    FileLoader is responsible for:
     - listing new documents from Blob Storage;
     - downloading them locally for processing;
     - basic validation of file type/size;
     - detecting whether OCR is needed;
     - returning DocumentFile objects to feed into later steps (chunking, etc).
    """

    def __init__(self, container_name: str, working_dir: Optional[str] = None):
        load_dotenv()
        self.container_name = container_name
        self.working_dir = working_dir or tempfile.gettempdir()
        os.makedirs(self.working_dir, exist_ok=True)

    def list_new_documents(self, prefix: str = "") -> List[str]:
        try:
            blob_names = list_blobs_in_container(self.container_name, prefix=prefix)
            return blob_names
        except BlobStorageError as e:
            raise FileLoaderError(f"Failed to list blobs: {e}")

    def download_document(self, blob_name: str, overwrite: bool = False) -> str:
        safe_filename = blob_name.replace("/", "_")
        local_path = os.path.join(self.working_dir, safe_filename)
        try:
            download_blob_to_file(self.container_name, blob_name, local_path, overwrite=overwrite)
            return local_path
        except BlobStorageError as e:
            raise FileLoaderError(f"Failed to download blob '{blob_name}': {e}")

    def validate_document(self, local_path: str) -> None:
        _, ext = os.path.splitext(local_path)
        ext = ext.lower()
        supported = {'.pdf', '.docx', '.html'}
        if ext not in supported:
            raise FileLoaderError(f"Unsupported file extension '{ext}' for file '{local_path}'")

        max_size_bytes = 400 * 1024 * 1024  # 400 MB for MVP
        size = os.path.getsize(local_path)
        if size > max_size_bytes:
            raise FileLoaderError(f"File '{local_path}' is too large ({size} bytes)")

    def detect_need_ocr(self, local_path: str) -> bool:
        """
        Heuristic to determine if the document likely needs OCR (like scanned PDF with no text layer).
        Returns True if OCR is likely needed.
        """
        import pymupdf as fitz
        try:
            doc = fitz(local_path)
            total_area = 0.0
            text_area = 0.0
            for page in doc:
                page_rect = page.rect
                total_area += abs(page_rect)
                for b in page.get_text("blocks"):
                    r = fitz.Rect(b[:4])
                    text_area += abs(r)
            doc.close()
            ratio = text_area / total_area if total_area > 0 else 0.0
            # if text ratio < 0.1 we assume its scan
            return ratio < 0.10
        except Exception as e:
            # fail to open, use ocr
            return True

    def load(self, prefix: str = "", overwrite: bool = False) -> List[DocumentFile]:
        local_docs: List[DocumentFile] = []
        blob_names = self.list_new_documents(prefix=prefix)
        for blob_name in blob_names:
            try:
                local_path = self.download_document(blob_name, overwrite=overwrite)
                self.validate_document(local_path)
                needs_ocr = False
                _, ext = os.path.splitext(local_path)
                if ext.lower() == '.pdf':
                    needs_ocr = self.detect_need_ocr(local_path)

                metadata = {
                    "blob_name": blob_name,
                    "original_ext": ext.lower(),
                    "size_bytes": os.path.getsize(local_path),
                    "needs_ocr": needs_ocr
                }
                doc_file = DocumentFile(local_path=local_path, blob_name=blob_name, metadata=metadata, needs_ocr=needs_ocr)
                local_docs.append(doc_file)
            except FileLoaderError as e:
                print(f"Warning: skipping blob '{blob_name}' due to error: {e}")
                continue
        return local_docs

if __name__ == "__main__":

    container = os.getenv("AZURE_BLOB_CONTAINER") or "orbe"
    loader = FileLoader(container_name=container, working_dir="./downloaded_files")
    print("Listing and downloading new documents from blob storage…")
    local_docs = loader.load(prefix="", overwrite=True)
    print("Downloaded documents:", local_docs)
