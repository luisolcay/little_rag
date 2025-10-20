"""
Complete document processing pipeline with Blob Storage, OCR, and chunking.
"""

import os
import tempfile
import logging
from typing import Dict, Any, List
from datetime import datetime

from .file_loader import FileLoader, DocumentFile
from .blob_utils import upload_blob_from_bytes
from .processing.text_processing.ocr import AzureDocumentIntelligenceOcrProvider
from .processing.chunkers.enhanced import EnhancedHybridChunker

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Complete document processing pipeline."""
    
    def __init__(self, container_name: str = "orbe"):
        self.container_name = container_name
        self.file_loader = FileLoader(container_name)
        self.ocr_provider = AzureDocumentIntelligenceOcrProvider()
        self.chunker = EnhancedHybridChunker()
        self.working_dir = tempfile.mkdtemp()
        
    async def process_uploaded_file(
        self,
        file_content: bytes,
        filename: str,
        file_id: str
    ) -> Dict[str, Any]:
        """Process uploaded file through complete pipeline."""
        
        try:
            # 1. Upload to Blob Storage
            blob_name = f"documents/{file_id}/{filename}"
            blob_url = upload_blob_from_bytes(
                self.container_name,
                blob_name,
                file_content
            )
            
            logger.info(f"Uploaded to Blob Storage: {blob_url}")
            
            # 2. Download and create DocumentFile
            local_path = os.path.join(self.working_dir, filename)
            self.file_loader.download_blob_to_file(
                self.container_name,
                blob_name,
                local_path,
                overwrite=True
            )
            
            # Create DocumentFile with metadata
            doc_file = DocumentFile(
                local_path=local_path,
                blob_name=blob_name,
                metadata={
                    "filename": filename,
                    "document_id": file_id,
                    "upload_timestamp": datetime.now().isoformat(),
                    "blob_url": blob_url
                },
                needs_ocr=self._needs_ocr(filename)
            )
            
            # 3. Apply OCR if needed
            if doc_file.needs_ocr:
                logger.info(f"Applying OCR to {filename}")
                content = self.ocr_provider.extract_text(doc_file.local_path)
            else:
                # Read content normally
                with open(doc_file.local_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # 4. Chunk the document
            logger.info(f"Chunking document {filename}")
            chunks = self.chunker.chunk_document(doc_file)
            
            # 5. Return processed data
            return {
                "success": True,
                "blob_url": blob_url,
                "blob_name": blob_name,
                "chunks": chunks,
                "needs_ocr": doc_file.needs_ocr,
                "metadata": doc_file.metadata,
                "content_length": len(content)
            }
            
        except Exception as e:
            logger.error(f"Document processing failed for {filename}: {e}")
            return {
                "success": False,
                "error": str(e),
                "blob_name": blob_name if 'blob_name' in locals() else None
            }
    
    def _needs_ocr(self, filename: str) -> bool:
        """Determine if file needs OCR based on extension and content."""
        # For now, assume PDFs need OCR
        # In production, you might check file content or use Document Intelligence's analyze capability
        return filename.lower().endswith('.pdf')
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup working directory: {e}")
