"""
Enhanced hybrid chunker with quality validation, semantic overlap, and reference preservation.

This module contains the production-ready enhanced chunker that combines
all advanced features for optimal document processing.
"""

import os
import time
from typing import List, Tuple, Dict, Any

from .base import HybridChunker
from ..models import Chunk
from ..analysis.quality_validator import ChunkQualityValidator
from ..text_processing.splitters import SemanticOverlapSplitter
from ..analysis.reference_preserver import ReferencePreserver
from ...file_loader import DocumentFile


class EnhancedHybridChunker(HybridChunker):
    """Enhanced version with quality validation, semantic overlap, and reference preservation"""
    
    def __init__(self,
                 extractor=None,
                 splitter=None,
                 docling_provider=None,  # NEW
                 azure_provider=None,     # Renamed from ocr_provider
                 max_tokens: int = 900,
                 overlap_tokens: int = 120,
                 ocr_threshold: float = 0.1,
                 min_text_length: int = 50,
                 # Quality validation settings
                 min_chunk_length: int = 50,
                 max_chunk_length: int = 2000,
                 min_sentence_count: int = 1,
                 max_repetition_ratio: float = 0.3,
                 max_special_char_ratio: float = 0.3,
                 min_quality_threshold: float = 0.5,
                 # Semantic overlap settings
                 overlap_sentences: int = 2,
                 preserve_paragraphs: bool = True,
                 # Reference preservation settings
                 enable_reference_preservation: bool = True,
                 # Logging settings
                 verbose: bool = True):
        
        # Initialize base chunker
        super().__init__(
            extractor=extractor,
            splitter=splitter,
            docling_provider=docling_provider,
            azure_provider=azure_provider,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            ocr_threshold=ocr_threshold,
            min_text_length=min_text_length
        )
        
        # Replace splitter with semantic version
        self.splitter = SemanticOverlapSplitter(
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
            preserve_paragraphs=preserve_paragraphs
        )
        
        # Initialize quality validator
        self.quality_validator = ChunkQualityValidator(
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
            min_sentence_count=min_sentence_count,
            max_repetition_ratio=max_repetition_ratio,
            max_special_char_ratio=max_special_char_ratio
        )
        
        # Initialize reference preserver
        self.reference_preserver = ReferencePreserver()
        
        # Configuration
        self.min_quality_threshold = min_quality_threshold
        self.enable_reference_preservation = enable_reference_preservation
        self.verbose = verbose
        
        # Statistics tracking
        self.processing_stats = {
            'total_documents': 0,
            'total_chunks_generated': 0,
            'total_chunks_filtered': 0,
            'average_quality_score': 0.0,
            'reference_coverage': 0.0
        }
    
    def chunk_document(self, doc_file: DocumentFile) -> List[Chunk]:
        """Process document with all enhancements"""
        if self.verbose:
            print(f"[ENHANCED] Processing: {doc_file.blob_name}")
        
        start_time = time.time()
        
        # Use base method to get initial chunks
        chunks = super().chunk_document(doc_file)
        
        if self.verbose:
            print(f"[ENHANCED] Base chunking generated {len(chunks)} chunks")
        
        # Step 1: Enhance chunks with reference context
        if self.enable_reference_preservation:
            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                previous_chunks = chunks[:i]  # Chunks processed before this one
                enhanced_chunk = self.reference_preserver.enhance_chunk_with_context(
                    chunk, previous_chunks
                )
                enhanced_chunks.append(enhanced_chunk)
            
            if self.verbose:
                ref_count = sum(1 for c in enhanced_chunks if c.metadata.get('has_references', False))
                print(f"[ENHANCED] Reference enhancement: {ref_count} chunks with references")
        else:
            enhanced_chunks = chunks
        
        # Step 2: Validate chunk quality
        quality_report = self.quality_validator.validate_chunks(enhanced_chunks)
        
        if self.verbose:
            print(f"[ENHANCED] Quality validation: {quality_report['average_quality_score']:.2f} avg score")
            print(f"[ENHANCED] Quality distribution: {quality_report['quality_distribution']}")
        
        # Step 3: Filter chunks based on quality threshold
        valid_chunks = self.quality_validator.filter_chunks(
            enhanced_chunks, 
            self.min_quality_threshold
        )
        
        if self.verbose:
            print(f"[ENHANCED] Quality filtering: {len(valid_chunks)}/{len(enhanced_chunks)} chunks passed")
        
        # Step 4: Add processing metadata to first chunk
        if valid_chunks:
            processing_metadata = {
                'processing_time': time.time() - start_time,
                'original_chunk_count': len(chunks),
                'enhanced_chunk_count': len(enhanced_chunks),
                'final_chunk_count': len(valid_chunks),
                'quality_report': {
                    'average_quality': quality_report['average_quality_score'],
                    'quality_distribution': quality_report['quality_distribution'],
                    'most_common_issues': quality_report['most_common_issues']
                }
            }
            
            # Add reference preservation stats if enabled
            if self.enable_reference_preservation:
                ref_validation = self.reference_preserver.validate_reference_integrity(valid_chunks)
                processing_metadata['reference_report'] = {
                    'total_references': ref_validation['total_references'],
                    'resolved_references': ref_validation['resolved_references'],
                    'reference_coverage': ref_validation['reference_coverage']
                }
            
            # Add to first chunk metadata
            valid_chunks[0].metadata['processing_metadata'] = processing_metadata
        
        # Update statistics
        self.processing_stats['total_documents'] += 1
        self.processing_stats['total_chunks_generated'] += len(chunks)
        self.processing_stats['total_chunks_filtered'] += len(valid_chunks)
        
        if valid_chunks:
            self.processing_stats['average_quality_score'] = quality_report['average_quality_score']
            
            if self.enable_reference_preservation:
                ref_validation = self.reference_preserver.validate_reference_integrity(valid_chunks)
                self.processing_stats['reference_coverage'] = ref_validation['reference_coverage']
        
        processing_time = time.time() - start_time
        if self.verbose:
            print(f"[ENHANCED] Processing completed in {processing_time:.2f}s")
            print(f"[ENHANCED] Final result: {len(valid_chunks)} high-quality chunks")
        
        return valid_chunks
    
    def chunk_documents(self, doc_files: List[DocumentFile]) -> List[Chunk]:
        """Process multiple documents with enhanced chunking"""
        if self.verbose:
            print(f"[ENHANCED] Processing {len(doc_files)} documents")
        
        all_chunks = []
        document_reports = []
        
        for doc_file in doc_files:
            try:
                chunks = self.chunk_document(doc_file)
                all_chunks.extend(chunks)
                
                # Collect document report
                if chunks and 'processing_metadata' in chunks[0].metadata:
                    doc_report = {
                        'document': doc_file.blob_name,
                        'chunks': len(chunks),
                        'processing_time': chunks[0].metadata['processing_metadata']['processing_time'],
                        'quality_score': chunks[0].metadata['processing_metadata']['quality_report']['average_quality']
                    }
                    document_reports.append(doc_report)
                
            except Exception as e:
                print(f"[ENHANCED] Error processing {doc_file.blob_name}: {e}")
                continue
        
        # Generate comprehensive report
        if self.verbose and document_reports:
            self._print_processing_summary(document_reports, all_chunks)
        
        return all_chunks
    
    def _print_processing_summary(self, document_reports: List[Dict], all_chunks: List[Chunk]):
        """Print comprehensive processing summary"""
        print("\n" + "="*60)
        print("ENHANCED CHUNKING PROCESSING SUMMARY")
        print("="*60)
        
        total_docs = len(document_reports)
        total_chunks = len(all_chunks)
        avg_quality = sum(r['quality_score'] for r in document_reports) / total_docs
        total_time = sum(r['processing_time'] for r in document_reports)
        
        print(f"Documents Processed: {total_docs}")
        print(f"Total Chunks Generated: {total_chunks}")
        print(f"Average Quality Score: {avg_quality:.2f}")
        print(f"Total Processing Time: {total_time:.2f}s")
        print(f"Average Time per Document: {total_time/total_docs:.2f}s")
        
        # Reference preservation summary
        if self.enable_reference_preservation:
            ref_validation = self.reference_preserver.validate_reference_integrity(all_chunks)
            print(f"Reference Coverage: {ref_validation['reference_coverage']:.2%}")
            print(f"Total References: {ref_validation['total_references']}")
        
        print("\nDocument Details:")
        for report in document_reports:
            print(f"  {report['document']}: {report['chunks']} chunks, "
                  f"quality: {report['quality_score']:.2f}, "
                  f"time: {report['processing_time']:.2f}s")
        
        print("="*60)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return {
            'processing_stats': self.processing_stats.copy(),
            'configuration': {
                'max_tokens': getattr(self, 'max_tokens', 900),
                'overlap_sentences': self.splitter.overlap_sentences,
                'min_quality_threshold': self.min_quality_threshold,
                'enable_reference_preservation': self.enable_reference_preservation,
                'preserve_paragraphs': self.splitter.preserve_paragraphs
            }
        }
    
    def generate_quality_report(self, chunks: List[Chunk]) -> str:
        """Generate comprehensive quality report"""
        return self.quality_validator.generate_quality_report(chunks)
    
    def generate_reference_report(self, chunks: List[Chunk]) -> str:
        """Generate comprehensive reference report"""
        if not self.enable_reference_preservation:
            return "Reference preservation is disabled"
        return self.reference_preserver.generate_reference_report(chunks)
