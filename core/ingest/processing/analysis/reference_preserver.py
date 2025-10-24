"""
Reference Preservation System for Document Chunks
=================================================

This module provides intelligent reference detection and preservation
across document chunks to maintain context and traceability.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from ..models import Chunk

@dataclass
class Reference:
    """Represents a detected reference in a chunk."""
    reference_type: str  # 'explicit', 'implicit', 'cross_reference'
    reference_value: str
    reference_position: int
    reference_context: str
    confidence: float

class ReferencePreserver:
    """
    Advanced reference preservation system for document chunks.
    
    Features:
    - Explicit reference detection (e.g., "see section 3.2")
    - Implicit reference detection (e.g., "as mentioned above")
    - Cross-reference linking between chunks
    - Context preservation and enhancement
    - Reference validation and integrity checking
    """
    
    def __init__(self):
        # Explicit reference patterns
        self.explicit_patterns = [
            r'(?:see|refer to|consult|check)\s+(?:section|chapter|page|figure|table|appendix)\s*(\d+(?:\.\d+)*)',
            r'(?:section|chapter|page|figure|table|appendix)\s*(\d+(?:\.\d+)*)',
            r'(?:as shown in|as illustrated in|as described in)\s+(?:figure|table|section|chapter)\s*(\d+(?:\.\d+)*)',
            r'(?:according to|based on)\s+(?:section|chapter|figure|table)\s*(\d+(?:\.\d+)*)',
            r'(?:see|refer to)\s+(?:above|below|previously|earlier)',
        ]
        
        # Implicit reference patterns
        self.implicit_patterns = [
            r'(?:as mentioned|as stated|as discussed|as noted)\s+(?:above|below|previously|earlier)',
            r'(?:this|the above|the following)\s+(?:section|chapter|figure|table|example)',
            r'(?:in the previous|in the following|in the next|in the last)\s+(?:section|chapter|paragraph)',
            r'(?:as we|as I|as the)\s+(?:mentioned|stated|discussed|noted)\s+(?:above|below|previously)',
        ]
        
        # Cross-reference patterns
        self.cross_reference_patterns = [
            r'(?:see also|compare with|contrast with|similar to)\s+(?:section|chapter|figure|table)\s*(\d+(?:\.\d+)*)',
            r'(?:related to|connected to|associated with)\s+(?:section|chapter|figure|table)\s*(\d+(?:\.\d+)*)',
        ]
        
        # Compiled regex patterns for performance
        self.compiled_explicit = [re.compile(pattern, re.IGNORECASE) for pattern in self.explicit_patterns]
        self.compiled_implicit = [re.compile(pattern, re.IGNORECASE) for pattern in self.implicit_patterns]
        self.compiled_cross = [re.compile(pattern, re.IGNORECASE) for pattern in self.cross_reference_patterns]
    
    def enhance_chunk_with_context(self, chunk: Chunk, previous_chunks: List[Chunk]) -> Chunk:
        """
        Enhance a chunk with reference context from previous chunks.
        
        Args:
            chunk: Current chunk to enhance
            previous_chunks: List of previously processed chunks
            
        Returns:
            Enhanced chunk with reference information
        """
        # Detect references in current chunk
        references = self._detect_references(chunk.content)
        
        # Resolve references using previous chunks
        resolved_references = self._resolve_references(references, previous_chunks)
        
        # Create enhanced metadata
        enhanced_metadata = chunk.metadata.copy()
        enhanced_metadata.update({
            'has_references': len(references) > 0,
            'reference_count': len(references),
            'resolved_references': resolved_references,
            'reference_types': list(set(ref.reference_type for ref in references)),
            'reference_confidence': sum(ref.confidence for ref in references) / len(references) if references else 0.0
        })
        
        # Create enhanced chunk
        enhanced_chunk = Chunk(
            content=chunk.content,
            metadata=enhanced_metadata,
            id=chunk.id
        )
        
        return enhanced_chunk
    
    def _detect_references(self, content: str) -> List[Reference]:
        """
        Detect all references in the given content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            List of detected references
        """
        references = []
        
        # Detect explicit references
        explicit_refs = self._detect_explicit_references(content)
        references.extend(explicit_refs)
        
        # Detect implicit references
        implicit_refs = self._detect_implicit_references(content)
        references.extend(implicit_refs)
        
        # Detect cross-references
        cross_refs = self._detect_cross_references(content)
        references.extend(cross_refs)
        
        return references
    
    def _detect_explicit_references(self, content: str) -> List[Reference]:
        """Detect explicit references in content."""
        references = []
        
        for i, pattern in enumerate(self.compiled_explicit):
            matches = pattern.finditer(content)
            for match in matches:
                reference_value = match.group(1) if match.groups() else match.group(0)
                
                # Extract context around the reference
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]
                
                references.append(Reference(
                    reference_type='explicit',
                    reference_value=reference_value,
                    reference_position=match.start(),
                    reference_context=context,
                    confidence=0.9
                ))
        
        return references
    
    def _detect_implicit_references(self, content: str) -> List[Reference]:
        """Detect implicit references in content."""
        references = []
        
        for i, pattern in enumerate(self.compiled_implicit):
            matches = pattern.finditer(content)
            for match in matches:
                reference_value = match.group(0)
                
                # Extract context around the reference
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]
                
                references.append(Reference(
                    reference_type='implicit',
                    reference_value=reference_value,
                    reference_position=match.start(),
                    reference_context=context,
                    confidence=0.7
                ))
        
        return references
    
    def _detect_cross_references(self, content: str) -> List[Reference]:
        """Detect cross-references in content."""
        references = []
        
        for i, pattern in enumerate(self.compiled_cross):
            matches = pattern.finditer(content)
            for match in matches:
                reference_value = match.group(1) if match.groups() else match.group(0)
                
                # Extract context around the reference
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]
                
                references.append(Reference(
                    reference_type='cross_reference',
                    reference_value=reference_value,
                    reference_position=match.start(),
                    reference_context=context,
                    confidence=0.8
                ))
        
        return references
    
    def _resolve_references(self, references: List[Reference], previous_chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        Resolve references using previous chunks.
        
        Args:
            references: List of detected references
            previous_chunks: List of previous chunks for context
            
        Returns:
            List of resolved reference information
        """
        resolved = []
        
        for ref in references:
            resolution = {
                'reference_type': ref.reference_type,
                'reference_value': ref.reference_value,
                'reference_position': ref.reference_position,
                'reference_context': ref.reference_context,
                'confidence': ref.confidence,
                'resolved': False,
                'resolution_context': None
            }
            
            # Try to resolve explicit references
            if ref.reference_type == 'explicit':
                resolution_context = self._resolve_explicit_reference(ref, previous_chunks)
                if resolution_context:
                    resolution['resolved'] = True
                    resolution['resolution_context'] = resolution_context
            
            # Try to resolve implicit references
            elif ref.reference_type == 'implicit':
                resolution_context = self._resolve_implicit_reference(ref, previous_chunks)
                if resolution_context:
                    resolution['resolved'] = True
                    resolution['resolution_context'] = resolution_context
            
            resolved.append(resolution)
        
        return resolved
    
    def _resolve_explicit_reference(self, reference: Reference, previous_chunks: List[Chunk]) -> Optional[str]:
        """Resolve explicit reference using previous chunks."""
        ref_value = reference.reference_value
        
        # Look for matching section numbers, figures, etc.
        for chunk in reversed(previous_chunks):  # Start from most recent
            content = chunk.content.lower()
            
            # Check for section numbers
            if re.search(rf'\b{re.escape(ref_value)}\b', content):
                return chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            
            # Check for figure/table references
            if re.search(rf'(?:figure|table|fig\.|tab\.)\s*{re.escape(ref_value)}', content):
                return chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        
        return None
    
    def _resolve_implicit_reference(self, reference: Reference, previous_chunks: List[Chunk]) -> Optional[str]:
        """Resolve implicit reference using previous chunks."""
        ref_value = reference.reference_value.lower()
        
        # Look for contextually similar content
        for chunk in reversed(previous_chunks):
            content = chunk.content.lower()
            
            # Check for similar topics or concepts
            if any(keyword in content for keyword in ['above', 'previously', 'earlier', 'mentioned']):
                return chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        
        return None
    
    def validate_reference_integrity(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Validate reference integrity across all chunks.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Validation results
        """
        total_references = 0
        resolved_references = 0
        
        for chunk in chunks:
            if chunk.metadata.get('has_references', False):
                ref_count = chunk.metadata.get('reference_count', 0)
                total_references += ref_count
                
                resolved_refs = chunk.metadata.get('resolved_references', [])
                resolved_count = sum(1 for ref in resolved_refs if ref.get('resolved', False))
                resolved_references += resolved_count
        
        reference_coverage = resolved_references / total_references if total_references > 0 else 0.0
        
        return {
            'total_references': total_references,
            'resolved_references': resolved_references,
            'reference_coverage': reference_coverage,
            'validation_passed': reference_coverage >= 0.7  # 70% resolution threshold
        }
    
    def generate_reference_report(self, chunks: List[Chunk]) -> str:
        """Generate a comprehensive reference report."""
        validation = self.validate_reference_integrity(chunks)
        
        report = f"""
Reference Preservation Report
============================

Total References: {validation['total_references']}
Resolved References: {validation['resolved_references']}
Reference Coverage: {validation['reference_coverage']:.2%}
Validation Status: {'PASSED' if validation['validation_passed'] else 'FAILED'}

Chunk Reference Details:
"""
        
        for i, chunk in enumerate(chunks):
            if chunk.metadata.get('has_references', False):
                ref_count = chunk.metadata.get('reference_count', 0)
                ref_types = chunk.metadata.get('reference_types', [])
                confidence = chunk.metadata.get('reference_confidence', 0.0)
                
                report += f"\nChunk {i+1}:"
                report += f"  - References: {ref_count}"
                report += f"  - Types: {', '.join(ref_types)}"
                report += f"  - Confidence: {confidence:.2f}"
        
        return report
