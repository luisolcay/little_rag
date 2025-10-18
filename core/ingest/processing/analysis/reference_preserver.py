import re
from typing import List, Dict, Any, Tuple
from ..models import Chunk


class ReferencePreserver:
    """Preserves cross-references and context between chunks"""
    
    def __init__(self):
        # Common reference patterns with improved regex
        self.reference_patterns = {
            'page_refs': r'(?:page|p\.|página)\s*(\d+)',
            'section_refs': r'(?:section|sec\.|sección)\s*(\d+(?:\.\d+)*)',
            'figure_refs': r'(?:figure|fig\.|figura)\s*(\d+)',
            'table_refs': r'(?:table|tab\.|tabla)\s*(\d+)',
            'chapter_refs': r'(?:chapter|chap\.|capítulo)\s*(\d+)',
            'article_refs': r'(?:article|art\.|artículo)\s*(\d+)',
            'paragraph_refs': r'(?:paragraph|para\.|párrafo)\s*(\d+)',
            'clause_refs': r'(?:clause|cláusula)\s*(\d+)',
            'see_also': r'(?:see also|see|ver también|ver|v\.\s*tamb\.)',
            'as_mentioned': r'(?:as mentioned|mentioned above|como se mencionó|mencionado anteriormente)',
            'previously': r'(?:previously|earlier|anteriormente|antes)',
            'below': r'(?:below|más abajo|a continuación)',
            'above': r'(?:above|arriba|anterior)',
            'following': r'(?:following|siguiente)',
            'preceding': r'(?:preceding|precedente)',
        }
        
        # Contextual reference patterns
        self.contextual_patterns = {
            'this_document': r'(?:this document|este documento|el presente documento)',
            'same_page': r'(?:same page|misma página)',
            'next_page': r'(?:next page|página siguiente)',
            'previous_page': r'(?:previous page|página anterior)',
        }
    
    def extract_references(self, text: str) -> Dict[str, List[str]]:
        """Extract all references from text with enhanced detection"""
        references = {}
        
        for ref_type, pattern in self.reference_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                references[ref_type] = list(set(matches))  # Remove duplicates
        
        # Also extract contextual references
        for ref_type, pattern in self.contextual_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                references[ref_type] = matches
                
        return references
    
    def extract_implicit_references(self, text: str) -> List[Dict[str, str]]:
        """Extract implicit references that don't follow standard patterns"""
        implicit_refs = []
        
        # Look for patterns like "the above", "the following", etc.
        implicit_patterns = [
            r'(?:the|el|la)\s+(?:above|arriba|anterior)',
            r'(?:the|el|la)\s+(?:following|siguiente)',
            r'(?:the|el|la)\s+(?:previous|anterior)',
            r'(?:the|el|la)\s+(?:next|siguiente)',
            r'(?:as|como)\s+(?:stated|mencionado|indicado)',
            r'(?:as|como)\s+(?:shown|mostrado|demostrado)',
        ]
        
        for pattern in implicit_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                implicit_refs.append({
                    'type': 'implicit_reference',
                    'text': match.group(),
                    'position': match.start(),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        
        return implicit_refs
    
    def find_reference_context(self, reference_value: str, reference_type: str, 
                              previous_chunks: List[Chunk]) -> Dict[str, Any]:
        """Find the context for a specific reference in previous chunks"""
        context_info = {
            'found': False,
            'context_chunk_id': None,
            'context_preview': None,
            'reference_position': None,
            'surrounding_context': None
        }
        
        for chunk in previous_chunks:
            # Look for the reference value in the chunk content
            if reference_value in chunk.content:
                context_info['found'] = True
                context_info['context_chunk_id'] = chunk.id
                
                # Find the position of the reference
                ref_position = chunk.content.find(reference_value)
                context_info['reference_position'] = ref_position
                
                # Extract surrounding context
                start = max(0, ref_position - 100)
                end = min(len(chunk.content), ref_position + len(reference_value) + 100)
                context_info['surrounding_context'] = chunk.content[start:end]
                
                # Create a preview
                context_info['context_preview'] = chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content
                break
        
        return context_info
    
    def enhance_chunk_with_context(self, chunk: Chunk, previous_chunks: List[Chunk]) -> Chunk:
        """Enhance chunk with reference context and metadata"""
        references = self.extract_references(chunk.content)
        implicit_refs = self.extract_implicit_references(chunk.content)
        
        # Find context for each reference
        context_info = []
        for ref_type, ref_values in references.items():
            for ref_value in ref_values:
                try:
                    context = self.find_reference_context(ref_value, ref_type, previous_chunks)
                    if context['found']:
                        context_info.append({
                            'reference_type': ref_type,
                            'reference_value': ref_value,
                            'context_chunk_id': context['context_chunk_id'],
                            'context_preview': context['context_preview'],
                            'surrounding_context': context['surrounding_context']
                        })
                except Exception as e:
                    # Skip problematic references
                    continue
        
        # Add implicit references to context
        for implicit_ref in implicit_refs:
            context_info.append({
                'reference_type': 'implicit',
                'reference_text': implicit_ref['text'],
                'reference_position': implicit_ref['position'],
                'reference_context': implicit_ref['context']
            })
        
        # Enhance metadata
        enhanced_metadata = chunk.metadata.copy()
        enhanced_metadata['references'] = references
        enhanced_metadata['implicit_references'] = implicit_refs
        enhanced_metadata['context_info'] = context_info
        enhanced_metadata['reference_count'] = len(references) + len(implicit_refs)
        
        # Add reference summary
        if references or implicit_refs:
            enhanced_metadata['has_references'] = True
            enhanced_metadata['reference_summary'] = self._create_reference_summary(references, implicit_refs)
        else:
            enhanced_metadata['has_references'] = False
        
        return Chunk(chunk.content, enhanced_metadata)
    
    def _create_reference_summary(self, references: Dict[str, List[str]], 
                                 implicit_refs: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create a summary of all references found"""
        summary = {
            'total_references': sum(len(refs) for refs in references.values()),
            'reference_types': list(references.keys()),
            'implicit_references': len(implicit_refs),
            'most_common_type': None
        }
        
        if references:
            # Find most common reference type
            type_counts = {ref_type: len(refs) for ref_type, refs in references.items()}
            summary['most_common_type'] = max(type_counts, key=type_counts.get)
        
        return summary
    
    def validate_reference_integrity(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Validate that all references have proper context"""
        validation_results = {
            'total_references': 0,
            'resolved_references': 0,
            'unresolved_references': 0,
            'reference_coverage': 0.0,
            'issues': []
        }
        
        for i, chunk in enumerate(chunks):
            if 'references' in chunk.metadata:
                references = chunk.metadata['references']
                context_info = chunk.metadata.get('context_info', [])
                
                for ref_type, ref_values in references.items():
                    for ref_value in ref_values:
                        validation_results['total_references'] += 1
                        
                        # Check if reference has context
                        has_context = any(
                            ctx['reference_value'] == ref_value and ctx['reference_type'] == ref_type
                            for ctx in context_info
                        )
                        
                        if has_context:
                            validation_results['resolved_references'] += 1
                        else:
                            validation_results['unresolved_references'] += 1
                            validation_results['issues'].append({
                                'chunk_id': chunk.id,
                                'reference_type': ref_type,
                                'reference_value': ref_value,
                                'issue': 'unresolved_reference'
                            })
        
        # Calculate coverage
        if validation_results['total_references'] > 0:
            validation_results['reference_coverage'] = (
                validation_results['resolved_references'] / 
                validation_results['total_references']
            )
        
        return validation_results
    
    def generate_reference_report(self, chunks: List[Chunk]) -> str:
        """Generate a comprehensive reference report"""
        validation_results = self.validate_reference_integrity(chunks)
        
        report = f"""
=== Reference Preservation Report ===
Total References: {validation_results['total_references']}
Resolved References: {validation_results['resolved_references']}
Unresolved References: {validation_results['unresolved_references']}
Reference Coverage: {validation_results['reference_coverage']:.2%}

Issues Found:
"""
        
        for issue in validation_results['issues']:
            report += f"- {issue['reference_type']} '{issue['reference_value']}' in chunk {issue['chunk_id']}\n"
        
        return report
