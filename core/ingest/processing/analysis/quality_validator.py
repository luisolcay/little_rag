import re
from typing import List, Dict, Any
from ..models import Chunk


class ChunkQualityValidator:
    """Validates chunk quality and provides comprehensive metrics"""
    
    def __init__(self, 
                 min_chunk_length: int = 50,
                 max_chunk_length: int = 2000,
                 min_sentence_count: int = 1,
                 max_repetition_ratio: float = 0.3,
                 max_special_char_ratio: float = 0.3):
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.min_sentence_count = min_sentence_count
        self.max_repetition_ratio = max_repetition_ratio
        self.max_special_char_ratio = max_special_char_ratio
    
    def validate_chunk(self, chunk: Chunk) -> Dict[str, Any]:
        """Validate individual chunk and return quality metrics"""
        content = chunk.content.strip()
        
        metrics = {
            'is_valid': True,
            'quality_score': 1.0,
            'issues': [],
            'suggestions': [],
            'statistics': {
                'length': len(content),
                'word_count': len(content.split()),
                'sentence_count': len([s for s in content.split('.') if s.strip()]),
                'character_ratio': len(content) / max(len(content), 1)
            }
        }
        
        # Length validation
        if len(content) < self.min_chunk_length:
            metrics['is_valid'] = False
            metrics['issues'].append('chunk_too_short')
            metrics['quality_score'] *= 0.3
            metrics['suggestions'].append('Consider merging with adjacent chunk')
            
        if len(content) > self.max_chunk_length:
            metrics['issues'].append('chunk_too_long')
            metrics['quality_score'] *= 0.8
            metrics['suggestions'].append('Consider splitting into smaller chunks')
            
        # Repetition analysis
        words = content.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_repetition = max(word_counts.values())
            repetition_ratio = max_repetition / len(words)
            metrics['statistics']['repetition_ratio'] = repetition_ratio
            
            if repetition_ratio > self.max_repetition_ratio:
                metrics['issues'].append('high_repetition')
                metrics['quality_score'] *= 0.6
                metrics['suggestions'].append('High word repetition detected')
                
        # Sentence structure validation
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        metrics['statistics']['sentence_count'] = len(sentences)
        
        if len(sentences) < self.min_sentence_count:
            metrics['issues'].append('incomplete_sentence')
            metrics['quality_score'] *= 0.7
            metrics['suggestions'].append('Chunk may contain incomplete sentences')
            
        # Special character analysis
        special_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
        special_char_ratio = special_chars / max(len(content), 1)
        metrics['statistics']['special_char_ratio'] = special_char_ratio
        
        if special_char_ratio > self.max_special_char_ratio:
            metrics['issues'].append('too_many_special_chars')
            metrics['quality_score'] *= 0.8
            metrics['suggestions'].append('High ratio of special characters')
            
        # Content coherence analysis
        if len(words) > 5:
            # Check for meaningful content (not just numbers/symbols)
            alpha_chars = sum(1 for c in content if c.isalpha())
            alpha_ratio = alpha_chars / max(len(content), 1)
            metrics['statistics']['alpha_ratio'] = alpha_ratio
            
            if alpha_ratio < 0.3:
                metrics['issues'].append('low_alpha_content')
                metrics['quality_score'] *= 0.7
                metrics['suggestions'].append('Low alphabetic content ratio')
                
        # Whitespace analysis
        whitespace_ratio = content.count(' ') / max(len(content), 1)
        metrics['statistics']['whitespace_ratio'] = whitespace_ratio
        
        if whitespace_ratio > 0.4:
            metrics['issues'].append('excessive_whitespace')
            metrics['quality_score'] *= 0.9
            metrics['suggestions'].append('High whitespace ratio')
            
        return metrics
    
    def validate_chunks(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Validate chunk list and return aggregate statistics"""
        if not chunks:
            return {
                'total_chunks': 0,
                'valid_chunks': 0,
                'invalid_chunks': 0,
                'average_quality_score': 0.0,
                'chunk_details': []
            }
            
        results = []
        for chunk in chunks:
            result = self.validate_chunk(chunk)
            result['chunk_id'] = chunk.id
            result['chunk_index'] = chunks.index(chunk)
            results.append(result)
            
        valid_chunks = [r for r in results if r['is_valid']]
        avg_quality = sum(r['quality_score'] for r in results) / len(results)
        
        # Calculate quality distribution
        quality_distribution = {
            'excellent': len([r for r in results if r['quality_score'] >= 0.9]),
            'good': len([r for r in results if 0.7 <= r['quality_score'] < 0.9]),
            'fair': len([r for r in results if 0.5 <= r['quality_score'] < 0.7]),
            'poor': len([r for r in results if r['quality_score'] < 0.5])
        }
        
        # Most common issues
        all_issues = []
        for result in results:
            all_issues.extend(result['issues'])
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            'total_chunks': len(chunks),
            'valid_chunks': len(valid_chunks),
            'invalid_chunks': len(chunks) - len(valid_chunks),
            'average_quality_score': avg_quality,
            'quality_distribution': quality_distribution,
            'most_common_issues': issue_counts,
            'chunk_details': results
        }
    
    def filter_chunks(self, chunks: List[Chunk], min_quality_threshold: float = 0.5) -> List[Chunk]:
        """Filter chunks based on quality threshold"""
        filtered_chunks = []
        
        for chunk in chunks:
            metrics = self.validate_chunk(chunk)
            if metrics['quality_score'] >= min_quality_threshold:
                # Add quality metrics to chunk metadata
                enhanced_metadata = chunk.metadata.copy()
                enhanced_metadata['quality_metrics'] = metrics
                filtered_chunk = Chunk(chunk.content, enhanced_metadata)
                filtered_chunks.append(filtered_chunk)
        
        return filtered_chunks
    
    def generate_quality_report(self, chunks: List[Chunk]) -> str:
        """Generate a human-readable quality report"""
        validation_results = self.validate_chunks(chunks)
        
        report = f"""
=== Chunk Quality Report ===
Total Chunks: {validation_results['total_chunks']}
Valid Chunks: {validation_results['valid_chunks']}
Invalid Chunks: {validation_results['invalid_chunks']}
Average Quality Score: {validation_results['average_quality_score']:.2f}

Quality Distribution:
- Excellent (≥0.9): {validation_results['quality_distribution']['excellent']}
- Good (0.7-0.9): {validation_results['quality_distribution']['good']}
- Fair (0.5-0.7): {validation_results['quality_distribution']['fair']}
- Poor (<0.5): {validation_results['quality_distribution']['poor']}

Most Common Issues:
"""
        
        for issue, count in validation_results['most_common_issues'].items():
            report += f"- {issue}: {count} chunks\n"
            
        return report
