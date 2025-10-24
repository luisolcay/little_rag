"""
Quality validation for document chunks.
"""

import re
from typing import List, Dict, Any
from ..models import Chunk

class ChunkQualityValidator:
    """Validates chunk quality based on various criteria."""
    
    def __init__(
        self,
        min_chunk_length: int = 50,
        max_chunk_length: int = 2000,
        min_sentence_count: int = 1,
        max_repetition_ratio: float = 0.3,
        max_special_char_ratio: float = 0.3
    ):
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.min_sentence_count = min_sentence_count
        self.max_repetition_ratio = max_repetition_ratio
        self.max_special_char_ratio = max_special_char_ratio
    
    def validate_chunks(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Validate a list of chunks and return quality report."""
        if not chunks:
            return {
                'average_quality_score': 0.0,
                'quality_distribution': {},
                'most_common_issues': [],
                'total_chunks': 0
            }
        
        quality_scores = []
        issues = []
        
        for chunk in chunks:
            score, chunk_issues = self._validate_single_chunk(chunk)
            quality_scores.append(score)
            issues.extend(chunk_issues)
        
        # Calculate statistics
        avg_score = sum(quality_scores) / len(quality_scores)
        
        # Quality distribution
        quality_dist = {
            'excellent': sum(1 for s in quality_scores if s >= 0.9),
            'good': sum(1 for s in quality_scores if 0.7 <= s < 0.9),
            'fair': sum(1 for s in quality_scores if 0.5 <= s < 0.7),
            'poor': sum(1 for s in quality_scores if s < 0.5)
        }
        
        # Most common issues
        issue_counts = {}
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        most_common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'average_quality_score': avg_score,
            'quality_distribution': quality_dist,
            'most_common_issues': most_common_issues,
            'total_chunks': len(chunks)
        }
    
    def _validate_single_chunk(self, chunk: Chunk) -> tuple[float, List[str]]:
        """Validate a single chunk and return score and issues."""
        content = chunk.content
        issues = []
        score = 1.0
        
        # Length validation
        if len(content) < self.min_chunk_length:
            issues.append("too_short")
            score -= 0.3
        elif len(content) > self.max_chunk_length:
            issues.append("too_long")
            score -= 0.2
        
        # Sentence count validation
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count < self.min_sentence_count:
            issues.append("insufficient_sentences")
            score -= 0.2
        
        # Repetition validation
        words = content.lower().split()
        if len(words) > 0:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_repetition = max(word_counts.values())
            repetition_ratio = max_repetition / len(words)
            
            if repetition_ratio > self.max_repetition_ratio:
                issues.append("high_repetition")
                score -= 0.2
        
        # Special character validation
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', content))
        if len(content) > 0:
            special_char_ratio = special_chars / len(content)
            
            if special_char_ratio > self.max_special_char_ratio:
                issues.append("high_special_chars")
                score -= 0.1
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score, issues
    
    def filter_chunks(self, chunks: List[Chunk], min_threshold: float = 0.5) -> List[Chunk]:
        """Filter chunks based on quality threshold."""
        valid_chunks = []
        
        for chunk in chunks:
            score, _ = self._validate_single_chunk(chunk)
            if score >= min_threshold:
                valid_chunks.append(chunk)
        
        return valid_chunks
    
    def generate_quality_report(self, chunks: List[Chunk]) -> str:
        """Generate a human-readable quality report."""
        report = self.validate_chunks(chunks)
        
        report_text = f"""
Chunk Quality Report
===================

Total Chunks: {report['total_chunks']}
Average Quality Score: {report['average_quality_score']:.2f}

Quality Distribution:
- Excellent (≥0.9): {report['quality_distribution']['excellent']}
- Good (0.7-0.9): {report['quality_distribution']['good']}
- Fair (0.5-0.7): {report['quality_distribution']['fair']}
- Poor (<0.5): {report['quality_distribution']['poor']}

Most Common Issues:
"""
        
        for issue, count in report['most_common_issues']:
            report_text += f"- {issue}: {count} occurrences\n"
        
        return report_text
