"""
Automatic Repetitive Pattern Detection for Chunks
=================================================

This class automatically detects repetitive content (like headers)
in PDF documents to improve chunk quality.
"""

import re
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher
from collections import Counter
import json
from datetime import datetime

class RepetitivePatternDetector:
    """
    Detects repetitive patterns in document chunks.
    
    Features:
    - Chunk similarity detection
    - Common header identification
    - Position-based pattern analysis
    - Automatic cleaning rule generation
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.8,
                 min_pattern_length: int = 50,
                 max_pattern_length: int = 500):
        """
        Initializes the repetitive pattern detector.
        
        Args:
            similarity_threshold: Similarity threshold to consider content repetitive
            min_pattern_length: Minimum pattern length to consider
            max_pattern_length: Maximum pattern length to analyze
        """
        self.similarity_threshold = similarity_threshold
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.detected_patterns = []
        self.pattern_statistics = {}
    
    def detect_repetitive_content(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detects repetitive content in a list of chunks.
        
        Args:
            chunks: List of chunks with content and metadata
            
        Returns:
            Dictionary with detected patterns and statistics
        """
        print(f"[PATTERN_DETECTOR] Analyzing {len(chunks)} chunks for repetitive patterns...")
        
        # 1. Chunk similarity analysis
        similarity_patterns = self._analyze_chunk_similarity(chunks)
        
        # 2. Position-based pattern analysis
        position_patterns = self._analyze_position_patterns(chunks)
        
        # 3. Common lines analysis
        common_lines = self._analyze_common_lines(chunks)
        
        # 4. Generate statistics
        statistics = self._generate_statistics(chunks, similarity_patterns, position_patterns, common_lines)
        
        # 5. Consolidate results
        results = {
            'similarity_patterns': similarity_patterns,
            'position_patterns': position_patterns,
            'common_lines': common_lines,
            'statistics': statistics,
            'recommendations': self._generate_recommendations(statistics)
        }
        
        self.detected_patterns = results
        return results
    
    def _analyze_chunk_similarity(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyzes similarity between chunks to detect repetitive patterns."""
        patterns = []
        
        for i, chunk1 in enumerate(chunks):
            chunk1_content = chunk1.get('content', '')
            
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                chunk2_content = chunk2.get('content', '')
                
                # Analyze similarity in different segments
                similarities = self._calculate_segment_similarities(chunk1_content, chunk2_content)
                
                for segment_type, similarity in similarities.items():
                    if similarity['ratio'] > self.similarity_threshold:
                        patterns.append({
                            'type': 'similarity',
                            'segment_type': segment_type,
                            'similarity_ratio': similarity['ratio'],
                            'chunk_indices': [i, j],
                            'pattern_content': similarity['content'],
                            'pattern_length': len(similarity['content']),
                            'page_numbers': [
                                chunk1.get('metadata', {}).get('page_number', i+1),
                                chunk2.get('metadata', {}).get('page_number', j+1)
                            ]
                        })
        
        return patterns
    
    def _calculate_segment_similarities(self, content1: str, content2: str) -> Dict[str, Dict[str, Any]]:
        """Calculates similarities in different content segments."""
        similarities = {}
        
        # 1. Similarity at the beginning (first N characters)
        start_segment = min(500, len(content1), len(content2))
        if start_segment > self.min_pattern_length:
            start1 = content1[:start_segment]
            start2 = content2[:start_segment]
            ratio = SequenceMatcher(None, start1, start2).ratio()
            similarities['start'] = {
                'ratio': ratio,
                'content': start1,
                'segment_length': start_segment
            }
        
        # 2. Similarity in individual lines
        lines1 = content1.split('\n')
        lines2 = content2.split('\n')
        
        common_lines = []
        for line1 in lines1[:10]:  # Analyze first 10 lines
            for line2 in lines2[:10]:
                if len(line1.strip()) > 10:  # Ignore very short lines
                    ratio = SequenceMatcher(None, line1.strip(), line2.strip()).ratio()
                    if ratio > 0.9:  # Almost identical lines
                        common_lines.append({
                            'line': line1.strip(),
                            'similarity': ratio
                        })
        
        if common_lines:
            similarities['common_lines'] = {
                'ratio': max(line['similarity'] for line in common_lines),
                'content': common_lines[0]['line'],
                'common_lines_count': len(common_lines)
            }
        
        return similarities
    
    def _analyze_position_patterns(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes patterns based on content position."""
        position_analysis = {
            'header_patterns': [],
            'footer_patterns': [],
            'repeated_phrases': []
        }
        
        # Analyze first lines of each chunk
        first_lines = []
        for chunk in chunks:
            content = chunk.get('content', '')
            lines = content.split('\n')
            if lines:
                first_lines.append(lines[0].strip())
        
        # Detect lines that appear frequently at the beginning
        line_counts = Counter(first_lines)
        for line, count in line_counts.items():
            if count > 1 and len(line) > 10:
                position_analysis['header_patterns'].append({
                    'line': line,
                    'frequency': count,
                    'percentage': (count / len(chunks)) * 100
                })
        
        # Analyze page numbering patterns
        page_patterns = self._detect_page_numbering_patterns(chunks)
        if page_patterns:
            position_analysis['page_numbering'] = page_patterns
        
        return position_analysis
    
    def _detect_page_numbering_patterns(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detects page numbering patterns."""
        page_patterns = []
        
        # Search for patterns like "Página: X de Y"
        page_regex = r'Página:\s*(\d+)\s*de\s*(\d+)'
        
        for chunk in chunks:
            content = chunk.get('content', '')
            matches = re.findall(page_regex, content)
            
            if matches:
                for match in matches:
                    page_patterns.append({
                        'pattern': f'Página: {match[0]} de {match[1]}',
                        'current_page': int(match[0]),
                        'total_pages': int(match[1]),
                        'chunk_index': chunks.index(chunk)
                    })
        
        return page_patterns
    
    def _analyze_common_lines(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes common lines between chunks."""
        all_lines = []
        
        # Extract all lines from all chunks
        for chunk in chunks:
            content = chunk.get('content', '')
            lines = content.split('\n')
            for line in lines:
                if len(line.strip()) > 5:  # Ignore very short lines
                    all_lines.append(line.strip())
        
        # Count line frequency
        line_counts = Counter(all_lines)
        
        # Identify very common lines
        common_lines = []
        total_chunks = len(chunks)
        
        for line, count in line_counts.items():
            if count > 1:  # Appears in more than one chunk
                frequency_percentage = (count / total_chunks) * 100
                common_lines.append({
                    'line': line,
                    'frequency': count,
                    'frequency_percentage': frequency_percentage,
                    'is_likely_header': frequency_percentage > 50  # If appears in more than 50% of chunks
                })
        
        # Sort by frequency
        common_lines.sort(key=lambda x: x['frequency'], reverse=True)
        
        return {
            'common_lines': common_lines,
            'total_unique_lines': len(line_counts),
            'repetitive_lines_count': len(common_lines)
        }
    
    def _generate_statistics(self, chunks: List[Dict[str, Any]], 
                           similarity_patterns: List[Dict[str, Any]],
                           position_patterns: Dict[str, Any],
                           common_lines: Dict[str, Any]) -> Dict[str, Any]:
        """Generates pattern analysis statistics."""
        
        # Calculate noise metrics
        total_chars = sum(len(chunk.get('content', '')) for chunk in chunks)
        repetitive_chars = 0
        
        # Estimate repetitive characters based on common lines
        for line_info in common_lines['common_lines']:
            if line_info['frequency'] > 1:
                repetitive_chars += len(line_info['line']) * line_info['frequency']
        
        noise_percentage = (repetitive_chars / total_chars) * 100 if total_chars > 0 else 0
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'repetitive_characters': repetitive_chars,
            'noise_percentage': noise_percentage,
            'similarity_patterns_count': len(similarity_patterns),
            'header_patterns_count': len(position_patterns.get('header_patterns', [])),
            'common_lines_count': len(common_lines['common_lines']),
            'average_chunk_length': total_chars / len(chunks) if chunks else 0,
            'most_common_line': common_lines['common_lines'][0] if common_lines['common_lines'] else None
        }
    
    def _generate_recommendations(self, statistics: Dict[str, Any]) -> List[str]:
        """Generates recommendations based on statistics."""
        recommendations = []
        
        noise_percentage = statistics.get('noise_percentage', 0)
        
        if noise_percentage > 20:
            recommendations.append(f"HIGH NOISE DETECTED: {noise_percentage:.1f}% of content is repetitive")
            recommendations.append("Recommendation: Implement automatic header cleaning")
        
        if statistics.get('header_patterns_count', 0) > 0:
            recommendations.append("HEADERS DETECTED: Repetitive header patterns found")
            recommendations.append("Recommendation: Configure specific header filter")
        
        if statistics.get('common_lines_count', 0) > 5:
            recommendations.append("MULTIPLE REPETITIVE LINES: Many common lines detected")
            recommendations.append("Recommendation: Review chunking configuration")
        
        if noise_percentage < 5:
            recommendations.append("LOW NOISE: Document has good chunking quality")
        
        return recommendations
    
    def generate_cleaning_rules(self) -> List[Dict[str, Any]]:
        """Generates cleaning rules based on detected patterns."""
        if not self.detected_patterns:
            return []
        
        cleaning_rules = []
        
        # Rule for frequent common lines
        common_lines = self.detected_patterns.get('common_lines', {}).get('common_lines', [])
        for line_info in common_lines:
            if line_info['frequency_percentage'] > 50:  # Appears in more than 50% of chunks
                cleaning_rules.append({
                    'type': 'remove_line',
                    'pattern': line_info['line'],
                    'description': f"Repetitive line that appears in {line_info['frequency_percentage']:.1f}% of chunks",
                    'confidence': min(1.0, line_info['frequency_percentage'] / 100)
                })
        
        # Rule for page numbering patterns
        page_patterns = self.detected_patterns.get('position_patterns', {}).get('page_numbering', [])
        if page_patterns:
            cleaning_rules.append({
                'type': 'remove_regex',
                'pattern': r'Página:\s*\d+\s*de\s*\d+',
                'description': 'Page numbering pattern detected',
                'confidence': 0.9
            })
        
        return cleaning_rules
    
    def save_analysis_report(self, filepath: str):
        """Saves analysis report to a JSON file."""
        if self.detected_patterns:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.detected_patterns, f, indent=2, ensure_ascii=False)
            print(f"[PATTERN_DETECTOR] Report saved to: {filepath}")
