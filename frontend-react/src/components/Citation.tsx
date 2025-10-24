import React, { useState } from 'react';
import { Citation } from '../types';

interface CitationProps {
  citation: Citation;
  index: number;
}

export function CitationComponent({ citation, index }: CitationProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getScoreLabel = (score: number) => {
    if (score >= 0.8) return 'High';
    if (score >= 0.6) return 'Medium';
    return 'Low';
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center space-x-2 mb-2">
            <div className="flex items-center space-x-1">
              <svg className="w-4 h-4 text-primary-600" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
              </svg>
              <span className="font-semibold text-primary-700 text-sm">
                {citation.document_name}
              </span>
            </div>
            {citation.page_number && (
              <span className="text-xs font-medium text-white bg-primary-600 px-2 py-1 rounded">
                Página {citation.page_number}
              </span>
            )}
          </div>
          
          <p className="text-sm text-gray-700 leading-relaxed">
            {isExpanded 
              ? citation.content_snippet 
              : `${citation.content_snippet.substring(0, 200)}${citation.content_snippet.length > 200 ? '...' : ''}`
            }
          </p>
          
          {citation.content_snippet.length > 200 && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-primary-600 hover:text-primary-700 text-xs font-medium mt-1"
            >
              {isExpanded ? 'Show less' : 'Show more'}
            </button>
          )}
        </div>
        
        <div className="ml-4 flex flex-col items-end space-y-2">
          <div className={`px-2 py-1 rounded-full text-xs font-medium ${getScoreColor(citation.relevance_score)}`}>
            {getScoreLabel(citation.relevance_score)} ({citation.relevance_score.toFixed(2)})
          </div>
          
          <div className="text-xs text-gray-500">
            #{index + 1}
          </div>
        </div>
      </div>
      
    </div>
  );
}
