import React from 'react';
import { Citation as CitationType } from '../types';
import { CitationComponent } from './Citation';

interface CitationListProps {
  citations: CitationType[];
  className?: string;
}

export function CitationList({ citations, className = '' }: CitationListProps) {
  if (!citations || citations.length === 0) {
    return null;
  }

  return (
    <div className={`space-y-3 ${className}`}>
      <div className="flex items-center space-x-2 text-sm text-gray-600 mb-2">
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
        </svg>
        <span className="font-medium">
          Fuentes ({citations.length})
        </span>
      </div>
      
      <div className="space-y-3">
        {citations.map((citation, index) => (
          <CitationComponent
            key={citation.chunk_id}
            citation={citation}
            index={index}
          />
        ))}
      </div>
    </div>
  );
}
