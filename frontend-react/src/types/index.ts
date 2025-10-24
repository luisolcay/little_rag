// TypeScript interfaces matching backend Pydantic models

export interface Citation {
  chunk_id: string;
  document_name: string;
  page_number?: number;
  content_snippet: string;
  relevance_score: number;
  metadata?: Record<string, any>;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  citations?: Citation[];
  metadata?: Record<string, any>;
}

export interface ChatResponse {
  content: string;
  session_id: string;
  model: string;
  query_type?: string;
  tokens_used: number;
  processing_time: number;
  confidence_score?: number;
  citations?: Citation[];
  metadata?: Record<string, any>;
  created_at: string;
}

export interface Session {
  id: string;
  name: string;
  created_at: string;
  last_message?: string;
  last_activity: string;
  message_count: number;
}

export interface ConversationHistory {
  session_id: string;
  messages: ChatMessage[];
  summary?: string;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

export interface ApiError {
  detail: string;
  status_code?: number;
}

export interface LoadingState {
  isLoading: boolean;
  message?: string;
}

export interface SearchResult {
  chunk_id: string;
  content: string;
  score: number;
  metadata: Record<string, any>;
  relevance_score: number;
  document_name: string;
  page_number?: number;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total_results: number;
  search_time: number;
  search_type: string;
}
