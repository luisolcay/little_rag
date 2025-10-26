import axios, { AxiosResponse } from 'axios';
import { ConversationHistory, ApiError, Citation } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export interface SendMessageRequest {
  query: string;
  session_id?: string;
  query_type?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
  structured_output?: boolean;
  retrieval_strategy?: string;
}

export interface SendMessageResponse {
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

class ApiService {
  /**
   * Send a message to the chat endpoint
   */
  async sendMessage(request: SendMessageRequest): Promise<SendMessageResponse> {
    try {
      const response: AxiosResponse<SendMessageResponse> = await api.post('/llm/chat', request);
      return response.data;
    } catch (error: any) {
      throw this.handleError(error);
    }
  }

  /**
   * Get conversation history for a session
   */
  async getConversation(sessionId: string): Promise<ConversationHistory> {
    console.log('[API] getConversation called with sessionId:', sessionId);
    try {
      const response: AxiosResponse<ConversationHistory> = await api.get(`/llm/conversation/${sessionId}`);
      console.log('[API] getConversation response:', response.data);
      return response.data;
    } catch (error: any) {
      console.error('[API] getConversation error:', error);
      throw this.handleError(error);
    }
  }

  /**
   * Clear conversation for a session
   */
  async clearConversation(sessionId: string): Promise<{ message: string }> {
    try {
      const response: AxiosResponse<{ message: string }> = await api.delete(`/llm/conversation/${sessionId}`);
      return response.data;
    } catch (error: any) {
      throw this.handleError(error);
    }
  }

  /**
   * Get API health status
   */
  async getHealthStatus(): Promise<any> {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error: any) {
      throw this.handleError(error);
    }
  }

  /**
   * Get LLM service health status
   */
  async getLLMHealthStatus(): Promise<any> {
    try {
      const response = await api.get('/llm/health');
      return response.data;
    } catch (error: any) {
      throw this.handleError(error);
    }
  }

  /**
   * Handle API errors consistently
   */
  private handleError(error: any): Error {
    if (error.response) {
      // Server responded with error status
      const apiError: ApiError = error.response.data;
      return new Error(apiError.detail || `Server error: ${error.response.status}`);
    } else if (error.request) {
      // Request was made but no response received
      return new Error('No response from server. Please check your connection.');
    } else {
      // Something else happened
      return new Error(error.message || 'An unexpected error occurred');
    }
  }
}

export const apiService = new ApiService();
export default apiService;
