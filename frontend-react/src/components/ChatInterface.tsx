import React, { useState, useEffect, useCallback } from 'react';
import { ChatMessage } from '../types';
import { MessageList } from './MessageList';
import { MessageInput } from './MessageInput';
import { SessionManager } from './SessionManager';
import { LoadingIndicator } from './LoadingIndicator';
import { useSession } from '../contexts/SessionContext';
import { apiService } from '../services/api';

interface ChatInterfaceProps {
  className?: string;
}

export function ChatInterface({ className = '' }: ChatInterfaceProps) {
  const { getActiveSession, updateLastMessage } = useSession();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  const activeSession = getActiveSession();

  // Load conversation history when active session changes
  useEffect(() => {
    if (activeSession) {
      loadConversationHistory(activeSession.id);
    } else {
      setMessages([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSession?.id]);

  const loadConversationHistory = async (sessionId: string) => {
    console.log('[ChatInterface] loadConversationHistory called for:', sessionId);
    setIsLoadingHistory(true);
    setError(null);
    
    try {
      console.log('[ChatInterface] Calling apiService.getConversation...');
      const history = await apiService.getConversation(sessionId);
      console.log('[ChatInterface] History received:', history);
      console.log('[ChatInterface] Messages array:', history.messages);
      console.log('[ChatInterface] Messages length:', history.messages?.length);
      
      // Convert conversation history to ChatMessage format
      const chatMessages: ChatMessage[] = (history.messages || []).map((msg: any) => {
        console.log('[ChatInterface] Mapping message:', msg);
        return {
          role: msg.role as 'user' | 'assistant',
          content: msg.content,
          timestamp: msg.timestamp,
          citations: msg.metadata?.citations || [],
          metadata: msg.metadata
        };
      });
      
      console.log('[ChatInterface] Converted messages:', chatMessages);
      setMessages(chatMessages);
    } catch (err: any) {
      console.error('[ChatInterface] Failed to load conversation history:', err);
      console.error('[ChatInterface] Error details:', err.message, err.stack);
      // Don't show error for empty conversations, just start fresh
      if (err.message && (err.message.includes('404') || err.message.includes('not found'))) {
        setMessages([]);
      } else {
        setError('Failed to load conversation history');
      }
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const handleSendMessage = useCallback(async (messageText: string) => {
    console.log('[ChatInterface] handleSendMessage called:', messageText);
    console.log('[ChatInterface] activeSession:', activeSession);
    
    if (!activeSession) {
      console.error('[ChatInterface] NO ACTIVE SESSION!');
      setError('No active session. Please create or select a session.');
      return;
    }

    setIsLoading(true);
    setError(null);

    // Add user message immediately
    const userMessage: ChatMessage = {
      role: 'user',
      content: messageText,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);

    try {
      console.log('[ChatInterface] Calling apiService.sendMessage...');
      const response = await apiService.sendMessage({
        query: messageText,
        session_id: activeSession.id,
        query_type: 'general_qa',
        model: 'gpt-4o-mini',
        temperature: 0.1,
        max_tokens: 4096,
      });
      console.log('[ChatInterface] Response received:', response);

      // Add assistant response
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.content,
        timestamp: response.created_at,
        citations: response.citations || [],
        metadata: response.metadata,
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Update session with last message
      updateLastMessage(activeSession.id, messageText);

    } catch (err: any) {
      console.error('Failed to send message:', err);
      setError(err.message || 'Failed to send message. Please try again.');
      
      // Remove the user message if sending failed
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setIsLoading(false);
    }
  }, [activeSession, updateLastMessage]);

  const handleRetry = () => {
    setError(null);
  };

  if (isLoadingHistory) {
    return (
      <div className={`flex items-center justify-center h-full ${className}`}>
        <LoadingIndicator message="Loading conversation..." size="lg" />
      </div>
    );
  }

  return (
    <div className={`flex h-full ${className}`}>
      {/* Session Manager Sidebar */}
      <div className="w-80 border-r border-gray-200 flex-shrink-0">
        <SessionManager />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="border-b border-gray-200 bg-white px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-gray-900">
                {activeSession ? activeSession.name : 'No Session Selected'}
              </h1>
              {activeSession && (
                <p className="text-sm text-gray-500">
                  {activeSession.message_count} messages • Last active {new Date(activeSession.last_activity).toLocaleDateString()}
                </p>
              )}
            </div>
            
            {error && (
              <div className="flex items-center space-x-2">
                <div className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">
                  {error}
                </div>
                <button
                  onClick={handleRetry}
                  className="text-sm text-primary-600 hover:text-primary-700"
                >
                  Retry
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 flex flex-col min-h-0">
          {activeSession ? (
            <>
              <MessageList 
                messages={messages} 
                isLoading={isLoading}
                className="flex-1"
              />
              <MessageInput
                onSendMessage={handleSendMessage}
                isLoading={isLoading}
                placeholder="Ask a question about your documents..."
              />
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center bg-gray-50">
              <div className="text-center text-gray-500">
                <svg className="w-16 h-16 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <h3 className="text-lg font-medium mb-2">No Session Selected</h3>
                <p className="text-sm">Create a new session or select an existing one to start chatting.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
