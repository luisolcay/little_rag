import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage } from '../types';
import { CitationList } from './CitationList';
import { TypingIndicator } from './LoadingIndicator';

interface MessageListProps {
  messages: ChatMessage[];
  isLoading: boolean;
  className?: string;
}

export function MessageList({ messages, isLoading, className = '' }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className={`flex-1 overflow-y-auto p-4 space-y-4 ${className}`}>
      {messages.length === 0 && !isLoading && (
        <div className="flex items-center justify-center h-full">
          <div className="text-center text-gray-500">
            <svg className="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            <p className="text-lg font-medium">Welcome to RAG Chat</p>
            <p className="text-sm">Ask questions about your documents and get answers with sources.</p>
          </div>
        </div>
      )}

      {messages.map((message, index) => (
        <div
          key={`${message.timestamp}-${index}`}
          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          <div
            className={`max-w-3xl px-4 py-3 rounded-lg ${
              message.role === 'user'
                ? 'bg-primary-600 text-white'
                : 'bg-white border border-gray-200 shadow-sm'
            }`}
          >
            <div className="prose prose-sm max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({node, children, ...props}: any) => <h1 className="text-xl font-bold mb-2 mt-3 first:mt-0" {...props}>{children}</h1>,
                  h2: ({node, children, ...props}: any) => <h2 className="text-lg font-semibold mb-2 mt-3 first:mt-0" {...props}>{children}</h2>,
                  h3: ({node, children, ...props}: any) => <h3 className="text-base font-semibold mb-2 mt-3 first:mt-0" {...props}>{children}</h3>,
                  ul: ({node, ...props}: any) => <ul className="list-disc pl-5 space-y-1 my-2" {...props} />,
                  ol: ({node, ...props}: any) => <ol className="list-decimal pl-5 space-y-1 my-2" {...props} />,
                  li: ({node, ...props}: any) => <li className="leading-relaxed" {...props} />,
                  p: ({node, ...props}: any) => <p className="mb-2 last:mb-0" {...props} />,
                  strong: ({node, ...props}: any) => <strong className="font-semibold" {...props} />,
                  em: ({node, ...props}: any) => <em className="italic" {...props} />,
                  code: ({node, inline, ...props}: any) => 
                    inline ? (
                      <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono" {...props} />
                    ) : (
                      <code className="block bg-gray-100 p-2 rounded text-sm font-mono overflow-x-auto" {...props} />
                    ),
                  blockquote: ({node, ...props}: any) => <blockquote className="border-l-4 border-gray-300 pl-4 italic my-2" {...props} />,
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
            
            {message.citations && message.citations.length > 0 && (
              <div className="mt-4">
                <CitationList citations={message.citations} />
              </div>
            )}
            
            <div className={`text-xs mt-2 ${
              message.role === 'user' ? 'text-primary-100' : 'text-gray-500'
            }`}>
              {formatTimestamp(message.timestamp)}
            </div>
          </div>
        </div>
      ))}

      {isLoading && (
        <div className="flex justify-start">
          <div className="bg-white border border-gray-200 rounded-lg px-4 py-3 shadow-sm">
            <TypingIndicator />
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
}
