import React, { useState } from 'react';
import { useSession } from '../contexts/SessionContext';

interface SessionManagerProps {
  className?: string;
}

export function SessionManager({ className = '' }: SessionManagerProps) {
  const { state, createSession, updateSession, deleteSession, setActiveSession, getActiveSession } = useSession();
  const [isCreating, setIsCreating] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');

  const activeSession = getActiveSession();

  const handleCreateSession = () => {
    if (newSessionName.trim()) {
      createSession(newSessionName.trim());
      setNewSessionName('');
      setIsCreating(false);
    }
  };

  const handleStartEditing = (sessionId: string, currentName: string) => {
    setEditingSessionId(sessionId);
    setEditingName(currentName);
  };

  const handleSaveEdit = () => {
    if (editingSessionId && editingName.trim()) {
      updateSession(editingSessionId, { name: editingName.trim() });
      setEditingSessionId(null);
      setEditingName('');
    }
  };

  const handleCancelEdit = () => {
    setEditingSessionId(null);
    setEditingName('');
  };

  const handleDeleteSession = (sessionId: string) => {
    if (window.confirm('Are you sure you want to delete this session? This action cannot be undone.')) {
      deleteSession(sessionId);
    }
  };

  const formatLastActivity = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60);
    
    if (diffInHours < 1) return 'Just now';
    if (diffInHours < 24) return `${Math.floor(diffInHours)}h ago`;
    if (diffInHours < 168) return `${Math.floor(diffInHours / 24)}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className={`bg-white border-b border-gray-200 ${className}`}>
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Sessions</h2>
          <button
            onClick={() => setIsCreating(true)}
            className="px-3 py-2 bg-primary-600 text-white text-sm rounded-lg hover:bg-primary-700 focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 transition-colors duration-200"
          >
            + New Session
          </button>
        </div>

        {/* Create new session form */}
        {isCreating && (
          <div className="mb-4 p-3 bg-gray-50 rounded-lg">
            <input
              type="text"
              value={newSessionName}
              onChange={(e) => setNewSessionName(e.target.value)}
              placeholder="Enter session name..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              autoFocus
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleCreateSession();
                if (e.key === 'Escape') setIsCreating(false);
              }}
            />
            <div className="flex space-x-2 mt-2">
              <button
                onClick={handleCreateSession}
                disabled={!newSessionName.trim()}
                className="px-3 py-1 bg-primary-600 text-white text-sm rounded hover:bg-primary-700 disabled:bg-gray-300"
              >
                Create
              </button>
              <button
                onClick={() => setIsCreating(false)}
                className="px-3 py-1 bg-gray-300 text-gray-700 text-sm rounded hover:bg-gray-400"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* Sessions list */}
        <div className="space-y-2">
          {state.sessions.length === 0 ? (
            <div className="text-center text-gray-500 py-4">
              <p className="text-sm">No sessions yet</p>
              <p className="text-xs">Create your first session to start chatting</p>
            </div>
          ) : (
            state.sessions.map((session) => (
              <div
                key={session.id}
                className={`p-3 rounded-lg border cursor-pointer transition-colors duration-200 ${
                  activeSession?.id === session.id
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                }`}
                onClick={() => setActiveSession(session.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    {editingSessionId === session.id ? (
                      <input
                        type="text"
                        value={editingName}
                        onChange={(e) => setEditingName(e.target.value)}
                        className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                        autoFocus
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleSaveEdit();
                          if (e.key === 'Escape') handleCancelEdit();
                        }}
                        onClick={(e) => e.stopPropagation()}
                      />
                    ) : (
                      <div className="flex items-center space-x-2">
                        <h3 className="text-sm font-medium text-gray-900 truncate">
                          {session.name}
                        </h3>
                        {activeSession?.id === session.id && (
                          <span className="text-xs bg-primary-100 text-primary-800 px-2 py-1 rounded-full">
                            Active
                          </span>
                        )}
                      </div>
                    )}
                    
                    <div className="mt-1 text-xs text-gray-500">
                      <div className="flex items-center space-x-2">
                        <span>{session.message_count} messages</span>
                        <span>•</span>
                        <span>{formatLastActivity(session.last_activity)}</span>
                      </div>
                      {session.last_message && (
                        <p className="mt-1 truncate text-gray-600">
                          {session.last_message}
                        </p>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-1 ml-2">
                    {editingSessionId === session.id ? (
                      <>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleSaveEdit();
                          }}
                          className="p-1 text-green-600 hover:text-green-700"
                        >
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCancelEdit();
                          }}
                          className="p-1 text-gray-600 hover:text-gray-700"
                        >
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                          </svg>
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStartEditing(session.id, session.name);
                          }}
                          className="p-1 text-gray-400 hover:text-gray-600"
                        >
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
                          </svg>
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteSession(session.id);
                          }}
                          className="p-1 text-gray-400 hover:text-red-600"
                        >
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" clipRule="evenodd" />
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                          </svg>
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
