import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { Session } from '../types';

interface SessionState {
  sessions: Session[];
  activeSessionId: string | null;
  isLoading: boolean;
  error: string | null;
}

type SessionAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'LOAD_SESSIONS'; payload: Session[] }
  | { type: 'CREATE_SESSION'; payload: Session }
  | { type: 'UPDATE_SESSION'; payload: Session }
  | { type: 'DELETE_SESSION'; payload: string }
  | { type: 'SET_ACTIVE_SESSION'; payload: string | null }
  | { type: 'UPDATE_LAST_MESSAGE'; payload: { sessionId: string; message: string } };

interface SessionContextType {
  state: SessionState;
  createSession: (name?: string) => Session;
  updateSession: (sessionId: string, updates: Partial<Session>) => void;
  deleteSession: (sessionId: string) => void;
  setActiveSession: (sessionId: string | null) => void;
  updateLastMessage: (sessionId: string, message: string) => void;
  getActiveSession: () => Session | null;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

const STORAGE_KEY = 'rag_sessions';

const initialState: SessionState = {
  sessions: [],
  activeSessionId: null,
  isLoading: false,
  error: null,
};

function sessionReducer(state: SessionState, action: SessionAction): SessionState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    
    case 'LOAD_SESSIONS':
      return { ...state, sessions: action.payload };
    
    case 'CREATE_SESSION':
      return { 
        ...state, 
        sessions: [...state.sessions, action.payload],
        activeSessionId: action.payload.id
      };
    
    case 'UPDATE_SESSION':
      return {
        ...state,
        sessions: state.sessions.map(session =>
          session.id === action.payload.id ? action.payload : session
        )
      };
    
    case 'DELETE_SESSION':
      const updatedSessions = state.sessions.filter(s => s.id !== action.payload);
      const newActiveSessionId = state.activeSessionId === action.payload 
        ? (updatedSessions.length > 0 ? updatedSessions[0].id : null)
        : state.activeSessionId;
      
      return {
        ...state,
        sessions: updatedSessions,
        activeSessionId: newActiveSessionId
      };
    
    case 'SET_ACTIVE_SESSION':
      return { ...state, activeSessionId: action.payload };
    
    case 'UPDATE_LAST_MESSAGE':
      return {
        ...state,
        sessions: state.sessions.map(session =>
          session.id === action.payload.sessionId
            ? {
                ...session,
                last_message: action.payload.message,
                last_activity: new Date().toISOString(),
                message_count: session.message_count + 1
              }
            : session
        )
      };
    
    default:
      return state;
  }
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(sessionReducer, initialState);

  // Load sessions from localStorage on mount
  useEffect(() => {
    try {
      const storedSessions = localStorage.getItem(STORAGE_KEY);
      if (storedSessions) {
        const sessions = JSON.parse(storedSessions);
        dispatch({ type: 'LOAD_SESSIONS', payload: sessions });
        
        // Set active session to the most recent one if none is active
        if (sessions.length > 0 && !state.activeSessionId) {
          const mostRecent = sessions.reduce((latest: Session, current: Session) =>
            new Date(current.last_activity) > new Date(latest.last_activity) ? current : latest
          );
          dispatch({ type: 'SET_ACTIVE_SESSION', payload: mostRecent.id });
        }
      }
    } catch (error) {
      console.error('Failed to load sessions from localStorage:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Failed to load sessions' });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Save sessions to localStorage whenever sessions change
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.sessions));
    } catch (error) {
      console.error('Failed to save sessions to localStorage:', error);
    }
  }, [state.sessions]);

  const createSession = (name?: string): Session => {
    const now = new Date().toISOString();
    const sessionName = name || `Session ${state.sessions.length + 1}`;
    
    const newSession: Session = {
      id: `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: sessionName,
      created_at: now,
      last_activity: now,
      message_count: 0,
    };

    dispatch({ type: 'CREATE_SESSION', payload: newSession });
    return newSession;
  };

  const updateSession = (sessionId: string, updates: Partial<Session>) => {
    const session = state.sessions.find(s => s.id === sessionId);
    if (session) {
      const updatedSession = { ...session, ...updates };
      dispatch({ type: 'UPDATE_SESSION', payload: updatedSession });
    }
  };

  const deleteSession = (sessionId: string) => {
    dispatch({ type: 'DELETE_SESSION', payload: sessionId });
  };

  const setActiveSession = (sessionId: string | null) => {
    dispatch({ type: 'SET_ACTIVE_SESSION', payload: sessionId });
  };

  const updateLastMessage = (sessionId: string, message: string) => {
    dispatch({ type: 'UPDATE_LAST_MESSAGE', payload: { sessionId, message } });
  };

  const getActiveSession = (): Session | null => {
    return state.sessions.find(s => s.id === state.activeSessionId) || null;
  };

  const value: SessionContextType = {
    state,
    createSession,
    updateSession,
    deleteSession,
    setActiveSession,
    updateLastMessage,
    getActiveSession,
  };

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const context = useContext(SessionContext);
  if (context === undefined) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
}
