import React from 'react';
import { SessionProvider } from './contexts/SessionContext';
import { ChatInterface } from './components/ChatInterface';
import './App.css';

function App() {
  return (
    <SessionProvider>
      <div className="App h-screen bg-gray-50">
        <ChatInterface />
      </div>
    </SessionProvider>
  );
}

export default App;