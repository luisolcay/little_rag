# RAG Frontend

A simple React frontend for the RAG (Retrieval-Augmented Generation) system.

## Features

- **Chat Interface**: Clean, minimal chat UI for asking questions
- **Citations**: Display document sources with page numbers and relevance scores
- **Session Management**: Create, switch, and manage multiple conversation sessions
- **Real-time Responses**: Get answers from the LLM with source citations
- **Error Handling**: Graceful error messages and loading states

## Quick Setup

Run the setup script to get started quickly:

```bash
./setup.sh
```

Or follow the manual setup below.

## Manual Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure environment**:
   Copy the example environment file and update as needed:
   ```bash
   cp env.example .env
   ```
   
   The default configuration expects the backend at `http://localhost:8000`

3. **Start the development server**:
   ```bash
   npm start
   ```

4. **Make sure the backend is running**:
   The frontend expects the FastAPI backend to be running on `http://localhost:8000`

## Usage

1. **Create a Session**: Click "New Session" to create a new conversation
2. **Ask Questions**: Type your questions in the input field and press Enter
3. **View Sources**: Each response includes citations showing which documents were used
4. **Manage Sessions**: Switch between sessions, rename them, or delete old ones

## API Integration

The frontend connects to these backend endpoints:

- `POST /llm/chat` - Send messages and receive responses
- `GET /llm/conversation/{session_id}` - Retrieve conversation history
- `DELETE /llm/conversation/{session_id}` - Clear conversation

## Technology Stack

- **React** with TypeScript
- **Tailwind CSS** for styling
- **Axios** for API calls
- **React Context** for state management
- **localStorage** for session persistence

## Project Structure

```
src/
├── components/          # React components
│   ├── ChatInterface.tsx    # Main chat UI
│   ├── MessageList.tsx      # Display messages
│   ├── MessageInput.tsx     # Input field
│   ├── Citation.tsx         # Individual citation
│   ├── CitationList.tsx     # List of citations
│   ├── SessionManager.tsx   # Session management
│   └── LoadingIndicator.tsx # Loading states
├── contexts/            # React contexts
│   └── SessionContext.tsx   # Session state management
├── services/            # API services
│   └── api.ts              # Axios client
├── types/               # TypeScript interfaces
│   └── index.ts            # Type definitions
└── App.tsx              # Main app component
```