"""
Enhanced memory manager with LangChain integration as primary system.
This memory manager now uses LangChain by default with fallback to manual processing.
"""

import os
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from .cosmos_chat_history import CosmosDBNoSQLChatMessageHistory
from .cosmos_store import cosmos_store
from .redis_cache import redis_cache

class MemoryManager:
    """Enhanced Memory Manager with LangChain integration."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
        return cls._instance

    async def initialize(self):
        """Initialize LangChain memory stores."""
        cosmos_store.initialize()
        await redis_cache.initialize()
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_MINI", "gpt-4o-mini"), temperature=0)
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Condense the above chat history into a concise summary in Spanish, retaining all key information and context. This summary will be used to provide context for future conversations. If the conversation is short, just return the conversation as is.")
        ])
        self.enable_summarization = os.getenv("ENABLE_CONVERSATION_SUMMARIZATION", "true").lower() == "true"
        self.redis_ttl = int(os.getenv("REDIS_SESSION_TTL", 86400)) # 24 hours
        
        # Memory configurations
        self.window_size = int(os.getenv("MEMORY_WINDOW_SIZE", "10"))
        self.summary_threshold = int(os.getenv("MEMORY_SUMMARY_THRESHOLD", "20"))
        
        print("[MEMORY] Initialized Enhanced Memory Manager with LangChain.")

    def get_session_memory(self, session_id: str, memory_type: str = "hybrid"):
        """Get LangChain memory instance for a session."""
        
        if memory_type == "buffer":
            return ConversationBufferWindowMemory(
                k=self.window_size,
                return_messages=True,
                memory_key="chat_history"
            )
        
        elif memory_type == "summary":
            return ConversationSummaryMemory(
                llm=self.llm,
                return_messages=True,
                memory_key="chat_history"
            )
        
        elif memory_type == "redis":
            return RedisChatMessageHistory(
                session_id=session_id,
                url=f"rediss://:{os.getenv('AZURE_REDIS_PRIMARY_KEY')}@{os.getenv('AZURE_REDIS_HOST')}:{os.getenv('AZURE_REDIS_PORT')}/0",
                ttl=self.redis_ttl
            )
        
        elif memory_type == "cosmos":
            return CosmosDBNoSQLChatMessageHistory(
                cosmos_client=cosmos_store,
                session_id=session_id,
                user_id="default_user"
            )
        
        else:  # hybrid
            return self._create_hybrid_memory(session_id)
    
    def _create_hybrid_memory(self, session_id: str):
        """Create hybrid memory combining Redis and Cosmos DB."""
        
        class HybridMemory:
            def __init__(self, session_id: str, redis_cache, cosmos_store):
                self.session_id = session_id
                self.redis_cache = redis_cache
                self.cosmos_store = cosmos_store
                self._redis_memory = None
                self._cosmos_memory = None
                self._initialize_memories()
            
            def _initialize_memories(self):
                try:
                    # Try Redis first
                    self._redis_memory = RedisChatMessageHistory(
                        session_id=self.session_id,
                        url=f"rediss://:{os.getenv('AZURE_REDIS_PRIMARY_KEY')}@{os.getenv('AZURE_REDIS_HOST')}:{os.getenv('AZURE_REDIS_PORT')}/0",
                        ttl=int(os.getenv("REDIS_SESSION_TTL", 86400))
                    )
                except Exception as e:
                    print(f"[MEMORY] Redis memory not available: {e}")
                    self._redis_memory = None
                
                # Always initialize Cosmos DB
                try:
                    self._cosmos_memory = CosmosDBNoSQLChatMessageHistory(
                        cosmos_client=cosmos_store,
                        session_id=self.session_id,
                        user_id="default_user"
                    )
                except Exception as e:
                    print(f"[MEMORY] Cosmos DB memory not available: {e}")
                    self._cosmos_memory = None
            
            @property
            def messages(self) -> List[BaseMessage]:
                """Get messages from Redis first, fallback to Cosmos DB."""
                if self._redis_memory and self._redis_memory.messages:
                    return self._redis_memory.messages
                elif self._cosmos_memory:
                    return self._cosmos_memory.messages
                return []
            
            def add_user_message(self, message: str):
                """Add user message to both stores."""
                if self._redis_memory:
                    try:
                        self._redis_memory.add_user_message(message)
                    except Exception as e:
                        print(f"[MEMORY] Error adding to Redis: {e}")
                
                if self._cosmos_memory:
                    try:
                        self._cosmos_memory.add_user_message(message)
                    except Exception as e:
                        print(f"[MEMORY] Error adding to Cosmos DB: {e}")
            
            def add_ai_message(self, message: str):
                """Add AI message to both stores."""
                if self._redis_memory:
                    try:
                        self._redis_memory.add_ai_message(message)
                    except Exception as e:
                        print(f"[MEMORY] Error adding to Redis: {e}")
                
                if self._cosmos_memory:
                    try:
                        self._cosmos_memory.add_ai_message(message)
                    except Exception as e:
                        print(f"[MEMORY] Error adding to Cosmos DB: {e}")
            
            def clear(self):
                """Clear both stores."""
                if self._redis_memory:
                    try:
                        self._redis_memory.clear()
                    except Exception as e:
                        print(f"[MEMORY] Error clearing Redis: {e}")
                
                if self._cosmos_memory:
                    try:
                        self._cosmos_memory.clear()
                    except Exception as e:
                        print(f"[MEMORY] Error clearing Cosmos DB: {e}")
        
        return HybridMemory(session_id, redis_cache, cosmos_store)

    async def get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        """Get conversation history using LangChain memory."""
        try:
            memory = self.get_session_memory(session_id, "hybrid")
            messages = memory.messages
            
            # If too many messages, use summary memory
            if len(messages) > self.summary_threshold:
                summary_memory = self.get_session_memory(session_id, "summary")
                summary_memory.chat_memory.messages = messages
                return summary_memory.chat_memory.messages
            
            print(f"[MEMORY] Retrieved history for session {session_id} using LangChain.")
            return messages
            
        except Exception as e:
            print(f"[MEMORY] Error getting conversation history: {e}")
            return []

    async def add_message_to_history(self, session_id: str, human_message: str, ai_message: str):
        """Add messages to history using LangChain memory."""
        try:
            memory = self.get_session_memory(session_id, "hybrid")
            memory.add_user_message(human_message)
            memory.add_ai_message(ai_message)
            
            print(f"[MEMORY] Added messages to LangChain memory for session {session_id}.")
            
        except Exception as e:
            print(f"[MEMORY] Error adding messages to history: {e}")

    async def _summarize_conversation(self, session_id: str, messages: List[BaseMessage]):
        """Summarize conversation using LangChain."""
        if self.enable_summarization and len(messages) > 10:
            print(f"[MEMORY] Summarizing conversation for session {session_id} using LangChain...")
            try:
                summarization_chain = self.summarization_prompt | self.llm
                summary = await summarization_chain.invoke({"chat_history": messages})
                print(f"[MEMORY] Summary for session {session_id}: {summary.content[:200]}...")
            except Exception as e:
                print(f"[MEMORY] Error during summarization for session {session_id}: {e}")

    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "type": "enhanced_langchain_memory",
            "memory_types": ["buffer", "summary", "redis", "cosmos", "hybrid"],
            "window_size": self.window_size,
            "summary_threshold": self.summary_threshold,
            "redis_status": "unavailable",
            "cosmos_status": "unavailable",
            "enable_summarization": self.enable_summarization
        }
        
        # Check Redis status
        try:
            if redis_cache.get_client():
                redis_cache.get_client().ping()
                stats["redis_status"] = "connected"
        except Exception:
            stats["redis_status"] = "error"
        
        # Check Cosmos DB status
        try:
            if cosmos_store._client:
                list(cosmos_store._container.query_items(
                    query="SELECT * FROM c OFFSET 0 LIMIT 1",
                    enable_cross_partition_query=True
                ))
                stats["cosmos_status"] = "connected"
        except Exception:
            stats["cosmos_status"] = "error"
        
        return stats

memory_manager = MemoryManager()