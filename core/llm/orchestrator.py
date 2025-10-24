"""
Enhanced LLM orchestrator with LangChain integration as primary system.
This orchestrator now uses LangChain by default with fallback to manual processing.
"""

import os
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .models import LLMRequest, LLMResponse, ModelName, EnvironmentalQueryType, Citation
from .langchain_outputs import langchain_structured_outputs
from .cost_tracking import CostTracker
from core.memory.memory_manager import memory_manager

logging.basicConfig(filename='app.log', level=logging.INFO)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service is back

class CircuitBreaker:
    """Circuit breaker for LLM requests."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"Circuit breaker opened after {self.failure_count} failures")

class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)

class LLMOrchestrator:
    """Enhanced LLM Orchestrator with LangChain integration."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMOrchestrator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.llms = self._initialize_llms()
            self.cost_tracker = CostTracker()
            self._setup_chains()
            
            # Circuit breaker and retry configuration
            self.circuit_breakers = {
                model: CircuitBreaker(failure_threshold=5, timeout=60)
                for model in self.llms.keys()
            }
            self.retry_config = RetryConfig(max_retries=3, base_delay=1.0, max_delay=60.0)
            
            # Performance metrics
            self.metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "circuit_breaker_trips": 0,
                "retry_attempts": 0
            }
            
            self._initialized = True
            print("[LLM_ORCHESTRATOR] Initialized Enhanced LLM Orchestrator with LangChain, Circuit Breaker, and Retry Logic.")

    def _initialize_llms(self) -> Dict[ModelName, AzureChatOpenAI]:
        """Initialize Azure Chat OpenAI LLMs."""
        llms = {}
        default_temperature = float(os.getenv("LLM_DEFAULT_TEMPERATURE", 0.1))
        
        print(f"[LLM_ORCHESTRATOR] Initializing Azure Chat OpenAI LLMs")
        
        # Initialize GPT-4o
        gpt4o_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
        if gpt4o_deployment:
            llms[ModelName.GPT4_O] = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                deployment_name=gpt4o_deployment,
                temperature=default_temperature,
                streaming=os.getenv("ENABLE_STREAMING", "true").lower() == "true"
            )
            print(f"[LLM_ORCHESTRATOR] Initialized GPT-4o with deployment: {gpt4o_deployment}")

        # Initialize GPT-4o-mini
        gpt4omini_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_MINI", "gpt-4o-mini")
        if gpt4omini_deployment:
            llms[ModelName.GPT4_O_MINI] = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                deployment_name=gpt4omini_deployment,
                temperature=default_temperature,
                streaming=os.getenv("ENABLE_STREAMING", "true").lower() == "true"
            )
            print(f"[LLM_ORCHESTRATOR] Initialized GPT-4o-mini with deployment: {gpt4omini_deployment}")
        
        if not llms:
            logging.error("No LLM deployments found. Please check AZURE_OPENAI_CHAT_DEPLOYMENT and AZURE_OPENAI_CHAT_DEPLOYMENT_MINI environment variables.")
            raise ValueError("No LLM deployments configured.")
        
        return llms

    def _setup_chains(self):
        """Setup LangChain chains."""
        
        # Contextualization prompt
        self.contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        # Environmental consultant system prompt
        self.environmental_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert environmental consultant specializing in projects in the north of Chile. Your role is to provide accurate, concise, and actionable advice based on the provided context.

Context: {context}

Guidelines:
- Always prioritize information from the context
- If context doesn't contain the answer, state that you don't have enough information
- Focus on Chilean environmental regulations (SEA, SMA, DGA, etc.)
- Provide specific, actionable recommendations
- Consider both technical and regulatory aspects
- Use professional language appropriate for environmental consulting"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

    async def _execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic and circuit breaker."""
        model = kwargs.get('model') or args[0].model if args else None
        circuit_breaker = self.circuit_breakers.get(model)
        
        if not circuit_breaker or not circuit_breaker.can_execute():
            self.metrics["circuit_breaker_trips"] += 1
            raise Exception(f"Circuit breaker is open for model {model}")
        
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                circuit_breaker.record_success()
                self.metrics["successful_requests"] += 1
                return result
                
            except Exception as e:
                last_exception = e
                circuit_breaker.record_failure()
                
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    self.metrics["retry_attempts"] += 1
                    print(f"Attempt {attempt + 1} failed for model {model}, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    print(f"All retry attempts failed for model {model}: {e}")
        
        self.metrics["failed_requests"] += 1
        raise last_exception

    def _get_llm(self, model_name: ModelName) -> AzureChatOpenAI:
        """Get LLM instance."""
        llm = self.llms.get(model_name)
        if not llm:
            raise ValueError(f"LLM for model {model_name.value} not configured.")
        return llm

    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process LLM request using LangChain with retry logic and circuit breaker."""
        self.metrics["total_requests"] += 1
        
        try:
            return await self._execute_with_retry(self._process_request_internal, request)
        except Exception as e:
            logging.error(f"Error processing LLM request for session {request.session_id}: {e}")
            raise
    
    async def _process_request_internal(self, request: LLMRequest) -> LLMResponse:
        """Internal method for processing LLM request."""
        start_time = datetime.now()
        llm = self._get_llm(request.model)
        
        try:
            # 1. Contextualize query if chat history exists
            contextualized_query = request.query
            chat_history = request.conversation_history or []
            if chat_history:
                contextualize_chain = self.contextualize_prompt | llm | StrOutputParser()
                contextualized_query = await contextualize_chain.ainvoke({
                    "chat_history": chat_history,
                    "input": request.query
                })
                logging.info(f"Contextualized query: {contextualized_query}")
            
            # 2. Prepare context
            context_str = "\n\n".join([chunk["content"] for chunk in request.context_chunks]) if request.context_chunks else "No additional context provided."
            
            # 3. Determine output format and execute
            if request.query_type == EnvironmentalQueryType.GENERAL_QA:
                response_content = await self._process_general_qa(
                    llm, contextualized_query, context_str, chat_history
                )
            
            elif request.query_type == EnvironmentalQueryType.STRUCTURED_COMPLIANCE:
                response_content = await self._process_structured_compliance(
                    contextualized_query, context_str
                )
            
            elif request.query_type == EnvironmentalQueryType.STRUCTURED_RISK_ASSESSMENT:
                response_content = await self._process_structured_risk_assessment(
                    contextualized_query, context_str
                )
            
            elif request.query_type == EnvironmentalQueryType.STRUCTURED_CITATION:
                response_content = await self._process_structured_citation(
                    contextualized_query, context_str
                )
            
            else:
                raise ValueError(f"Unsupported query type: {request.query_type}")
            
            # 4. Add to conversation history using LangChain memory (disabled for testing)
            try:
                await memory_manager.add_message_to_history(
                    request.session_id, request.query, str(response_content)
                )
            except Exception as e:
                logging.warning(f"Memory not available: {e}")
                # Continue without memory
            
            # 5. Track cost
            prompt_tokens = len(contextualized_query.split()) + len(context_str.split())
            if request.conversation_history:
                prompt_tokens += sum(len(m.content.split()) for m in request.conversation_history)
            completion_tokens = len(str(response_content).split())
            try:
                self.cost_tracker.track_usage(request.model, prompt_tokens, completion_tokens)
            except Exception as e:
                logging.warning(f"Cost tracking not available: {e}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                session_id=request.session_id,
                query=request.query,
                query_type=request.query_type,
                content=str(response_content),
                model=request.model,
                processing_time=processing_time,
                tokens_used=prompt_tokens + completion_tokens,
                metadata={
                    "contextualized_query": contextualized_query,
                    "query_type": request.query_type.value,
                    "langchain_processed": True
                }
            )
            
        except Exception as e:
            logging.error(f"Error in _process_request_internal: {e}")
            raise
    
    async def _process_general_qa(self, llm: AzureChatOpenAI, query: str, context: str, chat_history: List[BaseMessage]) -> str:
        """Process general QA using LangChain."""
        qa_chain = self.environmental_prompt | llm | StrOutputParser()
        
        return await qa_chain.ainvoke({
            "context": context,
            "chat_history": chat_history,
            "input": query
        })
    
    async def _process_structured_compliance(self, query: str, context: str) -> Any:
        """Process structured compliance using LangChain structured outputs."""
        return langchain_structured_outputs.generate_compliance_report(query, context)
    
    async def _process_structured_risk_assessment(self, query: str, context: str) -> Any:
        """Process structured risk assessment using LangChain structured outputs."""
        return langchain_structured_outputs.generate_risk_assessment(query, context)
    
    async def _process_structured_citation(self, query: str, context: str) -> Any:
        """Process structured citation using LangChain structured outputs."""
        # For now, return a simple citation
        from .citation_extractor import citation_extractor
        return Citation(
            chunk_id="langchain_generated",
            document_name="Generated Response",
            content_snippet=context[:200] + "..." if len(context) > 200 else context,
            relevance_score=0.9
        )
    
    async def stream_response(self, request: LLMRequest):
        """Stream response using LangChain streaming."""
        llm = self._get_llm(request.model)
        
        # Prepare context
        context_str = "\n\n".join([chunk["content"] for chunk in request.context_chunks]) if request.context_chunks else "No additional context provided."
        
        # Create streaming chain
        streaming_chain = self.environmental_prompt | llm
        
        # Stream response
        async for chunk in streaming_chain.astream({
            "context": context_str,
            "chat_history": request.conversation_history,
            "input": request.query
        }):
            if hasattr(chunk, 'content'):
                yield chunk.content
    
    async def get_orchestrator_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        # Calculate success rate
        total_requests = self.metrics["total_requests"]
        success_rate = (self.metrics["successful_requests"] / max(total_requests, 1)) * 100
        
        # Circuit breaker status
        circuit_status = {}
        for model, breaker in self.circuit_breakers.items():
            circuit_status[model.value] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure_time": breaker.last_failure_time
            }
        
        stats = {
            "type": "enhanced_langchain_orchestrator",
            "initialized": self._initialized,
            "available_models": [model.value for model in self.llms.keys()],
            "cost_tracking": self.cost_tracker.get_current_costs(),
            "memory_manager": await memory_manager.get_memory_statistics(),
            "performance_metrics": {
                "total_requests": total_requests,
                "successful_requests": self.metrics["successful_requests"],
                "failed_requests": self.metrics["failed_requests"],
                "success_rate": success_rate,
                "circuit_breaker_trips": self.metrics["circuit_breaker_trips"],
                "retry_attempts": self.metrics["retry_attempts"]
            },
            "circuit_breaker_status": circuit_status,
            "retry_config": {
                "max_retries": self.retry_config.max_retries,
                "base_delay": self.retry_config.base_delay,
                "max_delay": self.retry_config.max_delay
            }
        }
        return stats

llm_orchestrator = LLMOrchestrator()