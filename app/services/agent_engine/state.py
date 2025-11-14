from typing import Optional, List, Annotated, Dict, Any
from datetime import datetime
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(dict):
    """Estado del agente conversacional con LangGraph"""
    
    # Mensajes de la conversación (con reducer especial)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Información del cliente
    customer_phone: str
    customer_name: Optional[str]
    
    # Contexto de negocio
    business_id: str
    conversation_id: str
    
    # Estado de la conversación
    intent: Optional[str]
    sentiment: Optional[str]  # positive, neutral, negative (deprecated, usar customer_sentiment)
    should_handoff: bool
    handoff_reason: Optional[str]
    is_first_message: bool  # Nuevo: para detectar primer mensaje
    needs_knowledge: bool   # Nuevo: si necesita buscar en KB (deprecated, usar needs_knowledge_base)
    
    # Sprint 1: Orchestrator fields
    confidence: Optional[float]  # 0.0-1.0
    needs_knowledge_base: Optional[bool]
    kb_search_strategy: Optional[str]  # exact|broad|multi_query|none
    search_queries: Optional[List[str]]
    complexity: Optional[str]  # simple|medium|complex
    response_strategy: Optional[str]  # direct|with_context|multi_step|deflect
    customer_sentiment: Optional[str]  # very_positive|positive|neutral|negative|very_negative
    orchestrator_reasoning: Optional[str]
    routing_decision: Optional[str]  # force_handoff|suggest_handoff|greet|retrieve_knowledge|direct_respond
    suggest_handoff_in_response: Optional[bool]  # Flag para disclaimers
    use_full_orchestrator: Optional[bool]  # Flag del smart_router
    conversation_summary: Optional[Dict[str, Any]]  # Summary para Sprint 3
    
    # Knowledge base
    retrieved_docs: Optional[str]
    
    # Tracking
    nodes_visited: List[str]
    tools_used: List[str]
    
    # Metadata
    execution_id: str
    started_at: datetime


def create_initial_state(
    business_id: str,
    conversation_id: str,
    customer_phone: str,
    execution_id: str,
    message: BaseMessage,
    customer_name: Optional[str] = None
) -> AgentState:
    """Crear estado inicial del agente"""
    return AgentState(
        messages=[message],
        customer_phone=customer_phone,
        customer_name=customer_name,
        business_id=business_id,
        conversation_id=conversation_id,
        intent=None,
        sentiment=None,
        should_handoff=False,
        handoff_reason=None,
        is_first_message=False,  # Lo determina smart_router/orchestrator
        needs_knowledge=False,   # Lo determina smart_router/orchestrator
        # Sprint 1: Orchestrator fields
        confidence=None,
        needs_knowledge_base=None,
        kb_search_strategy=None,
        search_queries=None,
        complexity=None,
        response_strategy=None,
        customer_sentiment=None,
        orchestrator_reasoning=None,
        routing_decision=None,
        suggest_handoff_in_response=False,
        use_full_orchestrator=None,
        conversation_summary=None,
        retrieved_docs=None,
        nodes_visited=[],
        tools_used=[],
        execution_id=execution_id,
        started_at=datetime.now()
    )

