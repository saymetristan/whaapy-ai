from typing import Optional, List, Annotated
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
    sentiment: Optional[str]  # positive, neutral, negative
    should_handoff: bool
    handoff_reason: Optional[str]
    
    # Knowledge base
    retrieved_docs: Optional[List[str]]
    
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
        retrieved_docs=None,
        nodes_visited=[],
        tools_used=[],
        execution_id=execution_id,
        started_at=datetime.now()
    )

