from typing import Dict, Any
from langgraph.graph import StateGraph, END
from app.services.agent_engine.state import AgentState
from app.services.agent_engine.nodes.greet import greet_node
from app.services.agent_engine.nodes.analyze_intent import analyze_intent_node
from app.services.agent_engine.nodes.retrieve_knowledge import retrieve_knowledge_node
from app.services.agent_engine.nodes.call_tools import call_tools_node
from app.services.agent_engine.nodes.respond import respond_node
from app.services.agent_engine.nodes.handoff import handoff_node


def needs_knowledge(state: Dict[str, Any]) -> bool:
    """
    Determina si el mensaje requiere búsqueda en knowledge base.
    
    Usa heurística simple basada en palabras clave de preguntas.
    """
    # Obtener último mensaje del usuario
    human_messages = [m for m in state['messages'] if m.type == 'human']
    
    if not human_messages:
        return False
    
    last_message = human_messages[-1].content.lower()
    
    # Palabras clave que sugieren preguntas
    question_words = [
        'qué', 'que', 'cómo', 'como', 'cuándo', 'cuando', 
        'dónde', 'donde', 'por qué', 'porque', 'cuál', 'cual',
        'quién', 'quien', 'cuánto', 'cuanto'
    ]
    
    # Si contiene signos de pregunta o palabras clave
    has_question_mark = '?' in last_message
    has_question_word = any(word in last_message for word in question_words)
    
    return has_question_mark or has_question_word


def needs_tools(state: Dict[str, Any]) -> bool:
    """
    Determina si se necesita ejecutar tools/webhooks.
    
    Por ahora siempre retorna False (implementación en Fase 2).
    """
    # TODO Fase 2: Implementar lógica para detectar cuando ejecutar webhooks
    return False


def route_after_analysis(state: Dict[str, Any]) -> str:
    """
    Router condicional después del análisis de intención.
    
    Decide el siguiente nodo según el estado.
    """
    # Si debe transferir a humano
    if state.get('should_handoff'):
        return 'handoff'
    
    # Si necesita knowledge base
    if needs_knowledge(state):
        return 'retrieve_knowledge'
    
    # Si necesita ejecutar tools (siempre False por ahora)
    if needs_tools(state):
        return 'call_tools'
    
    # Responder directamente
    return 'respond'


def create_agent_graph():
    """
    Crear y compilar el grafo del agente con LangGraph.
    
    Flujo:
    START → greet → analyze_intent → [conditional routing] → END
    
    Routing condicional:
    - Si should_handoff → handoff → END
    - Si needs_knowledge → retrieve_knowledge → respond → END
    - Si needs_tools → call_tools → respond → END
    - Sino → respond → END
    """
    # Crear grafo con estado tipado
    workflow = StateGraph(AgentState)
    
    # Agregar nodos
    workflow.add_node("greet", greet_node)
    workflow.add_node("analyze_intent", analyze_intent_node)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
    workflow.add_node("call_tools", call_tools_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("handoff", handoff_node)
    
    # Entry point
    workflow.set_entry_point("greet")
    
    # Edges simples
    workflow.add_edge("greet", "analyze_intent")
    
    # Edge condicional desde analyze_intent
    workflow.add_conditional_edges(
        "analyze_intent",
        route_after_analysis,
        {
            "handoff": "handoff",
            "retrieve_knowledge": "retrieve_knowledge",
            "call_tools": "call_tools",
            "respond": "respond"
        }
    )
    
    # Después de retrieve_knowledge → respond
    workflow.add_edge("retrieve_knowledge", "respond")
    
    # Después de call_tools → respond
    workflow.add_edge("call_tools", "respond")
    
    # Respond y handoff terminan
    workflow.add_edge("respond", END)
    workflow.add_edge("handoff", END)
    
    # Compilar grafo
    return workflow.compile()

