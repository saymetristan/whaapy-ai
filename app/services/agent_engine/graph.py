from typing import Dict, Any
from langgraph.graph import StateGraph, END
from app.services.agent_engine.state import AgentState
from app.services.agent_engine.nodes.greet import greet_node
from app.services.agent_engine.nodes.analyze_intent import analyze_intent_node
from app.services.agent_engine.nodes.retrieve_knowledge import retrieve_knowledge_node
from app.services.agent_engine.nodes.call_tools import call_tools_node
from app.services.agent_engine.nodes.respond import respond_node
from app.services.agent_engine.nodes.handoff import handoff_node


def route_after_analysis(state: Dict[str, Any]) -> str:
    """
    Router condicional después del análisis de intención.
    
    Decide el siguiente nodo según el análisis:
    1. Si should_handoff → handoff (y termina)
    2. Si is_first_message → greet (luego respond)
    3. Si needs_knowledge → retrieve_knowledge (luego respond)
    4. Si needs_tools → call_tools (luego respond)
    5. Sino → respond directo
    """
    # Prioridad 1: Handoff (termina aquí)
    if state.get('should_handoff'):
        print("🔀 Routing: handoff")
        return 'handoff'
    
    # Prioridad 2: Primer mensaje → saludar
    if state.get('is_first_message'):
        print("🔀 Routing: greet (primer mensaje)")
        return 'greet'
    
    # Prioridad 3: Necesita conocimiento
    if state.get('needs_knowledge'):
        print("🔀 Routing: retrieve_knowledge")
        return 'retrieve_knowledge'
    
    # Prioridad 4: Necesita herramientas (stub por ahora)
    # En Fase 2 esto será dinámico
    
    # Default: responder directamente
    print("🔀 Routing: respond")
    return 'respond'


def create_agent_graph():
    """
    Crear y compilar el grafo del agente con LangGraph.
    
    Flujo optimizado:
    START → analyze_intent → [conditional routing] → END
    
    Routing condicional:
    - Si should_handoff → handoff → END
    - Si is_first_message → greet → respond → END
    - Si needs_knowledge → retrieve_knowledge → respond → END
    - Sino → respond → END
    """
    workflow = StateGraph(AgentState)
    
    # Agregar nodos
    workflow.add_node("analyze_intent", analyze_intent_node)
    workflow.add_node("greet", greet_node)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
    workflow.add_node("call_tools", call_tools_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("handoff", handoff_node)
    
    # ✅ Entry point: analyze_intent (SIEMPRE primero)
    workflow.set_entry_point("analyze_intent")
    
    # ✅ Routing condicional desde analyze_intent
    workflow.add_conditional_edges(
        "analyze_intent",
        route_after_analysis,
        {
            "handoff": "handoff",
            "greet": "greet",
            "retrieve_knowledge": "retrieve_knowledge",
            "call_tools": "call_tools",
            "respond": "respond"
        }
    )
    
    # ✅ Greet siempre va a respond después
    workflow.add_edge("greet", "respond")
    
    # ✅ Retrieve knowledge va a respond
    workflow.add_edge("retrieve_knowledge", "respond")
    
    # ✅ Call tools va a respond
    workflow.add_edge("call_tools", "respond")
    
    # ✅ Respond y handoff terminan
    workflow.add_edge("respond", END)
    workflow.add_edge("handoff", END)
    
    return workflow.compile()
