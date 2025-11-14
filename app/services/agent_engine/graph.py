from typing import Dict, Any
from langgraph.graph import StateGraph, END
from app.services.agent_engine.state import AgentState
from app.services.agent_engine.nodes.greet import greet_node
from app.services.agent_engine.nodes.smart_router import smart_router_node
from app.services.agent_engine.nodes.orchestrator import orchestrator_node
from app.services.agent_engine.nodes.retrieve_knowledge import retrieve_knowledge_node
# call_tools_node será usado en Sprint 2+ cuando se implementen herramientas dinámicas
from app.services.agent_engine.nodes.respond import respond_node
from app.services.agent_engine.nodes.handoff import handoff_node


def route_after_smart_router(state: Dict[str, Any]) -> str:
    """
    Router después del smart_router.
    
    Si fast-path detectado → responder directamente
    Si no → pasar a orchestrator completo
    """
    use_full_orchestrator = state.get('use_full_orchestrator', True)
    
    if not use_full_orchestrator:
        print("🔀 [ROUTER] Fast-path detected → direct_respond")
        return 'direct_respond'
    
    print("🔀 [ROUTER] No fast-path → orchestrator")
    return 'orchestrator'


def route_after_orchestrator(state: Dict[str, Any]) -> str:
    """
    Router condicional después del orchestrator.
    
    Prioridades:
    1. Handoff forzado (confidence < 0.4 o should_handoff)
    2. Handoff sugerido (0.4 <= confidence < 0.6) - set flag, continuar
    3. Primer mensaje → greet
    4. Necesita KB → retrieve_knowledge
    5. Default → respuesta directa
    """
    confidence = state.get('confidence', 0.5)
    should_handoff = state.get('should_handoff', False)
    is_first_message = state.get('is_first_message', False)
    needs_kb = state.get('needs_knowledge_base', False)
    
    # Prioridad 1: Handoff explícito o muy baja confianza
    if should_handoff or confidence < 0.4:
        print(f"🔀 [ROUTER] force_handoff (confidence={confidence:.2f})")
        return 'force_handoff'
    
    # Prioridad 2: Confianza baja-media → sugerir handoff en respuesta
    if 0.4 <= confidence < 0.6:
        print(f"🔀 [ROUTER] suggest_handoff (confidence={confidence:.2f})")
        state['suggest_handoff_in_response'] = True
        # Continúa a respond pero con flag para agregar disclaimer
    
    # Prioridad 3: Primer mensaje
    if is_first_message:
        print(f"🔀 [ROUTER] greet (first message)")
        return 'greet'
    
    # Prioridad 4: Necesita KB
    if needs_kb:
        print(f"🔀 [ROUTER] retrieve_knowledge (confidence={confidence:.2f})")
        return 'retrieve_knowledge'
    
    # Default: respuesta directa
    print(f"🔀 [ROUTER] direct_respond (confidence={confidence:.2f})")
    return 'direct_respond'


def create_agent_graph():
    """
    Crear y compilar el grafo del agente con LangGraph.
    
    Flujo optimizado (Sprint 1):
    START → smart_router → [conditional]
      ├─ fast_path (40%) → respond → END
      └─ full (60%) → orchestrator → [conditional routing] → END
    
    Routing condicional desde orchestrator:
    - Si confidence < 0.4 → force_handoff → END
    - Si 0.4 <= confidence < 0.6 → suggest_handoff (set flag, continuar)
    - Si is_first_message → greet → respond → END
    - Si needs_knowledge_base → retrieve_knowledge → respond → END
    - Else → respond → END
    """
    workflow = StateGraph(AgentState)
    
    # Agregar nodos
    workflow.add_node("smart_router", smart_router_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("greet", greet_node)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
    # call_tools no se agrega porque no se usa en Sprint 1 (será para Sprint 2+)
    workflow.add_node("respond", respond_node)
    workflow.add_node("handoff", handoff_node)
    
    # ✅ Entry point: smart_router (detecta fast-paths primero)
    workflow.set_entry_point("smart_router")
    
    # ✅ Routing desde smart_router
    workflow.add_conditional_edges(
        "smart_router",
        route_after_smart_router,
        {
            "direct_respond": "respond",  # Fast-path
            "orchestrator": "orchestrator"  # Full analysis
        }
    )
    
    # ✅ Routing condicional desde orchestrator
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "force_handoff": "handoff",
            "greet": "greet",
            "retrieve_knowledge": "retrieve_knowledge",
            "direct_respond": "respond"
        }
    )
    
    # ✅ Greet siempre va a respond después
    workflow.add_edge("greet", "respond")
    
    # ✅ Retrieve knowledge va a respond
    workflow.add_edge("retrieve_knowledge", "respond")
    
    # ✅ Respond y handoff terminan
    workflow.add_edge("respond", END)
    workflow.add_edge("handoff", END)
    
    return workflow.compile()
