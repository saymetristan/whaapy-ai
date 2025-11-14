from typing import Dict, Any
from langgraph.graph import StateGraph, END
from app.services.agent_engine.state import AgentState
from app.services.agent_engine.nodes.greet import greet_node
from app.services.agent_engine.nodes.smart_router import smart_router_node
from app.services.agent_engine.nodes.orchestrator import orchestrator_node
from app.services.agent_engine.nodes.optimized_rag import optimized_rag_node
# call_tools_node será usado en Sprint 3+ cuando se implementen herramientas dinámicas
from app.services.agent_engine.nodes.respond import respond_node
from app.services.agent_engine.nodes.handoff import handoff_node
from app.services.agent_engine.nodes.validate_response import validate_response_node, retry_respond_node


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
    3. Necesita KB → retrieve_knowledge (ANTES de greet para primer mensaje)
    4. Primer mensaje sin KB → greet
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
    
    # Prioridad 3: Necesita KB (incluso en primer mensaje)
    if needs_kb:
        print(f"🔀 [ROUTER] optimized_rag (confidence={confidence:.2f}, first_msg={is_first_message})")
        return 'optimized_rag'
    
    # Prioridad 4: Primer mensaje sin necesidad de KB → greet simple
    if is_first_message:
        print(f"🔀 [ROUTER] greet (first message, no KB needed)")
        return 'greet'
    
    # Default: respuesta directa
    print(f"🔀 [ROUTER] direct_respond (confidence={confidence:.2f})")
    return 'direct_respond'


def route_after_respond(state: Dict[str, Any]) -> str:
    """
    Sprint 3: Router condicional después de generar respuesta.
    
    Decide si validar la respuesta o terminar:
    - confidence >= 0.75 → skip validation (ahorro tokens) → END
    - confidence < 0.75 → validar calidad → validate_response
    
    Optimización: Solo validamos respuestas con confianza media-baja.
    High confidence (>0.75) = skip validation = ahorro ~$0.0001 por mensaje.
    """
    confidence = state.get('confidence', 1.0)
    
    # High confidence → skip validation
    if confidence >= 0.75:
        print(f"🔀 [ROUTER] High confidence ({confidence:.2f}) → skip validation → END")
        return END
    
    # Low-medium confidence → validate
    print(f"🔀 [ROUTER] Low-medium confidence ({confidence:.2f}) → validate_response")
    return 'validate_response'


def route_after_validation(state: Dict[str, Any]) -> str:
    """
    Sprint 3: Router después de validation.
    
    Decide si hacer retry o terminar:
    - Si passed → END
    - Si was_retried → END (máximo 1 retry, evitar loops)
    - Si failed y no retried → retry_respond
    """
    passed = state.get('validation_passed', True)
    was_retried = state.get('was_retried', False)
    quality_score = state.get('quality_score', 0.0)
    
    # Si ya hicimos retry, terminar (evitar loops infinitos)
    if was_retried:
        print(f"🔀 [ROUTER] Already retried → END (quality={quality_score:.2f})")
        return END
    
    # Si pasó validation, terminar
    if passed:
        print(f"🔀 [ROUTER] Validation passed (quality={quality_score:.2f}) → END")
        return END
    
    # Si falló y no hemos reintentado, hacer retry
    print(f"🔀 [ROUTER] Validation failed (quality={quality_score:.2f}) → retry_respond")
    return 'retry_respond'


def create_agent_graph():
    """
    Crear y compilar el grafo del agente con LangGraph.
    
    Flujo optimizado (Sprint 3):
    START → smart_router → [conditional]
      ├─ fast_path (40%) → respond → [conditional validation] → END
      └─ full (60%) → orchestrator → [conditional routing] → respond → [conditional validation] → END
    
    Sprint 3 - Validation condicional:
    respond → route_after_respond:
      - confidence >= 0.75 → END (skip validation, ahorro tokens)
      - confidence < 0.75 → validate_response → route_after_validation:
        - passed → END
        - failed + not retried → retry_respond → END
        - failed + already retried → END (evitar loops)
    """
    workflow = StateGraph(AgentState)
    
    # Agregar nodos
    workflow.add_node("smart_router", smart_router_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("greet", greet_node)
    workflow.add_node("optimized_rag", optimized_rag_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("handoff", handoff_node)
    
    # Sprint 3: Agregar nodos de validation y retry
    workflow.add_node("validate_response", validate_response_node)
    workflow.add_node("retry_respond", retry_respond_node)
    
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
            "optimized_rag": "optimized_rag",
            "direct_respond": "respond"
        }
    )
    
    # ✅ Greet siempre va a respond después
    workflow.add_edge("greet", "respond")
    
    # ✅ Optimized RAG va a respond
    workflow.add_edge("optimized_rag", "respond")
    
    # ✅ Sprint 3: Respond → routing condicional (validar o terminar)
    workflow.add_conditional_edges(
        "respond",
        route_after_respond,
        {
            END: END,
            "validate_response": "validate_response"
        }
    )
    
    # ✅ Sprint 3: Validation → routing (retry o terminar)
    workflow.add_conditional_edges(
        "validate_response",
        route_after_validation,
        {
            END: END,
            "retry_respond": "retry_respond"
        }
    )
    
    # ✅ Sprint 3: Retry siempre termina (no re-valida, evitar loops)
    workflow.add_edge("retry_respond", END)
    
    # ✅ Handoff termina
    workflow.add_edge("handoff", END)
    
    return workflow.compile()
