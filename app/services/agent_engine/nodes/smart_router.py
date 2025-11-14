from typing import Dict, Any


# Patterns obvios para fast-path (hard-coded)
OBVIOUS_PATTERNS = {
    'greeting': ['hola', 'buenos días', 'buenas tardes', 'buenas noches', 'hey', 'hi', 'buenas'],
    'farewell': ['adiós', 'adios', 'chao', 'chau', 'hasta luego', 'bye', 'nos vemos'],
    'thanks': ['gracias', 'thank', 'thanks', 'grazie', 'muchas gracias'],
    'request_human': ['hablar con', 'persona', 'humano', 'agente', 'operador', 'asesor']
}


def smart_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Smart router con fast-path para patterns obvios.
    
    Detecta mensajes simples (saludos, despedidas, gracias, request humano)
    y los procesa sin pasar por el orchestrator completo.
    
    Objetivo: 40% de mensajes usan fast-path, ahorrando ~200 tokens/mensaje.
    """
    # Obtener mensajes del usuario
    human_messages = [m for m in state['messages'] if m.type == 'human']
    
    if not human_messages:
        return {
            'use_full_orchestrator': True,
            'nodes_visited': state.get('nodes_visited', []) + ['smart_router']
        }
    
    last_user_message = human_messages[-1]
    is_first_message = len(human_messages) == 1
    message_content = last_user_message.content.lower()
    
    # Detectar patterns obvios
    detected_intent = None
    for intent_type, keywords in OBVIOUS_PATTERNS.items():
        if any(kw in message_content for kw in keywords):
            detected_intent = intent_type
            break
    
    # Si no detectamos pattern obvio → usar orchestrator completo
    if detected_intent is None:
        print("🔀 [SMART_ROUTER] No fast-path detected → using full orchestrator")
        return {
            'use_full_orchestrator': True,
            'nodes_visited': state.get('nodes_visited', []) + ['smart_router']
        }
    
    # Fast-path detectado
    print(f"⚡ [SMART_ROUTER] Fast-path: {detected_intent} (skipping orchestrator)")
    
    # Configuración según el pattern detectado
    if detected_intent == 'greeting':
        return {
            'use_full_orchestrator': False,
            'intent': 'greeting',
            'confidence': 0.95,
            'needs_knowledge_base': False,
            'kb_search_strategy': 'none',
            'search_queries': [],
            'complexity': 'simple',
            'should_handoff': False,
            'handoff_reason': None,
            'response_strategy': 'direct',
            'customer_sentiment': 'neutral',
            'orchestrator_reasoning': f'Fast-path: detected greeting pattern',
            'is_first_message': is_first_message,
            'routing_decision': 'fast_path_greeting',
            'nodes_visited': state.get('nodes_visited', []) + ['smart_router']
        }
    
    elif detected_intent == 'farewell':
        return {
            'use_full_orchestrator': False,
            'intent': 'other',
            'confidence': 0.95,
            'needs_knowledge_base': False,
            'kb_search_strategy': 'none',
            'search_queries': [],
            'complexity': 'simple',
            'should_handoff': False,
            'handoff_reason': None,
            'response_strategy': 'direct',
            'customer_sentiment': 'positive',
            'orchestrator_reasoning': f'Fast-path: detected farewell pattern',
            'is_first_message': is_first_message,
            'routing_decision': 'fast_path_farewell',
            'nodes_visited': state.get('nodes_visited', []) + ['smart_router']
        }
    
    elif detected_intent == 'thanks':
        return {
            'use_full_orchestrator': False,
            'intent': 'other',
            'confidence': 0.95,
            'needs_knowledge_base': False,
            'kb_search_strategy': 'none',
            'search_queries': [],
            'complexity': 'simple',
            'should_handoff': False,
            'handoff_reason': None,
            'response_strategy': 'direct',
            'customer_sentiment': 'positive',
            'orchestrator_reasoning': f'Fast-path: detected thanks pattern',
            'is_first_message': is_first_message,
            'routing_decision': 'fast_path_thanks',
            'nodes_visited': state.get('nodes_visited', []) + ['smart_router']
        }
    
    elif detected_intent == 'request_human':
        return {
            'use_full_orchestrator': False,
            'intent': 'request_human',
            'confidence': 0.95,
            'needs_knowledge_base': False,
            'kb_search_strategy': 'none',
            'search_queries': [],
            'complexity': 'simple',
            'should_handoff': True,
            'handoff_reason': 'Cliente solicitó explícitamente hablar con humano',
            'response_strategy': 'deflect',
            'customer_sentiment': 'neutral',
            'orchestrator_reasoning': f'Fast-path: detected request for human agent',
            'is_first_message': is_first_message,
            'routing_decision': 'fast_path_handoff',
            'nodes_visited': state.get('nodes_visited', []) + ['smart_router']
        }
    
    # Fallback (no debería llegar aquí)
    return {
        'use_full_orchestrator': True,
        'nodes_visited': state.get('nodes_visited', []) + ['smart_router']
    }

