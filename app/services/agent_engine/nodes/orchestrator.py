import json
from typing import Dict, Any, List, Optional
from app.services.agent_engine.llm_factory import LLMFactory
from langchain_core.messages import BaseMessage


# JSON Schema para structured outputs (garantiza formato correcto)
ORCHESTRATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["greeting", "question", "complaint", "request_human", "other"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "needs_knowledge_base": {"type": "boolean"},
        "kb_search_strategy": {
            "type": "string",
            "enum": ["exact", "broad", "multi_query", "none"]
        },
        "search_queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 3
        },
        "complexity": {
            "type": "string",
            "enum": ["simple", "medium", "complex"]
        },
        "should_handoff": {"type": "boolean"},
        "handoff_reason": {"type": ["string", "null"]},
        "response_strategy": {
            "type": "string",
            "enum": ["direct", "with_context", "multi_step", "deflect"]
        },
        "customer_sentiment": {
            "type": "string",
            "enum": ["very_positive", "positive", "neutral", "negative", "very_negative"]
        },
        "reasoning": {"type": "string"}
    },
    "required": [
        "intent", "confidence", "needs_knowledge_base",
        "kb_search_strategy", "search_queries", "complexity",
        "should_handoff", "handoff_reason", "response_strategy",
        "customer_sentiment", "reasoning"
    ],
    "additionalProperties": False
}


# System prompt hard-coded (NO editable por usuario)
ORCHESTRATOR_SYSTEM_PROMPT = """Eres el orchestrator de un agente conversacional inteligente.

CONTEXTO DE NEGOCIO:
{business_context}

CONVERSACIÓN (últimos 3 mensajes + resumen):
{conversation_history}

MENSAJE ACTUAL DEL CLIENTE:
"{current_message}"

ESTADO:
- Es primer mensaje: {is_first_message}
- Resumen conversación: {conversation_summary}

ANALIZA Y RESPONDE EN JSON ESTRUCTURADO:
{{
  "intent": "greeting|question|complaint|request_human|other",
  "confidence": 0.0-1.0,  // TU CONFIANZA en poder responder bien
  "needs_knowledge_base": boolean,
  "kb_search_strategy": "exact|broad|multi_query|none",
  "search_queries": ["query1", "query2"],  // Si multi_query, 2-3 variaciones
  "complexity": "simple|medium|complex",
  "should_handoff": boolean,
  "handoff_reason": "string|null",
  "response_strategy": "direct|with_context|multi_step|deflect",
  "customer_sentiment": "very_positive|positive|neutral|negative|very_negative",
  "reasoning": "Tu análisis paso a paso"
}}

CRITERIOS DE CONFIDENCE:
• 0.9-1.0: Muy seguro (pregunta simple O info clara en KB esperada)
• 0.7-0.9: Seguro moderado (pregunta estándar, probablemente en KB)
• 0.5-0.7: Inseguro (pregunta ambigua, puede no estar en KB)
• 0.3-0.5: Muy inseguro (pregunta compleja/fuera de alcance)
• 0.0-0.3: Sin capacidad (pregunta imposible de responder)

CRITERIOS DE HANDOFF:
• Cliente pide explícitamente hablar con humano
• Pregunta fuera de alcance del negocio
• Sentimiento muy negativo + frustración creciente
• Confidence < 0.5 en temas críticos (precios, garantías, problemas técnicos)
• Cliente repite la misma pregunta 2+ veces (señal de insatisfacción)

KB SEARCH STRATEGY:
• exact: Query directa (ej: "horarios", "precio de X")
• broad: Expandir con sinónimos (ej: "costo" → "precio, costo, valor")
• multi_query: 2-3 variaciones (ej: "cuándo abren" → ["horarios apertura", "horario tienda", "cuándo abren"])
• none: No necesita KB (saludo, despedida, conversación casual)

RESPONSE STRATEGY:
• direct: Respuesta simple sin KB (saludos, confirmaciones)
• with_context: Respuesta usando KB (preguntas sobre productos/servicios)
• multi_step: Requiere varias interacciones (proceso complejo)
• deflect: No podemos responder, sugerir alternativa o handoff
"""


def build_conversation_context(
    messages: List[BaseMessage], 
    summary: Optional[Dict[str, Any]] = None
) -> str:
    """
    Construir contexto conversacional con sliding window.
    
    Últimos 3 mensajes completos + summary de conversación previa.
    Reduce contexto de ~400 tokens a ~150 tokens.
    """
    # Últimos 3 mensajes
    recent_messages = messages[-3:] if len(messages) > 3 else messages
    
    lines = []
    
    # Agregar summary si existe y hay más de 5 mensajes
    if summary and len(messages) > 5:
        lines.append(f"[Resumen conversación previa: {summary.get('main_topic', 'N/A')}]")
        if summary.get('key_facts'):
            lines.append(f"Hechos clave: {', '.join(summary['key_facts'][:3])}")
        lines.append("")
    
    # Agregar últimos mensajes
    for msg in recent_messages:
        role = "Cliente" if msg.type == 'human' else "Asistente"
        lines.append(f"{role}: {msg.content}")
    
    return "\n".join(lines)


def _default_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Estado por defecto si no hay mensajes"""
    return {
        'intent': 'other',
        'confidence': 0.5,
        'needs_knowledge_base': False,
        'kb_search_strategy': 'none',
        'search_queries': [],
        'complexity': 'simple',
        'should_handoff': False,
        'handoff_reason': None,
        'response_strategy': 'direct',
        'customer_sentiment': 'neutral',
        'orchestrator_reasoning': 'No messages to analyze',
        'routing_decision': 'direct_respond',
        'nodes_visited': state.get('nodes_visited', []) + ['orchestrator']
    }


def _fallback_decision(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback conservador si falla el orchestrator"""
    return {
        'intent': 'question',
        'confidence': 0.4,  # Baja confianza = cauteloso
        'needs_knowledge_base': True,
        'kb_search_strategy': 'broad',
        'search_queries': [state['messages'][-1].content],
        'complexity': 'medium',
        'should_handoff': False,
        'handoff_reason': None,
        'response_strategy': 'with_context',
        'customer_sentiment': 'neutral',
        'orchestrator_reasoning': 'Fallback por error en orchestrator',
        'routing_decision': 'fallback_kb_search',
        'nodes_visited': state.get('nodes_visited', []) + ['orchestrator']
    }


async def orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo orchestrator - Cerebro del agente con gpt-5-nano extended.
    Analiza conversación completa y genera plan de ejecución.
    
    Usa gpt-5-nano (NO mini) con reasoning: extended para optimizar costos 5x.
    """
    import time
    orchestrator_start = time.time()
    
    # Extraer mensajes
    messages = state['messages']
    human_messages = [m for m in messages if m.type == 'human']
    
    if not human_messages:
        return _default_state(state)
    
    # Construir contexto
    is_first_message = len(human_messages) == 1
    current_message = human_messages[-1].content
    
    # Obtener summary de conversación si existe
    conversation_summary = state.get('conversation_summary', {})
    summary_text = "Sin resumen previo"
    if conversation_summary and isinstance(conversation_summary, dict):
        summary_text = f"""
Tema: {conversation_summary.get('main_topic', 'N/A')}
Estado: {conversation_summary.get('conversation_state', 'ongoing')}
Hechos clave: {', '.join(conversation_summary.get('key_facts', [])[:3])}
"""
    
    # Construir contexto con sliding window
    conversation_history = build_conversation_context(
        messages, 
        conversation_summary if conversation_summary else None
    )
    
    # Business context (desde state)
    business_context = state.get('business_context', 'Negocio de atención al cliente')
    
    # Construir prompt
    prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
        business_context=business_context,
        conversation_history=conversation_history,
        current_message=current_message,
        is_first_message=is_first_message,
        conversation_summary=summary_text
    )
    
    # Llamar a gpt-5-nano con structured output
    try:
        client = LLMFactory.create_responses_client()
        
        llm_start = time.time()
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            # SIN reasoning para velocidad (~2-3s vs 27s con gpt-5-nano+reasoning:high)
            text={
                "verbosity": "low",  # Respuestas concisas para ahorrar tokens
                "format": {
                    "type": "json_schema",
                    "name": "orchestrator_decision",
                    "strict": True,
                    "schema": ORCHESTRATOR_SCHEMA
                }
            }
        )
        
        llm_time = (time.time() - llm_start) * 1000
        decision = json.loads(response.output_text)
        
        print(f"🧠 [ORCHESTRATOR] Decision: confidence={decision['confidence']:.2f}, strategy={decision['kb_search_strategy']}, handoff={decision['should_handoff']}")
        print(f"   Reasoning: {decision['reasoning'][:100]}...")
        print(f"⏱️ [ORCHESTRATOR] LLM call: {llm_time:.0f}ms")
        
        # Determinar routing_decision basado en el análisis
        if decision['should_handoff']:
            routing_decision = 'force_handoff'
        elif decision['confidence'] < 0.6:
            routing_decision = 'suggest_handoff'
        elif is_first_message:
            routing_decision = 'greet'
        elif decision['needs_knowledge_base']:
            routing_decision = 'retrieve_knowledge'
        else:
            routing_decision = 'direct_respond'
        
        # Actualizar state con decisión
        orchestrator_time = (time.time() - orchestrator_start) * 1000
        print(f"⏱️ [ORCHESTRATOR] Total: {orchestrator_time:.0f}ms")
        
        return {
            **decision,  # Spread all orchestrator fields
            'orchestrator_reasoning': decision['reasoning'],
            'routing_decision': routing_decision,
            'is_first_message': is_first_message,
            'nodes_visited': state.get('nodes_visited', []) + ['orchestrator']
        }
        
    except Exception as e:
        print(f"❌ Error en orchestrator: {e}")
        return _fallback_decision(state)

