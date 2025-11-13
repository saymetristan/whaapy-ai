import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.agent_engine.llm_factory import LLMFactory


async def analyze_intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de análisis de intención.
    Usa gpt-5-mini para analizar rápidamente la intención del mensaje.
    """
    # Obtener último mensaje del usuario
    human_messages = [m for m in state['messages'] if m.type == 'human']
    
    if not human_messages:
        return {
            'nodes_visited': state.get('nodes_visited', []) + ['analyze_intent']
        }
    
    last_user_message = human_messages[-1]
    
    # LLM rápido para análisis
    llm = LLMFactory.create_fast()
    
    # Prompt para análisis
    analysis_prompt = f"""Analiza el siguiente mensaje del cliente y responde en JSON:

Mensaje: "{last_user_message.content}"

Responde con:
{{
  "intent": "greeting|question|complaint|request_human|other",
  "sentiment": "positive|neutral|negative",
  "should_handoff": boolean,
  "reason": "razón si should_handoff es true"
}}

Si el cliente pide explícitamente hablar con un humano, should_handoff debe ser true."""
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content="Eres un analizador de intenciones. Responde solo con JSON válido."),
            HumanMessage(content=analysis_prompt)
        ])
        
        # Parsear respuesta
        analysis = json.loads(response.content)
        
    except Exception as e:
        print(f"Error analizando intención: {e}")
        # Fallback a valores por defecto
        analysis = {
            'intent': 'other',
            'sentiment': 'neutral',
            'should_handoff': False,
            'reason': None
        }
    
    return {
        'intent': analysis.get('intent'),
        'sentiment': analysis.get('sentiment'),
        'should_handoff': analysis.get('should_handoff', False),
        'handoff_reason': analysis.get('reason'),
        'nodes_visited': state.get('nodes_visited', []) + ['analyze_intent']
    }

