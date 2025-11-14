import json
from typing import Dict, Any
from app.services.agent_engine.llm_factory import LLMFactory


async def analyze_intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de análisis de intención usando gpt-5-nano con minimal reasoning.
    
    Este nodo SIEMPRE se ejecuta primero para determinar:
    - Si es el primer mensaje (para ejecutar greet)
    - Intent del mensaje (greeting, question, complaint, etc)
    - Sentiment (positive, neutral, negative)
    - Si requiere handoff a humano
    """
    # Obtener mensajes del usuario
    human_messages = [m for m in state['messages'] if m.type == 'human']
    
    if not human_messages:
        return {
            'intent': 'other',
            'sentiment': 'neutral',
            'should_handoff': False,
            'is_first_message': False,
            'nodes_visited': state.get('nodes_visited', []) + ['analyze_intent']
        }
    
    last_user_message = human_messages[-1]
    is_first_message = len(human_messages) == 1
    message_content = last_user_message.content.lower()
    
    # Prompt optimizado para gpt-5-nano
    analysis_prompt = f"""Analiza el mensaje y responde SOLO con JSON válido:

Mensaje: "{last_user_message.content}"
Es primer mensaje: {is_first_message}

JSON requerido:
{{
  "intent": "greeting|question|complaint|request_human|other",
  "sentiment": "positive|neutral|negative",
  "should_handoff": boolean,
  "needs_knowledge": boolean,
  "reason": "razón del handoff si aplica"
}}

Reglas:
- should_handoff=true si pide hablar con humano
- needs_knowledge=true si hace pregunta específica sobre productos, servicios, precios, horarios, ubicación, etc.
- intent=greeting SOLO si es saludo puro (hola, buenos días, etc) SIN pregunta
- intent=question si hace cualquier pregunta"""

    system_prompt = "Eres un clasificador de intenciones. Responde SOLO con JSON válido."
    
    try:
        # Usar gpt-5-nano con minimal reasoning
        response_text = await LLMFactory.call_gpt5_nano_minimal(
            input_text=analysis_prompt,
            system_prompt=system_prompt
        )
        
        # Parsear JSON de la respuesta
        analysis = json.loads(response_text)
        
        print(f"🧠 [GPT-5-NANO] Intención analizada: intent={analysis.get('intent')}, needs_knowledge={analysis.get('needs_knowledge')}, first_msg={is_first_message}")
        
    except Exception as e:
        print(f"❌ Error analizando intención con gpt-5-nano: {e}")
        print(f"❌ Error completo: {str(e)}")
        
        # Fallback mejorado: detectar preguntas por keywords
        question_keywords = ['qué', 'que', 'cuál', 'cual', 'cómo', 'como', 'cuándo', 'cuando', 
                            'dónde', 'donde', 'por qué', 'porque', '?', 'precio', 'cuesta', 
                            'horario', 'hora', 'ubicación', 'dirección']
        handoff_keywords = ['humano', 'persona', 'agente', 'hablar con', 'operador']
        greeting_keywords = ['hola', 'buenos días', 'buenas tardes', 'buenas noches', 'hey', 'hi']
        
        is_question = any(kw in message_content for kw in question_keywords)
        is_handoff = any(kw in message_content for kw in handoff_keywords)
        is_greeting = any(kw in message_content for kw in greeting_keywords) and not is_question
        
        analysis = {
            'intent': 'greeting' if is_greeting else ('question' if is_question else 'other'),
            'sentiment': 'neutral',
            'should_handoff': is_handoff,
            'needs_knowledge': is_question,  # Si es pregunta, necesita KB
            'reason': 'Usuario solicitó hablar con humano' if is_handoff else None
        }
        print(f"⚠️ Usando fallback heurístico: intent={analysis['intent']}, needs_knowledge={analysis['needs_knowledge']}")
    
    return {
        'intent': analysis.get('intent', 'other'),
        'sentiment': analysis.get('sentiment', 'neutral'),
        'should_handoff': analysis.get('should_handoff', False),
        'needs_knowledge': analysis.get('needs_knowledge', False),
        'handoff_reason': analysis.get('reason'),
        'is_first_message': is_first_message,
        'nodes_visited': state.get('nodes_visited', []) + ['analyze_intent']
    }
