"""
Validation & Self-Correction Nodes - Sprint 3

Implementa validation condicional (solo si confidence < 0.75) y self-correction
con máximo 1 retry para evitar loops infinitos.
"""

import json
from typing import Dict, Any
from app.services.agent_engine.llm_factory import LLMFactory
from app.services.llm_tracker import LLMCallTracker
from langchain_core.messages import AIMessage


# JSON Schema para validation response
VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {
            "type": "boolean",
            "description": "True si la respuesta es de calidad, False si necesita mejora"
        },
        "quality_score": {
            "type": "number",
            "description": "Score de calidad entre 0.0 y 1.0"
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Lista de problemas encontrados en la respuesta"
        },
        "suggestions": {
            "type": "string",
            "description": "Feedback específico para mejorar la respuesta"
        }
    },
    "required": ["passed", "quality_score", "issues", "suggestions"],
    "additionalProperties": False
}


VALIDATION_SYSTEM_PROMPT = """Eres un validador de calidad de respuestas de IA conversacional.

Tu trabajo es evaluar si una respuesta de IA es de ALTA CALIDAD o necesita mejora.

CRITERIOS DE EVALUACIÓN:

1. **Responde la pregunta** (25 puntos)
   - ✅ Aborda directamente lo que el cliente preguntó
   - ❌ Ignora la pregunta o da respuestas tangenciales

2. **Especificidad** (25 puntos)
   - ✅ Respuesta concreta con datos/detalles útiles
   - ❌ Respuesta vaga, genérica, sin sustancia

3. **Uso de contexto** (25 puntos)
   - ✅ Usa información relevante del conocimiento base
   - ❌ No usa contexto disponible o lo usa incorrectamente

4. **Profesionalismo** (15 puntos)
   - ✅ Tono apropiado, bien estructurada, sin errores
   - ❌ Errores gramaticales, tono inadecuado, mal formateo

5. **No inventa información** (10 puntos)
   - ✅ Solo usa información verificable del contexto
   - ❌ Inventa datos, hace suposiciones infundadas

SCORING:
- 0.85-1.0: Excelente (passed=true)
- 0.70-0.84: Buena (passed=true)
- 0.50-0.69: Regular (passed=false, necesita retry)
- 0.0-0.49: Mala (passed=false, necesita retry urgente)

RESPONDE EN JSON ESTRUCTURADO:
{
  "passed": boolean,
  "quality_score": 0.0-1.0,
  "issues": ["problema 1", "problema 2", ...],
  "suggestions": "Feedback específico para mejorar: ..."
}

**IMPORTANTE**: Si la respuesta es razonablemente buena (>0.70), marca passed=true aunque no sea perfecta."""


async def validate_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validar calidad de la respuesta generada.
    
    Solo se ejecuta si confidence < 0.75 (optimización tokens).
    Usa gpt-5-mini con reasoning low para velocidad y costo.
    """
    import time
    validation_start = time.time()
    
    # Extraer última respuesta del agente
    messages = state['messages']
    ai_messages = [m for m in messages if m.type == 'ai']
    
    if not ai_messages:
        # No hay respuesta que validar (edge case)
        print("⚠️ [VALIDATION] No AI response to validate")
        return {
            'validation_passed': True,
            'quality_score': 1.0,
            'validation_issues': [],
            'validation_feedback': '',
            'nodes_visited': state.get('nodes_visited', []) + ['validate_response']
        }
    
    assistant_response = ai_messages[-1].content
    
    # Extraer query del cliente (último mensaje humano)
    human_messages = [m for m in messages if m.type == 'human']
    user_query = human_messages[-1].content if human_messages else "N/A"
    
    # Contexto de KB usado
    retrieved_docs = state.get('retrieved_docs', [])
    if retrieved_docs and len(retrieved_docs) > 0:
        context_preview = "\n".join(retrieved_docs)[:500]  # Primeros 500 chars
        context_info = f"Contexto disponible (preview):\n{context_preview}..."
    else:
        context_info = "Sin contexto de knowledge base"
    
    # Construir input para validation
    validation_input = f"""
PREGUNTA DEL CLIENTE:
{user_query}

RESPUESTA DEL ASISTENTE:
{assistant_response}

CONTEXTO DISPONIBLE:
{context_info}

Evalúa la calidad de la respuesta según los criterios definidos.
"""
    
    try:
        # Crear OpenAI client (gpt-5-mini)
        llm_factory = LLMFactory()
        openai_client = llm_factory.create_openai_client()
        
        # Track LLM call
        async with LLMCallTracker(
            business_id=state['business_id'],
            operation_type='validation',
            provider='openai',
            model='gpt-5-mini',
            execution_id=state['execution_id'],
            operation_context={
                'node': 'validate_response',
                'conversation_id': state.get('conversation_id'),
                'confidence': state.get('confidence')
            },
            reasoning_effort='low'
        ) as tracker:
            
            llm_start = time.time()
            response = openai_client.responses.create(
                model="gpt-5-mini",
                reasoning={"effort": "low"},  # Balance costo/calidad
                text={
                    "verbosity": "low",
                    "format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "validation_result",
                            "strict": True,
                            "schema": VALIDATION_SCHEMA
                        }
                    }
                },
                messages=[
                    {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                    {"role": "user", "content": validation_input}
                ]
            )
            
            # Record tokens
            tracker.record(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )
            
            # Parse result
            validation_result = json.loads(response.choices[0].message.content)
        
        llm_time = (time.time() - llm_start) * 1000
        validation_time = (time.time() - validation_start) * 1000
        
        passed = validation_result['passed']
        quality_score = validation_result['quality_score']
        status_emoji = "✅" if passed else "❌"
        
        print(f"{status_emoji} [VALIDATION] Quality: {quality_score:.2f}, Passed: {passed}")
        if not passed:
            print(f"   Issues: {', '.join(validation_result['issues'][:2])}")
        print(f"⏱️ [VALIDATION] LLM: {llm_time:.0f}ms, Total: {validation_time:.0f}ms")
        
        return {
            'validation_passed': passed,
            'quality_score': quality_score,
            'validation_issues': validation_result.get('issues', []),
            'validation_feedback': validation_result.get('suggestions', ''),
            'nodes_visited': state.get('nodes_visited', []) + ['validate_response']
        }
        
    except Exception as e:
        print(f"❌ Error en validation: {e}")
        # Fallback: asumir que pasó (no bloquear flujo)
        return {
            'validation_passed': True,
            'quality_score': 0.8,
            'validation_issues': [f"Validation error: {str(e)}"],
            'validation_feedback': '',
            'nodes_visited': state.get('nodes_visited', []) + ['validate_response']
        }


async def retry_respond_node(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Regenerar respuesta usando feedback de validation.
    
    Máximo 1 retry - después de esto termina de todos modos (evitar loops infinitos).
    Usa el mismo modelo que respond_node pero con prompt mejorado.
    """
    import time
    retry_start = time.time()
    
    # Extraer feedback de validation
    validation_feedback = state.get('validation_feedback', 'La respuesta anterior no fue suficientemente específica.')
    validation_issues = state.get('validation_issues', [])
    
    # Construir system prompt mejorado con feedback
    base_system_prompt = config.get('system_prompt', 'Eres un asistente virtual de atención al cliente.')
    
    enhanced_system_prompt = f"""{base_system_prompt}

🔴 CRÍTICO - TU RESPUESTA ANTERIOR FUE RECHAZADA POR BAJA CALIDAD 🔴

Problemas encontrados:
{chr(10).join(f'- {issue}' for issue in validation_issues)}

Feedback para mejorar:
{validation_feedback}

INSTRUCCIONES PARA ESTA RESPUESTA:
1. NO repitas la respuesta anterior
2. Sé MÁS ESPECÍFICO con datos concretos
3. Usa TODA la información disponible del contexto
4. Estructura la respuesta de forma CLARA
5. Responde DIRECTAMENTE a lo que se preguntó

Esta es tu ÚNICA oportunidad de mejorar. Hazlo bien.
"""
    
    # Agregar contexto de KB si existe
    if state.get('retrieved_docs'):
        context = "\n\n".join(state['retrieved_docs'])
        enhanced_system_prompt += f"\n\nInformación relevante de la base de conocimiento:\n{context}"
    
    # Obtener últimos 5 mensajes (sin la respuesta fallida)
    messages_without_failed = [m for m in state['messages'][:-1]]  # Remover última respuesta
    recent_messages = messages_without_failed[-5:]
    
    # Construir conversation text para Responses API
    conversation_text = f"System: {enhanced_system_prompt}\n\n"
    
    for msg in recent_messages:
        role = "User" if msg.type == 'human' else "Assistant"
        conversation_text += f"{role}: {msg.content}\n"
    
    try:
        client = LLMFactory.create_groq_client()
        model = config.get('model', 'openai/gpt-oss-120b')
        
        # Track LLM call
        async with LLMCallTracker(
            business_id=state['business_id'],
            operation_type='chat',
            provider='groq',
            model=model,
            execution_id=state['execution_id'],
            operation_context={
                'node': 'retry_respond',
                'conversation_id': state.get('conversation_id'),
                'is_retry': True,
                'original_quality_score': state.get('quality_score')
            },
            reasoning_effort='high'  # Usar high reasoning en retry para mejor calidad
        ) as tracker:
            
            llm_start = time.time()
            response = client.responses.create(
                model=model,
                input=conversation_text,
                reasoning={"effort": "high"},  # Más effort para mejorar calidad
                temperature=0.3  # Algo más creativo que respond normal (0.2)
            )
            
            # Record tokens
            tracker.record(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )
            
            improved_response = response.output_text
        
        llm_time = (time.time() - llm_start) * 1000
        retry_time = (time.time() - retry_start) * 1000
        
        print(f"🔄 [RETRY] Respuesta mejorada generada: {improved_response[:100]}...")
        print(f"⏱️ [RETRY] LLM: {llm_time:.0f}ms, Total: {retry_time:.0f}ms")
        
        # Reemplazar la respuesta fallida con la mejorada
        # Remover última respuesta AI del state
        new_messages = [m for m in state['messages'] if not (m.type == 'ai' and m == state['messages'][-1])]
        
        # Agregar nueva respuesta mejorada
        new_messages.append(AIMessage(content=improved_response))
        
        return {
            'messages': new_messages,
            'was_retried': True,
            'nodes_visited': state.get('nodes_visited', []) + ['retry_respond']
        }
        
    except Exception as e:
        print(f"❌ Error en retry: {e}")
        # Si falla retry, mantener respuesta original (mejor que crashear)
        return {
            'was_retried': True,
            'nodes_visited': state.get('nodes_visited', []) + ['retry_respond']
        }

