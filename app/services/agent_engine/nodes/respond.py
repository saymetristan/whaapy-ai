from typing import Dict, Any
from app.services.agent_engine.llm_factory import LLMFactory, is_gpt5_model
from app.services.llm_tracker import LLMCallTracker
from langchain_core.messages import AIMessage


async def respond_node(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de generación de respuesta usando Responses API.
    Migrado de Chat Completions a Responses API para mejor performance y caching.
    """
    import time
    respond_start = time.time()
    
    # Construir system prompt con contexto de KB
    system_prompt = config.get('system_prompt', 'Eres un asistente virtual de atención al cliente.')
    
    # Agregar contexto de knowledge base si existe
    if state.get('retrieved_docs'):
        context = "\n\n".join(state['retrieved_docs'])
        system_prompt += f"\n\nInformación relevante de la base de conocimiento:\n{context}"
    
    # NUEVO: Agregar instrucciones según confidence (Sprint 4)
    confidence = state.get('confidence', 1.0)
    suggest_handoff = state.get('suggest_handoff_in_response', False)
    
    if confidence < 0.4:
        # Very low confidence → force handoff directo
        system_prompt += """

CRÍTICO: Tu nivel de confianza sobre esta consulta es MUY BAJO (<40%).
No tienes información suficiente para responder con certeza.
DEBES ofrecer conectar al usuario con un asesor humano de forma directa y clara.
Ejemplo: "Para ayudarte mejor con esto, te recomiendo hablar con uno de nuestros asesores. ¿Te conecto?"
"""
        print(f"⚠️ [RESPOND] Disclaimer inyectado (confidence {confidence:.2f}) - FORCE HANDOFF")
    elif 0.4 <= confidence < 0.6:
        # Low-medium confidence → sugerir handoff naturalmente
        system_prompt += """

NOTA: Tu nivel de confianza sobre esta consulta es MEDIO (40-60%).
Responde lo mejor que puedas con la información disponible, pero al final
sugiere de forma natural que pueden contactar a un asesor si necesitan más ayuda.
Ejemplo: "Si necesitas más detalles específicos, puedo conectarte con un asesor 👤"
"""
        print(f"⚠️ [RESPOND] Disclaimer inyectado (confidence {confidence:.2f}) - SUGGEST HANDOFF")
    elif suggest_handoff:
        # Orchestrator detectó necesidad de handoff (independiente de confidence)
        system_prompt += """

NOTA: Aunque puedes responder, el usuario podría beneficiarse de atención humana.
Incluye sutilmente la opción de hablar con un asesor si lo prefiere.
"""
        print(f"ℹ️ [RESPOND] Disclaimer sutil (suggest_handoff=true, confidence {confidence:.2f})")
    
    print(f"📊 [RESPOND] Confidence: {confidence:.2f}, Suggest handoff: {suggest_handoff}")
    
    # Obtener últimos 5 mensajes para contexto
    recent_messages = state['messages'][-5:]
    
    # Construir input completo para Responses API
    # Formato: "System: {system}\n\nUser: {msg1}\nAssistant: {msg2}\n..."
    conversation_text = f"System: {system_prompt}\n\n"
    
    for msg in recent_messages:
        role = "User" if msg.type == 'human' else "Assistant"
        conversation_text += f"{role}: {msg.content}\n"
    
    # Guardrail anti-hallucination: Solo si orchestrator INTENTÓ buscar KB
    retrieved_docs = state.get('retrieved_docs', [])
    has_context = retrieved_docs and len(retrieved_docs) > 0
    attempted_kb_search = state.get('needs_knowledge_base', False)
    
    if not has_context and attempted_kb_search:
        # SIN contexto KB Y orchestrator quería buscar → instruir explícitamente que NO alucine
        system_instruction = """

CRITICAL INSTRUCTION: 
You DO NOT have any information from the knowledge base about this query.
You MUST respond with:
"Lo siento, no tengo información específica sobre eso en mi base de conocimiento. ¿Te gustaría que te conecte con un asesor humano para ayudarte mejor?"

DO NOT make up or invent any information. DO NOT provide generic answers.
If you don't have the information in the knowledge base, you MUST say so and offer human assistance."""
        
        conversation_text = f"{system_instruction}\n\n{conversation_text}"
        print("⚠️ [RESPOND] NO KB context + orchestrator buscó → guardrail anti-hallucination")
    else:
        if has_context:
            print(f"✅ [RESPOND] KB context presente: {len(retrieved_docs)} docs")
        else:
            print(f"✅ [RESPOND] NO KB search needed (fast-path o no KB request)")
    
    # Llamar a Groq Responses API vía factory + tracking
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
                'node': 'respond',
                'conversation_id': state.get('conversation_id'),
                'has_kb_context': bool(state.get('retrieved_docs'))
            },
            reasoning_effort='medium'
        ) as tracker:
            # Groq Responses API con reasoning medium
            llm_start = time.time()
            response = client.responses.create(
                model=model,
                input=conversation_text,
                reasoning={"effort": "medium"},
                temperature=0.2
            )
            
            # Record tokens
            tracker.record(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )
            
            response_content = response.output_text
        
        llm_time = (time.time() - llm_start) * 1000
        respond_time = (time.time() - respond_start) * 1000
        print(f"🤖 Respuesta generada: {response_content[:100]}...")
        print(f"⏱️ [RESPOND] LLM call: {llm_time:.0f}ms, Total: {respond_time:.0f}ms")
        
    except Exception as e:
        print(f"Error generando respuesta: {e}")
        response_content = "Lo siento, tuve un problema al procesar tu mensaje. ¿Podrías intentar de nuevo?"
    
    return {
        'messages': [AIMessage(content=response_content)],
        'nodes_visited': state.get('nodes_visited', []) + ['respond']
    }
