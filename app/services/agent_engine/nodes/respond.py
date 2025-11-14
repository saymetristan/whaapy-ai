from typing import Dict, Any
from app.services.agent_engine.llm_factory import LLMFactory, is_gpt5_model
from app.services.llm_tracker import LLMCallTracker
from app.services.agent_engine.prompt_composer import PromptComposer
from langchain_core.messages import AIMessage


async def respond_node(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de generación de respuesta usando Responses API.
    Migrado de Chat Completions a Responses API para mejor performance y caching.
    Ahora usa PromptComposer para construcción multi-layer de prompts (Sprint 5).
    """
    import time
    respond_start = time.time()
    
    # Usar PromptComposer para construir el prompt completo (Sprint 5)
    system_prompt = PromptComposer.compose_full_prompt(
        config=config,
        state=state,
        include_kb_context=True,
        include_disclaimers=True
    )
    
    # Logging de confidence (mantener para debugging)
    confidence = state.get('confidence', 1.0)
    suggest_handoff = state.get('suggest_handoff_in_response', False)
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
