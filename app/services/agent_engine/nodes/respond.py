from typing import Dict, Any
from app.services.agent_engine.llm_factory import LLMFactory, is_gpt5_model
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
    
    # Obtener últimos 5 mensajes para contexto
    recent_messages = state['messages'][-5:]
    
    # Construir input completo para Responses API
    # Formato: "System: {system}\n\nUser: {msg1}\nAssistant: {msg2}\n..."
    conversation_text = f"System: {system_prompt}\n\n"
    
    for msg in recent_messages:
        role = "User" if msg.type == 'human' else "Assistant"
        conversation_text += f"{role}: {msg.content}\n"
    
    # Llamar a Responses API vía factory
    try:
        client = LLMFactory.create_responses_client()
        model = config.get('model', 'gpt-5-mini')
        
        # Responses API es SÍNCRONA, no usar await
        # Solo usar reasoning/text si el modelo soporta GPT-5 controls
        llm_start = time.time()
        if is_gpt5_model(model):
            response = client.responses.create(
                model=model,
                input=conversation_text,
                reasoning={ "effort": "medium" }  # Razonamiento moderado para respuestas
            )
        else:
            # Fallback para modelos no-GPT5 (sin reasoning controls)
            response = client.responses.create(
                model=model,
                input=conversation_text
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
