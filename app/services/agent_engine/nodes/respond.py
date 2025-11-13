from typing import Dict, Any
from langchain_core.messages import SystemMessage, AIMessage
from app.services.agent_engine.llm_factory import LLMFactory


async def respond_node(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de generación de respuesta.
    Usa el LLM configurado para el agente y construye el contexto completo.
    """
    # Crear LLM según configuración del agente
    llm = LLMFactory.create_from_dict(config)
    
    # Construir system prompt con contexto de KB
    system_prompt = config.get('system_prompt', 'Eres un asistente virtual de atención al cliente.')
    
    # Agregar contexto de knowledge base si existe
    if state.get('retrieved_docs'):
        context = "\n\n".join(state['retrieved_docs'])
        system_prompt += f"\n\nInformación relevante de la base de conocimiento:\n{context}"
    
    # Obtener últimos 5 mensajes para contexto
    recent_messages = state['messages'][-5:]
    
    # Construir lista de mensajes para el LLM
    messages_for_llm = [
        SystemMessage(content=system_prompt),
        *recent_messages
    ]
    
    try:
        # Generar respuesta
        response = await llm.ainvoke(messages_for_llm)
        
        print(f"🤖 Respuesta generada: {response.content[:100]}...")
        
    except Exception as e:
        print(f"Error generando respuesta: {e}")
        response = AIMessage(content="Lo siento, tuve un problema al procesar tu mensaje. ¿Podrías intentar de nuevo?")
    
    return {
        'messages': [response],
        'nodes_visited': state.get('nodes_visited', []) + ['respond']
    }

