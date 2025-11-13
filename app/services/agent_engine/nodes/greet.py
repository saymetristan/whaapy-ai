from typing import Dict, Any
from langchain_core.messages import AIMessage


async def greet_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de saludo inicial.
    Solo saluda en el primer mensaje de la conversación.
    """
    # Solo saludar si es el primer mensaje (solo hay 1 mensaje del usuario)
    human_messages = [m for m in state['messages'] if m.type == 'human']
    
    if len(human_messages) > 1:
        # No es el primer mensaje, skip el saludo
        return {
            'nodes_visited': state.get('nodes_visited', []) + ['greet']
        }
    
    # Primer mensaje, agregar saludo
    greeting = AIMessage(content="¡Hola! 👋 ¿En qué puedo ayudarte hoy?")
    
    return {
        'messages': [greeting],
        'nodes_visited': state.get('nodes_visited', []) + ['greet']
    }

