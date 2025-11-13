from typing import Dict, Any
from langchain_core.messages import AIMessage


async def greet_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de saludo inicial.
    
    Este nodo solo se ejecuta cuando analyze_intent detecta is_first_message=True.
    El routing garantiza que no se ejecuta innecesariamente.
    """
    greeting = AIMessage(content="¡Hola! 👋 ¿En qué puedo ayudarte hoy?")
    
    return {
        'messages': [greeting],
        'nodes_visited': state.get('nodes_visited', []) + ['greet']
    }
