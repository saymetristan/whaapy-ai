from typing import Dict, Any


async def call_tools_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de ejecución de tools/webhooks.
    
    Por ahora es un stub. La implementación real vendrá en Fase 2
    cuando se integren webhooks configurables.
    """
    print("🔧 call_tools_node: Stub - no tools configurados aún")
    
    # TODO Fase 2: 
    # - Obtener webhooks configurados del negocio
    # - Ejecutar webhooks relevantes según el contexto
    # - Agregar resultados al estado
    
    return {
        'nodes_visited': state.get('nodes_visited', []) + ['call_tools']
    }

