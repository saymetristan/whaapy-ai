from typing import Dict, Any
from datetime import datetime
from app.services.knowledge_base import KnowledgeBase
from app.services.agent_engine.analytics_tracking import save_tool_execution


async def retrieve_knowledge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de retrieval de knowledge base.
    Busca información relevante en los embeddings del negocio.
    """
    start_time = datetime.now()
    execution_id = state.get('execution_id')
    
    # Obtener último mensaje del usuario
    human_messages = [m for m in state['messages'] if m.type == 'human']
    
    if not human_messages:
        return {
            'nodes_visited': state.get('nodes_visited', []) + ['retrieve_knowledge']
        }
    
    last_user_message = human_messages[-1]
    
    # Buscar en knowledge base
    kb = KnowledgeBase()
    
    # Threshold adaptativo según confidence del orchestrator
    confidence = state.get('confidence', 0.5)
    
    if confidence > 0.85:
        threshold = 0.4  # Alta confianza → más estricto
    elif confidence > 0.7:
        threshold = 0.35  # Media confianza → balanceado
    else:
        threshold = 0.3  # Baja confianza → permisivo
    
    print(f"🎯 [KB] Adaptive threshold: {threshold} (confidence={confidence:.2f})")
    
    try:
        results = await kb.search(
            business_id=state['business_id'],
            query=last_user_message.content,
            k=3,
            threshold=threshold
        )
        
        # Extraer contenido de los documentos relevantes
        retrieved_docs = [doc['content'] for doc in results]
        
        print(f"📚 Retrieved {len(retrieved_docs)} docs from KB")
        
        # Log tool execution
        if execution_id:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            save_tool_execution(
                execution_id=execution_id,
                tool_name='knowledge_base_search',
                duration_ms=duration_ms,
                success=True,
                request_data={'query': last_user_message.content, 'k': 3, 'threshold': threshold},
                response_data={'results_count': len(retrieved_docs)}
            )
        
    except Exception as e:
        print(f"❌ [KB] Error en búsqueda: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        retrieved_docs = []
        
        # Log failed tool execution
        if execution_id:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            save_tool_execution(
                execution_id=execution_id,
                tool_name='knowledge_base_search',
                duration_ms=duration_ms,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                request_data={'query': last_user_message.content, 'k': 3, 'threshold': threshold}
            )
    
    return {
        'retrieved_docs': retrieved_docs if retrieved_docs else None,
        'nodes_visited': state.get('nodes_visited', []) + ['retrieve_knowledge']
    }

