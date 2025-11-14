from typing import Dict, Any
from datetime import datetime
from app.services.knowledge_base import KnowledgeBase
from app.services.agent_engine.analytics_tracking import save_tool_execution


async def retrieve_knowledge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de retrieval de knowledge base con RAG.
    Busca chunks relevantes usando búsqueda semántica con pgvector.
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
    
    # Buscar en knowledge base con RAG
    kb = KnowledgeBase()
    
    try:
        results = await kb.search(
            business_id=state['business_id'],
            query=last_user_message.content,
            k=5,  # Aumentado de 3 a 5 para mejor contexto
            threshold=0.7  # Threshold estándar para alta precisión
        )
        
        # Extraer contenido y calcular métricas RAG
        retrieved_docs = [doc['content'] for doc in results]
        
        # Calcular tokens totales de los chunks recuperados
        total_tokens = sum(doc.get('token_count', 0) for doc in results)
        
        # Extraer document_ids únicos (fuentes)
        unique_doc_ids = list(set(doc['document_id'] for doc in results))
        
        # Calcular similitud promedio
        avg_similarity = sum(doc['similarity'] for doc in results) / len(results) if results else 0
        
        print(f"📚 RAG Retrieved {len(retrieved_docs)} chunks from {len(unique_doc_ids)} documents")
        print(f"   Total tokens: {total_tokens}, Avg similarity: {avg_similarity:.3f}")
        
        # Almacenar métricas RAG en el estado para tracking posterior
        rag_metrics = {
            'chunks_retrieved': len(results),
            'total_tokens': total_tokens,
            'sources': unique_doc_ids,
            'avg_similarity': avg_similarity,
            'retrieval_time_ms': int((datetime.now() - start_time).total_seconds() * 1000)
        }
        
        # Log tool execution con métricas RAG detalladas
        if execution_id:
            save_tool_execution(
                execution_id=execution_id,
                tool_name='rag_knowledge_search',
                duration_ms=rag_metrics['retrieval_time_ms'],
                success=True,
                request_data={
                    'query': last_user_message.content, 
                    'k': 5, 
                    'threshold': 0.7
                },
                response_data={
                    'results_count': len(retrieved_docs),
                    'total_tokens': total_tokens,
                    'sources_count': len(unique_doc_ids),
                    'avg_similarity': round(avg_similarity, 3)
                }
            )
        
    except Exception as e:
        print(f"❌ Error retrieving knowledge with RAG: {e}")
        retrieved_docs = []
        rag_metrics = {
            'chunks_retrieved': 0,
            'total_tokens': 0,
            'sources': [],
            'avg_similarity': 0,
            'retrieval_time_ms': int((datetime.now() - start_time).total_seconds() * 1000)
        }
        
        # Log failed tool execution
        if execution_id:
            save_tool_execution(
                execution_id=execution_id,
                tool_name='rag_knowledge_search',
                duration_ms=rag_metrics['retrieval_time_ms'],
                success=False,
                error=str(e),
                request_data={'query': last_user_message.content, 'k': 5, 'threshold': 0.7}
            )
    
    return {
        'retrieved_docs': retrieved_docs if retrieved_docs else None,
        'rag_metrics': rag_metrics,  # Métricas RAG para tracking en agent_execution
        'nodes_visited': state.get('nodes_visited', []) + ['retrieve_knowledge']
    }

