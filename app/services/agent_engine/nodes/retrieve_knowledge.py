from typing import Dict, Any
from app.services.knowledge_base import KnowledgeBase


async def retrieve_knowledge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de retrieval de knowledge base.
    Busca información relevante en los embeddings del negocio.
    """
    # Obtener último mensaje del usuario
    human_messages = [m for m in state['messages'] if m.type == 'human']
    
    if not human_messages:
        return {
            'nodes_visited': state.get('nodes_visited', []) + ['retrieve_knowledge']
        }
    
    last_user_message = human_messages[-1]
    
    # Buscar en knowledge base
    kb = KnowledgeBase()
    
    try:
        results = await kb.search(
            business_id=state['business_id'],
            query=last_user_message.content,
            k=3,
            threshold=0.7
        )
        
        # Extraer contenido de los documentos relevantes
        retrieved_docs = [doc['content'] for doc in results]
        
        print(f"📚 Retrieved {len(retrieved_docs)} docs from KB")
        
    except Exception as e:
        print(f"Error retrieving knowledge: {e}")
        retrieved_docs = []
    
    return {
        'retrieved_docs': retrieved_docs if retrieved_docs else None,
        'nodes_visited': state.get('nodes_visited', []) + ['retrieve_knowledge']
    }

