from typing import Dict, Any
from langchain_core.messages import AIMessage
from app.db.database import get_db


async def handoff_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo de transferencia a humano.
    Marca la ejecución como handoff y envía mensaje de transferencia.
    """
    # Actualizar execution en DB con status handoff
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE ai.agent_executions
                SET status = 'handoff',
                    metadata = jsonb_set(
                        COALESCE(metadata, '{}'::jsonb),
                        '{handoff_reason}',
                        to_jsonb(%s::text)
                    )
                WHERE id = %s
            """, (
                state.get('handoff_reason', 'Usuario solicitó atención humana'),
                state['execution_id']
            ))
            
            conn.commit()
            print(f"👤 Handoff marcado para execution {state['execution_id']}")
            
        except Exception as e:
            print(f"Error actualizando handoff en DB: {e}")
            conn.rollback()
        
        finally:
            cursor.close()
    
    # Mensaje de transferencia
    message = AIMessage(
        content="Entiendo, te voy a conectar con un miembro de nuestro equipo. Un momento por favor... 👤"
    )
    
    # Nota: La actualización de public.conversations (ai_paused=true) 
    # se hará desde el backend cuando reciba la metadata de handoff
    
    return {
        'messages': [message],
        'nodes_visited': state.get('nodes_visited', []) + ['handoff']
    }

