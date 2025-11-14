import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage
from app.services.agent_engine.graph import create_agent_graph
from app.services.agent_engine.state import create_initial_state
from app.services.agent_engine.analytics_tracking import calculate_cost
from app.services.agent_engine.token_tracker import TokenTrackerCallback
from app.db.database import get_db


class AgentEngine:
    """
    Motor principal del agente conversacional con LangGraph.
    
    Gestiona la ejecución del grafo y el tracking en base de datos.
    """
    
    def __init__(self, agent_config: Dict[str, Any]):
        """
        Constructor del AgentEngine.
        
        Args:
            agent_config: Configuración del agente (system_prompt, provider, model, etc)
        """
        self.config = agent_config
        self.graph = create_agent_graph()
    
    async def chat(
        self,
        business_id: str,
        conversation_id: str,
        customer_phone: str,
        message: str,
        customer_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Procesar un mensaje del cliente y generar respuesta.
        
        Args:
            business_id: ID del negocio
            conversation_id: ID de la conversación
            customer_phone: Teléfono del cliente
            message: Mensaje del cliente
            customer_name: Nombre del cliente (opcional)
        
        Returns:
            Dict con response y metadata
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Crear execution record en DB
        with get_db() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO ai.agent_executions (
                        id, business_id, conversation_id, status, started_at, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    execution_id,
                    business_id,
                    conversation_id,
                    'active',
                    start_time,
                    start_time
                ))
                
                conn.commit()
                print(f"✅ Execution creada: {execution_id}")
                
            except Exception as e:
                print(f"Error creando execution: {e}")
                conn.rollback()
                raise
            
            finally:
                cursor.close()
        
        try:
            # Crear estado inicial
            initial_state = create_initial_state(
                business_id=business_id,
                conversation_id=conversation_id,
                customer_phone=customer_phone,
                execution_id=execution_id,
                message=HumanMessage(content=message),
                customer_name=customer_name
            )
            
            # Limpiar config para evitar enviar temperature (OpenAI no lo acepta)
            clean_config = {
                'provider': self.config.get('provider', 'openai'),
                'model': self.config.get('model', 'gpt-5-mini'),
                'max_tokens': self.config.get('max_tokens', 2000),
                'system_prompt': self.config.get('system_prompt', '')
            }
            
            # Crear callback para tracking de tokens
            token_tracker = TokenTrackerCallback()
            
            # Ejecutar grafo con configuración del agente y callback
            print(f"🚀 Ejecutando grafo para execution {execution_id}")
            result = await self.graph.ainvoke(
                initial_state,
                config={
                    "configurable": clean_config,
                    "callbacks": [token_tracker]
                }
            )
            
            # Extraer última respuesta del asistente
            ai_messages = [m for m in result['messages'] if m.type == 'ai']
            
            if ai_messages:
                response_text = ai_messages[-1].content
            else:
                response_text = "Lo siento, no pude procesar tu mensaje."
            
            # Calcular duración
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Obtener tokens del callback tracker
            tokens_used = token_tracker.get_total_tokens()
            
            print(f"📊 Tokens capturados: {tokens_used}")
            
            # Calcular costo
            model = self.config.get('model', 'gpt-5-mini')
            cost = calculate_cost(tokens_used, model)
            
            # Actualizar execution en DB con resultado + métricas RAG
            with get_db() as conn:
                cursor = conn.cursor()
                
                try:
                    # Extraer métricas RAG del resultado (si existen)
                    rag_metrics = result.get('rag_metrics', {})
                    
                    cursor.execute("""
                        UPDATE ai.agent_executions
                        SET status = 'completed',
                            completed_at = %s,
                            nodes_visited = %s,
                            tokens_used = %s,
                            cost = %s,
                            metadata = jsonb_build_object(
                                'intent', %s,
                                'sentiment', %s,
                                'handoff', %s,
                                'duration_ms', %s,
                                'rag', jsonb_build_object(
                                    'chunks_retrieved', %s,
                                    'rag_tokens', %s,
                                    'sources_count', %s,
                                    'avg_similarity', %s,
                                    'retrieval_time_ms', %s
                                )
                            )
                        WHERE id = %s
                    """, (
                        datetime.now(),
                        result.get('nodes_visited', []),
                        tokens_used,
                        cost,
                        result.get('intent'),
                        result.get('sentiment'),
                        result.get('should_handoff', False),
                        duration_ms,
                        # Métricas RAG
                        rag_metrics.get('chunks_retrieved', 0),
                        rag_metrics.get('total_tokens', 0),
                        len(rag_metrics.get('sources', [])),
                        round(rag_metrics.get('avg_similarity', 0), 3),
                        rag_metrics.get('retrieval_time_ms', 0),
                        execution_id
                    ))
                    
                    conn.commit()
                    
                    # Log con métricas RAG si existieron
                    if rag_metrics.get('chunks_retrieved', 0) > 0:
                        print(f"✅ Execution completada: {execution_id} ({duration_ms}ms, {tokens_used} tokens, ${cost:.6f})")
                        print(f"   RAG: {rag_metrics['chunks_retrieved']} chunks, {rag_metrics['total_tokens']} tokens, {len(rag_metrics.get('sources', []))} docs")
                    else:
                        print(f"✅ Execution completada: {execution_id} ({duration_ms}ms, {tokens_used} tokens, ${cost:.6f})")
                    
                except Exception as e:
                    print(f"Error actualizando execution: {e}")
                    conn.rollback()
                
                finally:
                    cursor.close()
            
            # Retornar respuesta con metadata
            return {
                'response': response_text,
                'metadata': {
                    'execution_id': execution_id,
                    'intent': result.get('intent'),
                    'sentiment': result.get('sentiment'),
                    'nodes_visited': result.get('nodes_visited', []),
                    'duration_ms': duration_ms,
                    'handoff': result.get('should_handoff', False)
                }
            }
        
        except Exception as error:
            print(f"❌ Error en ejecución del agente: {error}")
            
            # Marcar execution como failed
            with get_db() as conn:
                cursor = conn.cursor()
                
                try:
                    cursor.execute("""
                        UPDATE ai.agent_executions
                        SET status = 'failed',
                            completed_at = %s,
                            error = %s
                        WHERE id = %s
                    """, (
                        datetime.now(),
                        str(error),
                        execution_id
                    ))
                    
                    conn.commit()
                    
                except Exception as e:
                    print(f"Error marcando execution como failed: {e}")
                    conn.rollback()
                
                finally:
                    cursor.close()
            
            # Re-lanzar error
            raise

