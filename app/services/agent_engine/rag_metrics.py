"""
RAG Metrics Tracking Module

Guarda métricas detalladas de RAG en ai.rag_metrics para analytics:
- Multi-query expansion
- Reranking aplicado
- Relevance validation
- Performance (latencias)
"""

from typing import List, Optional
from app.db.database import get_db_connection, return_db_connection


def save_rag_metrics(
    execution_id: str,
    business_id: str,
    original_query: str,
    queries_generated: List[str],
    search_strategy: str,
    semantic_weight: float,
    keyword_weight: float,
    threshold_used: float,
    chunks_found: int,
    chunks_after_reranking: Optional[int],
    reranking_applied: bool,
    relevance_validation_passed: Optional[bool],
    search_duration_ms: int,
    reranking_duration_ms: Optional[int],
    total_duration_ms: int
) -> None:
    """
    Guardar métricas RAG en ai.rag_metrics
    
    Args:
        execution_id: UUID de la ejecución del agente
        business_id: UUID del negocio
        original_query: Query original del usuario
        queries_generated: Lista de queries generadas (multi-query)
        search_strategy: 'hybrid', 'semantic_only', 'multi_query'
        semantic_weight: Peso semantic (0.0-1.0)
        keyword_weight: Peso keyword (0.0-1.0)
        threshold_used: Threshold usado para filtrar
        chunks_found: Chunks encontrados (antes reranking)
        chunks_after_reranking: Chunks después de reranking (si aplicado)
        reranking_applied: Si se aplicó reranking
        relevance_validation_passed: Si validation passed o rechazó chunks
        search_duration_ms: Latencia de búsqueda
        reranking_duration_ms: Latencia de reranking (si aplicado)
        total_duration_ms: Latencia total del RAG node
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            """
            INSERT INTO ai.rag_metrics (
                execution_id,
                business_id,
                original_query,
                queries_generated,
                queries_executed,
                search_strategy,
                semantic_weight,
                keyword_weight,
                threshold_used,
                chunks_found,
                chunks_after_reranking,
                reranking_applied,
                relevance_validation_passed,
                search_duration_ms,
                reranking_duration_ms,
                total_duration_ms
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                execution_id,
                business_id,
                original_query,
                queries_generated,  # PostgreSQL array
                len(queries_generated),
                search_strategy,
                semantic_weight,
                keyword_weight,
                threshold_used,
                chunks_found,
                chunks_after_reranking,
                reranking_applied,
                relevance_validation_passed,
                search_duration_ms,
                reranking_duration_ms,
                total_duration_ms
            )
        )
        
        conn.commit()
        
        print(f"✅ [RAG Metrics] Guardadas: strategy={search_strategy}, chunks={chunks_found}, reranking={reranking_applied}")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ [RAG Metrics] Error al guardar: {type(e).__name__}: {str(e)}")
        # No propagar error - metrics son best-effort
        
    finally:
        cursor.close()
        return_db_connection(conn)

