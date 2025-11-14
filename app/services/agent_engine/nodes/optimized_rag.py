"""
Optimized RAG Node - Sprint 2

Features:
- Multi-query expansion (1-3 queries paralelas)
- Parallel hybrid search con asyncio.gather()
- Reranking con Groq gpt-oss-20b (reasoning low)
- Relevance validation (threshold filtering)
- Fallback a semantic-only si hybrid falla
- Métricas detalladas en ai.rag_metrics
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
from app.services.knowledge_base import KnowledgeBase
from app.services.agent_engine.analytics_tracking import save_tool_execution
from app.services.agent_engine.rag_metrics import save_rag_metrics
from app.services.agent_engine.llm_factory import create_groq_client
from app.services.llm_tracker import LLMCallTracker


async def generate_search_queries(
    original_query: str,
    kb_search_strategy: str,
    business_id: str
) -> List[str]:
    """
    Generar múltiples queries optimizadas según estrategia.
    
    Estrategias:
    - 'exact': 1 query (solo original)
    - 'broad': 2 queries (original + paráfrasis amplia)
    - 'multi_query': 3 queries (original + 2 variaciones)
    
    Usa Groq gpt-oss-20b con reasoning low (rápido y barato).
    """
    # Exact: solo query original
    if kb_search_strategy == 'exact':
        return [original_query]
    
    # Broad o multi_query: generar variaciones con LLM
    num_variations = 1 if kb_search_strategy == 'broad' else 2
    
    client = create_groq_client()
    
    system_prompt = """Eres un experto en reformular preguntas para búsqueda semántica.
Genera variaciones de la pregunta original enfocándote en:
- Sinónimos y términos relacionados
- Ejemplos específicos
- Conceptos más amplios o más específicos

Retorna SOLO un JSON con el formato: {"queries": ["query1", "query2"]}"""

    user_prompt = f"""Pregunta original: "{original_query}"

Genera {num_variations} variación(es) alternativa(s) de esta pregunta.
Las variaciones deben buscar la misma información pero con diferentes palabras."""

    try:
        async with LLMCallTracker(
            business_id=business_id,
            operation_type='multi_query_expansion',
            provider='groq',
            model='openai/gpt-oss-20b',
            operation_context={'original_query': original_query, 'strategy': kb_search_strategy}
        ) as tracker:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                reasoning={"effort": "low"},
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500
            )
            
            tracker.record(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
        
        result = json.loads(response.choices[0].message.content)
        variations = result.get('queries', [])
        
        # Siempre incluir original + variaciones
        all_queries = [original_query] + variations[:num_variations]
        
        print(f"🔍 [Multi-Query] Generadas {len(all_queries)} queries:")
        for i, q in enumerate(all_queries):
            print(f"  {i+1}. {q}")
        
        return all_queries
        
    except Exception as e:
        print(f"⚠️ [Multi-Query] Error generando queries: {type(e).__name__}: {str(e)}")
        # Fallback: solo query original
        return [original_query]


async def multi_query_search(
    kb: KnowledgeBase,
    business_id: str,
    queries: List[str],
    k: int,
    semantic_weight: float,
    keyword_weight: float,
    threshold: float
) -> List[Dict[str, Any]]:
    """
    Ejecutar búsquedas en paralelo para múltiples queries.
    Merge results y deduplicar por (document_id, chunk_index).
    Mantener el mejor combined_score por chunk.
    """
    # Ejecutar todas las búsquedas en paralelo
    search_tasks = [
        kb.hybrid_search(
            business_id=business_id,
            query=query,
            k=k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            threshold=threshold
        )
        for query in queries
    ]
    
    results_per_query = await asyncio.gather(*search_tasks, return_exceptions=True)
    
    # Merge y dedup por (document_id, chunk_index)
    chunks_by_id = {}  # key: (document_id, chunk_index), value: chunk con mejor score
    
    for i, result in enumerate(results_per_query):
        if isinstance(result, Exception):
            print(f"⚠️ [Multi-Query] Error en query {i+1}: {str(result)}")
            continue
        
        for chunk in result:
            chunk_key = (chunk['document_id'], chunk['chunk_index'])
            
            # Si ya existe, mantener el de mejor combined_score
            if chunk_key in chunks_by_id:
                if chunk['combined_score'] > chunks_by_id[chunk_key]['combined_score']:
                    chunks_by_id[chunk_key] = chunk
            else:
                chunks_by_id[chunk_key] = chunk
    
    # Convertir a lista y ordenar por combined_score
    merged_chunks = list(chunks_by_id.values())
    merged_chunks.sort(key=lambda x: x['combined_score'], reverse=True)
    
    print(f"📦 [Multi-Query] Merged {len(merged_chunks)} chunks únicos (de {sum(len(r) if not isinstance(r, Exception) else 0 for r in results_per_query)} totales)")
    
    return merged_chunks


async def rerank_results(
    original_query: str,
    chunks: List[Dict[str, Any]],
    business_id: str,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Reranking con Groq gpt-oss-20b (reasoning low).
    
    Input: chunks con combined_score (hybrid search)
    Output: chunks reordenados con rerank_score
    
    Solo rerankeamos top-10 para ahorrar tokens.
    """
    if not chunks:
        return []
    
    # Limitar a top-10 para reranking (ahorro de tokens)
    chunks_to_rerank = chunks[:10]
    
    client = create_groq_client()
    
    # Construir prompt con documentos numerados
    docs_text = "\n\n".join([
        f"{i+1}. {chunk['content'][:300]}..."  # Primeros 300 chars
        for i, chunk in enumerate(chunks_to_rerank)
    ])
    
    system_prompt = """Eres un experto en evaluar relevancia de documentos.
Evalúa qué tan relevante es cada documento para responder la pregunta del usuario.

Retorna un JSON con scores de 0.0 (nada relevante) a 1.0 (muy relevante).
Formato: {"scores": [0.85, 0.62, 0.91, ...]}"""

    user_prompt = f"""Pregunta: "{original_query}"

Documentos:
{docs_text}

Evalúa la relevancia de cada documento (1-{len(chunks_to_rerank)}) para esta pregunta."""

    try:
        async with LLMCallTracker(
            business_id=business_id,
            operation_type='reranking',
            provider='groq',
            model='openai/gpt-oss-20b',
            operation_context={'original_query': original_query, 'chunks_count': len(chunks_to_rerank)}
        ) as tracker:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                reasoning={"effort": "low"},
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=300
            )
            
            tracker.record(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )
        
        result = json.loads(response.choices[0].message.content)
        scores = result.get('scores', [])
        
        # Validar que tenemos scores para todos los chunks
        if len(scores) != len(chunks_to_rerank):
            print(f"⚠️ [Reranking] Mismatch: {len(scores)} scores vs {len(chunks_to_rerank)} chunks")
            # Rellenar con scores conservadores si faltan
            scores = scores + [0.5] * (len(chunks_to_rerank) - len(scores))
        
        # Agregar rerank_score a cada chunk
        for i, chunk in enumerate(chunks_to_rerank):
            chunk['rerank_score'] = float(scores[i])
        
        # Reordenar por rerank_score
        chunks_to_rerank.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        print(f"🔁 [Reranking] Top-3 después de reranking:")
        for i, chunk in enumerate(chunks_to_rerank[:3]):
            combined = chunk['combined_score']
            rerank = chunk['rerank_score']
            preview = chunk['content'][:60].replace('\n', ' ')
            print(f"  #{i+1}: combined={combined:.3f} rerank={rerank:.3f}")
            print(f"       \"{preview}...\"")
        
        # Retornar top_n después de reranking
        return chunks_to_rerank[:top_n]
        
    except Exception as e:
        print(f"❌ [Reranking] Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback: retornar sin reranking
        return chunks_to_rerank[:top_n]


def validate_relevance(
    chunks: List[Dict[str, Any]],
    combined_threshold: float = 0.4,
    rerank_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Validar relevancia final de chunks.
    
    Rechaza chunks si:
    - combined_score < 0.4 (muy bajo en hybrid search)
    - rerank_score < 0.5 (LLM dice que no es relevante)
    
    Returns: Solo chunks que pasan validación
    """
    if not chunks:
        return []
    
    validated = []
    rejected = []
    
    for chunk in chunks:
        combined = chunk.get('combined_score', 0)
        rerank = chunk.get('rerank_score', 1.0)  # Si no hay rerank, no filtrar por eso
        
        # Validar thresholds
        if combined >= combined_threshold and rerank >= rerank_threshold:
            validated.append(chunk)
        else:
            rejected.append({
                'combined': combined,
                'rerank': rerank,
                'preview': chunk['content'][:50]
            })
    
    if rejected:
        print(f"⚠️ [Validation] Rechazados {len(rejected)} chunks:")
        for r in rejected[:3]:  # Mostrar primeros 3
            print(f"  - combined={r['combined']:.3f} rerank={r['rerank']:.3f}: \"{r['preview']}...\"")
    
    print(f"✅ [Validation] Aprobados {len(validated)}/{len(chunks)} chunks")
    
    return validated


async def optimized_rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo RAG optimizado con multi-query + reranking + validation.
    
    Flujo:
    1. Generar queries (1-3 según strategy)
    2. Multi-query search en paralelo
    3. Merge y dedup results
    4. Si >5 chunks: aplicar reranking
    5. Validar relevance final
    6. Guardar métricas
    7. Fallback a semantic-only si 0 resultados
    """
    start_time = datetime.now()
    execution_id = state.get('execution_id')
    business_id = state['business_id']
    
    # Obtener último mensaje del usuario
    human_messages = [m for m in state['messages'] if m.type == 'human']
    
    if not human_messages:
        return {
            'nodes_visited': state.get('nodes_visited', []) + ['optimized_rag']
        }
    
    last_user_message = human_messages[-1]
    original_query = last_user_message.content
    
    # Obtener estrategia del orchestrator
    kb_search_strategy = state.get('kb_search_strategy', 'broad')
    confidence = state.get('confidence', 0.5)
    
    # Threshold adaptativo según confidence
    if confidence > 0.85:
        threshold = 0.3
    elif confidence > 0.7:
        threshold = 0.35
    else:
        threshold = 0.4
    
    print(f"🎯 [Optimized RAG] strategy={kb_search_strategy}, threshold={threshold} (confidence={confidence:.2f})")
    
    kb = KnowledgeBase()
    retrieved_docs = []
    queries_generated = []
    chunks_found = 0
    chunks_after_reranking = None
    reranking_applied = False
    relevance_validation_passed = None
    search_duration_ms = 0
    reranking_duration_ms = None
    
    try:
        # 1. Generar queries
        query_start = datetime.now()
        queries_generated = await generate_search_queries(
            original_query=original_query,
            kb_search_strategy=kb_search_strategy,
            business_id=business_id
        )
        
        # 2. Multi-query search en paralelo
        search_start = datetime.now()
        chunks = await multi_query_search(
            kb=kb,
            business_id=business_id,
            queries=queries_generated,
            k=5,
            semantic_weight=0.6,
            keyword_weight=0.4,
            threshold=threshold
        )
        search_duration_ms = int((datetime.now() - search_start).total_seconds() * 1000)
        
        chunks_found = len(chunks)
        print(f"📚 [Optimized RAG] Encontrados {chunks_found} chunks (multi-query)")
        
        # 3. Reranking si hay suficientes chunks
        if chunks_found >= 5:
            rerank_start = datetime.now()
            chunks = await rerank_results(
                original_query=original_query,
                chunks=chunks,
                business_id=business_id,
                top_n=5
            )
            reranking_duration_ms = int((datetime.now() - rerank_start).total_seconds() * 1000)
            reranking_applied = True
            chunks_after_reranking = len(chunks)
        
        # 4. Validar relevance
        validated_chunks = validate_relevance(
            chunks=chunks,
            combined_threshold=0.4,
            rerank_threshold=0.5
        )
        
        relevance_validation_passed = len(validated_chunks) > 0
        retrieved_docs = [chunk['content'] for chunk in validated_chunks]
        
        # 5. Fallback: Si 0 docs, retry con semantic-only
        if len(retrieved_docs) == 0 and threshold > 0.2:
            print(f"🔄 [Optimized RAG] Fallback: 0 docs, retry con semantic-only threshold 0.2")
            
            fallback_start = datetime.now()
            fallback_results = await kb.search(
                business_id=business_id,
                query=original_query,
                k=3,
                threshold=0.2
            )
            search_duration_ms += int((datetime.now() - fallback_start).total_seconds() * 1000)
            
            retrieved_docs = [doc['content'] for doc in fallback_results]
            chunks_found += len(fallback_results)
            print(f"📚 Fallback retrieved {len(retrieved_docs)} docs (semantic-only)")
        
        # Log tool execution para backward compatibility
        if execution_id:
            total_duration = int((datetime.now() - start_time).total_seconds() * 1000)
            save_tool_execution(
                execution_id=execution_id,
                tool_name='optimized_rag_multi_query',
                duration_ms=total_duration,
                success=True,
                request_data={
                    'original_query': original_query,
                    'queries_generated': queries_generated,
                    'strategy': kb_search_strategy,
                    'threshold': threshold
                },
                response_data={
                    'chunks_found': chunks_found,
                    'chunks_after_reranking': chunks_after_reranking,
                    'reranking_applied': reranking_applied,
                    'validation_passed': relevance_validation_passed,
                    'results_count': len(retrieved_docs)
                }
            )
        
    except Exception as e:
        print(f"❌ [Optimized RAG] Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Log failed execution
        if execution_id:
            total_duration = int((datetime.now() - start_time).total_seconds() * 1000)
            save_tool_execution(
                execution_id=execution_id,
                tool_name='optimized_rag_multi_query',
                duration_ms=total_duration,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                request_data={'original_query': original_query}
            )
    
    # Guardar métricas RAG (siempre, incluso si falló)
    if execution_id:
        total_duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        save_rag_metrics(
            execution_id=execution_id,
            business_id=business_id,
            original_query=original_query,
            queries_generated=queries_generated,
            search_strategy='multi_query' if len(queries_generated) > 1 else 'hybrid',
            semantic_weight=0.6,
            keyword_weight=0.4,
            threshold_used=threshold,
            chunks_found=chunks_found,
            chunks_after_reranking=chunks_after_reranking,
            reranking_applied=reranking_applied,
            relevance_validation_passed=relevance_validation_passed,
            search_duration_ms=search_duration_ms,
            reranking_duration_ms=reranking_duration_ms,
            total_duration_ms=total_duration_ms
        )
    
    return {
        'retrieved_docs': retrieved_docs if retrieved_docs else None,
        'nodes_visited': state.get('nodes_visited', []) + ['optimized_rag']
    }

