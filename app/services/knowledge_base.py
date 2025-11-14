import os
import json
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.db.database import get_db_connection, return_db_connection
from app.services.llm_tracker import LLMCallTracker, estimate_embedding_tokens

# Configuración de embeddings (text-embedding-3-small para compatibilidad)
EMBEDDINGS_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Configuración de chunking optimizada
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class KnowledgeBase:
    def __init__(self):
        # OpenAI Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDINGS_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def add_document(
        self,
        business_id: str,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Agregar documento a la knowledge base
        
        1. Chunking del contenido
        2. Generar embeddings con OpenAI
        3. Guardar en ai.documents_embeddings
        """
        # 1. Split en chunks
        chunks = self.text_splitter.split_text(content)
        
        if not chunks:
            raise ValueError("No se pudo extraer contenido del documento")
        
        print(f"📦 Documento dividido en {len(chunks)} chunks")
        
        # 2. Preparar metadata base
        base_metadata = {
            "business_id": business_id,
            "document_id": document_id,
            "total_chunks": len(chunks),
            **(metadata or {})
        }
        
        # 3. Generar embeddings para cada chunk
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            for idx, chunk in enumerate(chunks):
                # Generar embedding + tracking
                async with LLMCallTracker(
                    business_id=business_id,
                    operation_type='embedding',
                    provider='openai',
                    model=EMBEDDINGS_MODEL,
                    operation_context={
                        'operation': 'add_document',
                        'document_id': document_id,
                        'chunk_index': idx,
                        'total_chunks': len(chunks)
                    }
                ) as tracker:
                embedding = await self.embeddings.aembed_query(chunk)
                    
                    # Estimar tokens
                    estimated_tokens = estimate_embedding_tokens(chunk)
                    tracker.record(input_tokens=estimated_tokens, output_tokens=0)
                
                # Metadata específico del chunk
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": idx,
                    "chunk_size": len(chunk)
                }
                
                # Insertar en DB
                # Convertir embedding a formato vector de PostgreSQL
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                metadata_json = json.dumps(chunk_metadata)
                
                cursor.execute(
                    """
                    INSERT INTO ai.documents_embeddings 
                    (business_id, document_id, chunk_index, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s::ai.vector, %s)
                    """,
                    (
                        business_id,
                        document_id,
                        idx,
                        chunk,
                        embedding_str,
                        metadata_json
                    )
                )
                
                print(f"✅ Chunk {idx + 1}/{len(chunks)} embedido")
            
            conn.commit()
            
            print(f"🎉 Documento {document_id} procesado: {len(chunks)} chunks")
            
            return {
                "document_id": document_id,
                "chunks_created": len(chunks),
                "total_size": len(content)
            }
        
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            return_db_connection(conn)
    
    async def delete_document(self, document_id: str) -> None:
        """Eliminar todos los embeddings de un documento"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "DELETE FROM ai.documents_embeddings WHERE document_id = %s",
                (document_id,)
            )
            conn.commit()
            
            print(f"🗑️ Embeddings del documento {document_id} eliminados")
        
        finally:
            cursor.close()
            return_db_connection(conn)
    
    async def search(
        self,
        business_id: str,
        query: str,
        k: int = 5,
        threshold: float = 0.7,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda semántica en la knowledge base
        
        Returns: Lista de chunks relevantes con similarity scores
        """
        import time
        search_start = time.time()
        
        # 0. Quick check: si no hay documentos con embeddings, retornar vacío
        print(f"🔍 [KB] Buscando en business_id={business_id}, query='{query[:50]}...'")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT COUNT(*) as count FROM ai.documents_embeddings WHERE business_id = %s AND embedding IS NOT NULL",
                (business_id,)
            )
            result = cursor.fetchone()
            count = result['count'] if result else 0
            
            print(f"📊 [KB] Found {count} chunks con embeddings para business {business_id}")
            
            if count == 0:
                print(f"⚠️ [KB] Retornando vacío - no hay documentos")
                return []
        finally:
            cursor.close()
            return_db_connection(conn)
        
        # 1. Generar embedding de la query + tracking
        embed_start = time.time()
        
        async with LLMCallTracker(
            business_id=business_id,
            operation_type='embedding',
            provider='openai',
            model=EMBEDDINGS_MODEL,
            operation_context={'operation': 'search_query', 'query_length': len(query)}
        ) as tracker:
        query_embedding = await self.embeddings.aembed_query(query)
            
            # Embeddings: estimar tokens (1 token ≈ 4 chars)
            estimated_tokens = estimate_embedding_tokens(query)
            tracker.record(input_tokens=estimated_tokens, output_tokens=0)
        
        embed_time = (time.time() - embed_start) * 1000
        print(f"⏱️ [KB] Embedding generado en {embed_time:.0f}ms")
        
        # 2. Convertir embedding a formato string para PostgreSQL
        query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # 3. Buscar usando pgvector similarity search directo
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Construir WHERE clause para document_ids si se especifica
            doc_filter = ""
            params = [query_embedding_str, business_id, k]
            
            if document_ids and len(document_ids) > 0:
                placeholders = ','.join(['%s'] * len(document_ids))
                doc_filter = f"AND document_id IN ({placeholders})"
                params.extend(document_ids)
            
            query_sql = f"""
                SELECT 
                    id,
                    document_id,
                    chunk_index,
                    content,
                    metadata,
                    1 - (embedding OPERATOR(ai.<=>) %s::ai.vector) as similarity
                FROM ai.documents_embeddings
                WHERE business_id = %s
                {doc_filter}
                ORDER BY embedding OPERATOR(ai.<=>) %s::ai.vector
                LIMIT %s
            """
            
            # Agregar query_embedding_str una segunda vez para el ORDER BY
            params_with_order = [params[0], params[1]]  # embedding, business_id
            if document_ids and len(document_ids) > 0:
                params_with_order.extend(document_ids)
            params_with_order.extend([params[0], params[2]])  # embedding para ORDER BY, limit
            
            query_start = time.time()
            cursor.execute(query_sql, params_with_order)
            results = cursor.fetchall()
            query_time = (time.time() - query_start) * 1000
            print(f"⏱️ [KB] Query SQL ejecutada en {query_time:.0f}ms ({len(results)} resultados)")
            
            # Log todas las similarities antes de filtrar
            if results:
                similarities = [float(row['similarity']) for row in results]
                print(f"📊 [KB] Similarities: {[f'{s:.3f}' for s in similarities[:5]]}")  # Top 5
                
                # Preview del contenido top 1 para debugging
                if len(results) > 0:
                    top_content = results[0]['content'][:100]
                    print(f"📄 [KB] Top result preview: {top_content}...")
            
            # Filtrar por threshold
            # RealDictCursor retorna dict, no tuplas
            filtered_results = [
                {
                    "id": str(row['id']),
                    "document_id": str(row['document_id']),
                    "chunk_index": row['chunk_index'],
                    "content": row['content'],
                    "metadata": row['metadata'],
                    "similarity": float(row['similarity'])
                }
                for row in results
                if float(row['similarity']) >= threshold
            ]
            
            total_time = (time.time() - search_start) * 1000
            print(f"✅ [KB] Encontrados {len(filtered_results)}/{len(results)} chunks (threshold={threshold})")
            if filtered_results:
                top_similarity = max(r['similarity'] for r in filtered_results)
                print(f"📈 [KB] Top similarity: {top_similarity:.3f}")
            print(f"⏱️ [KB] Búsqueda total: {total_time:.0f}ms (embed: {embed_time:.0f}ms, query: {query_time:.0f}ms)")
            
            return filtered_results
        
        except Exception as e:
            raise e
        finally:
            cursor.close()
            return_db_connection(conn)
    
    async def hybrid_search(
        self,
        business_id: str,
        query: str,
        k: int = 5,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: combina semantic (embeddings) + keyword (full-text).
        
        Args:
            business_id: UUID del negocio
            query: Query de búsqueda
            k: Número máximo de resultados
            semantic_weight: Peso para cosine similarity (default 0.7)
            keyword_weight: Peso para keyword match (default 0.3)
            threshold: Threshold mínimo para combined_score (default 0.3)
        
        Returns:
            Lista de chunks ordenados por combined_score descendente
            Cada chunk incluye: semantic_score, keyword_score, combined_score
        """
        import time
        search_start = time.time()
        
        # 1. Generar embedding para semantic search
        embed_start = time.time()
        
        async with LLMCallTracker(
            business_id=business_id,
            operation_type='embedding',
            provider='openai',
            model=EMBEDDINGS_MODEL,
            operation_context={'operation': 'hybrid_search_query', 'query_length': len(query)}
        ) as tracker:
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Embeddings: estimar tokens (1 token ≈ 4 chars)
            estimated_tokens = estimate_embedding_tokens(query)
            tracker.record(input_tokens=estimated_tokens, output_tokens=0)
        
        query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        embed_time = (time.time() - embed_start) * 1000
        print(f"⏱️ [KB] Embedding generado en {embed_time:.0f}ms")
        
        # 2. Ejecutar hybrid query
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                # Query híbrido: LEFT JOIN semantic + keyword scores
                query_sql = """
                    WITH semantic_scores AS (
                        SELECT 
                            id,
                            document_id,
                            chunk_index,
                            content,
                            metadata,
                            1 - (embedding OPERATOR(ai.<=>) %s::ai.vector) as semantic_score
                        FROM ai.documents_embeddings
                        WHERE business_id = %s
                          AND embedding IS NOT NULL
                    ),
                    keyword_scores AS (
                        SELECT
                            id,
                            ts_rank(content_tsvector, plainto_tsquery('spanish', %s)) as keyword_score
                        FROM ai.documents_embeddings
                        WHERE business_id = %s
                          AND content_tsvector @@ plainto_tsquery('spanish', %s)
                    )
                    SELECT 
                        s.id,
                        s.document_id,
                        s.chunk_index,
                        s.content,
                        s.metadata,
                        s.semantic_score,
                        COALESCE(k.keyword_score, 0) as keyword_score,
                        (s.semantic_score * %s + COALESCE(k.keyword_score, 0) * %s) as combined_score
                    FROM semantic_scores s
                    LEFT JOIN keyword_scores k ON s.id = k.id
                    WHERE (s.semantic_score * %s + COALESCE(k.keyword_score, 0) * %s) >= %s
                    ORDER BY combined_score DESC
                    LIMIT %s
                """
                
                params = [
                    query_embedding_str, business_id,  # semantic search
                    query, business_id, query,          # keyword search (3x: rank + WHERE + WHERE)
                    semantic_weight, keyword_weight,    # pesos para combined_score
                    semantic_weight, keyword_weight,    # pesos para WHERE threshold
                    threshold,                          # threshold mínimo
                    k                                   # limit
                ]
                
                query_start = time.time()
                cursor.execute(query_sql, params)
                results = cursor.fetchall()
                query_time = (time.time() - query_start) * 1000
                
                print(f"⏱️ [KB] Hybrid query ejecutada en {query_time:.0f}ms ({len(results)} resultados)")
                
                # Logging de scores para debugging
                if results:
                    print(f"📊 [KB] Top 3 hybrid scores:")
                    for i, row in enumerate(results[:3]):
                        sem = float(row['semantic_score'])
                        kw = float(row['keyword_score'])
                        combined = float(row['combined_score'])
                        preview = row['content'][:60].replace('\n', ' ')
                        print(f"  #{i+1}: sem={sem:.3f} kw={kw:.3f} → combined={combined:.3f}")
                        print(f"       \"{preview}...\"")
                
                # Formatear resultados
                formatted_results = [
                    {
                        "id": str(row['id']),
                        "document_id": str(row['document_id']),
                        "chunk_index": row['chunk_index'],
                        "content": row['content'],
                        "metadata": row['metadata'] if row['metadata'] else {},
                        "semantic_score": float(row['semantic_score']),
                        "keyword_score": float(row['keyword_score']),
                        "combined_score": float(row['combined_score'])
                    }
                    for row in results
                ]
                
                total_time = (time.time() - search_start) * 1000
                print(f"✅ [KB] Hybrid search completada: {len(formatted_results)} chunks en {total_time:.0f}ms")
                
                return formatted_results
                
        finally:
            return_db_connection(conn)
    
    async def get_stats(self, business_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de embeddings del negocio"""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT 
                    total_documents,
                    total_chunks,
                    avg_chunk_chars,
                    last_embedding_created
                FROM ai.embeddings_stats
                WHERE business_id = %s
                """,
                (business_id,)
            )
            
            row = cursor.fetchone()
            
            if not row:
                return {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "avg_chunk_chars": 0,
                    "last_embedding_created": None
                }
            
            return {
                "total_documents": row[0],
                "total_chunks": row[1],
                "avg_chunk_chars": row[2],
                "last_embedding_created": row[3].isoformat() if row[3] else None
            }
        
        finally:
            cursor.close()
            return_db_connection(conn)

