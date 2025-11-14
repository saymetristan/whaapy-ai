import os
import json
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.db.database import get_db_connection, return_db_connection

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
                # Generar embedding
                embedding = await self.embeddings.aembed_query(chunk)
                
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
        
        # 1. Generar embedding de la query
        query_embedding = await self.embeddings.aembed_query(query)
        
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
                    1 - (embedding <=> %s::ai.vector) as similarity
                FROM ai.documents_embeddings
                WHERE business_id = %s
                {doc_filter}
                ORDER BY embedding <=> %s::ai.vector
                LIMIT %s
            """
            
            # Agregar query_embedding_str una segunda vez para el ORDER BY
            params_with_order = [params[0], params[1]]  # embedding, business_id
            if document_ids and len(document_ids) > 0:
                params_with_order.extend(document_ids)
            params_with_order.extend([params[0], params[2]])  # embedding para ORDER BY, limit
            
            cursor.execute(query_sql, params_with_order)
            
            results = cursor.fetchall()
            
            # Filtrar por threshold
            filtered_results = [
                {
                    "id": str(row[0]),
                    "document_id": str(row[1]),
                    "chunk_index": row[2],
                    "content": row[3],
                    "metadata": row[4],
                    "similarity": float(row[5])
                }
                for row in results
                if float(row[5]) >= threshold
            ]
            
            print(f"✅ [KB] Encontrados {len(filtered_results)}/{len(results)} chunks (threshold={threshold})")
            if filtered_results:
                top_similarity = max(r['similarity'] for r in filtered_results)
                print(f"📈 [KB] Top similarity: {top_similarity:.3f}")
            
            return filtered_results
        
        except Exception as e:
            raise e
        finally:
            cursor.close()
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

