import os
import json
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.db.database import get_db_connection

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
                
                # No usar cast explícito, dejar que PostgreSQL lo resuelva automáticamente
                cursor.execute(
                    """
                    INSERT INTO ai.documents_embeddings 
                    (business_id, document_id, chunk_index, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
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
            conn.close()
    
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
            conn.close()
    
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
        # 1. Generar embedding de la query
        query_embedding = await self.embeddings.aembed_query(query)
        
        # 2. Convertir embedding a formato string para PostgreSQL
        query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # 3. Buscar usando función ai.match_documents
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                """
                SELECT id, document_id, chunk_index, content, metadata, similarity
                FROM ai.match_documents(
                    %s::ai.vector,
                    %s::double precision,
                    %s::integer,
                    %s::uuid,
                    %s::uuid[]
                )
                """,
                (
                    query_embedding_str,
                    threshold,
                    k,
                    business_id,
                    document_ids
                )
            )
            
            results = cursor.fetchall()
            
            return [
                {
                    "id": str(row[0]),
                    "document_id": str(row[1]),
                    "chunk_index": row[2],
                    "content": row[3],
                    "metadata": row[4],
                    "similarity": row[5]
                }
                for row in results
            ]
        
        finally:
            cursor.close()
            conn.close()
    
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
            conn.close()

