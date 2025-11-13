from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.services.knowledge_base import KnowledgeBase
from app.middleware.auth import verify_token

router = APIRouter()
kb = KnowledgeBase()

class AddDocumentRequest(BaseModel):
    business_id: str
    document_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class DeleteDocumentRequest(BaseModel):
    document_id: str

class SearchRequest(BaseModel):
    business_id: str
    query: str
    k: Optional[int] = 5
    threshold: Optional[float] = 0.7
    document_ids: Optional[List[str]] = None

class GetStatsRequest(BaseModel):
    business_id: str

@router.post("/embeddings/add")
async def add_document(
    request: AddDocumentRequest,
    token: str = Depends(verify_token)
):
    """Agregar documento a la knowledge base"""
    try:
        result = await kb.add_document(
            business_id=request.business_id,
            document_id=request.document_id,
            content=request.content,
            metadata=request.metadata
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embeddings/delete")
async def delete_document(
    request: DeleteDocumentRequest,
    token: str = Depends(verify_token)
):
    """Eliminar embeddings de un documento"""
    try:
        await kb.delete_document(document_id=request.document_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embeddings/search")
async def search_knowledge_base(
    request: SearchRequest,
    token: str = Depends(verify_token)
):
    """Buscar en la knowledge base"""
    try:
        results = await kb.search(
            business_id=request.business_id,
            query=request.query,
            k=request.k,
            threshold=request.threshold,
            document_ids=request.document_ids
        )
        return {"success": True, "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embeddings/stats")
async def get_embeddings_stats(
    request: GetStatsRequest,
    token: str = Depends(verify_token)
):
    """Obtener estadísticas de embeddings"""
    try:
        stats = await kb.get_stats(business_id=request.business_id)
        return {"success": True, "data": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

