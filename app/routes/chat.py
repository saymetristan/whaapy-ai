from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.middleware.auth import verify_token
from app.services.agent_config import AgentConfigManager
from app.services.agent_engine.engine import AgentEngine


router = APIRouter()


class ChatRequest(BaseModel):
    """Schema para request de chat"""
    business_id: str
    conversation_id: str
    customer_phone: str
    message: str
    customer_name: Optional[str] = None


class ChatResponse(BaseModel):
    """Schema para response de chat"""
    response: str
    metadata: dict


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    _: bool = Depends(verify_token)
):
    """
    Endpoint principal de chat del agente.
    
    Procesa un mensaje del cliente usando el motor LangGraph y retorna
    la respuesta generada junto con metadata de la ejecución.
    
    Args:
        body: Datos del mensaje (business_id, conversation_id, message, etc)
    
    Returns:
        ChatResponse con la respuesta del agente y metadata
    
    Raises:
        HTTPException: 404 si no hay config del agente, 500 en error interno
    """
    try:
        # 1. Obtener configuración del agente
        config_manager = AgentConfigManager()
        agent_config = config_manager.get_config(body.business_id)
        
        if not agent_config:
            raise HTTPException(
                status_code=404,
                detail="Agent config not found for business"
            )
        
        # Verificar que el agente esté habilitado
        if not agent_config.get('enabled', True):
            raise HTTPException(
                status_code=403,
                detail="Agent is disabled for this business"
            )
        
        # 2. Crear instancia del AgentEngine
        engine = AgentEngine(agent_config)
        
        # 3. Procesar mensaje
        result = await engine.chat(
            business_id=body.business_id,
            conversation_id=body.conversation_id,
            customer_phone=body.customer_phone,
            message=body.message,
            customer_name=body.customer_name
        )
        
        # 4. Retornar respuesta
        return ChatResponse(
            response=result['response'],
            metadata=result['metadata']
        )
    
    except HTTPException:
        # Re-lanzar HTTPExceptions tal cual
        raise
    
    except Exception as e:
        # Log del error (en producción usaríamos logger apropiado)
        print(f"❌ Error en /ai/chat: {e}")
        
        # Retornar error genérico al cliente
        raise HTTPException(
            status_code=500,
            detail="Internal error processing message"
        )


@router.get("/chat/health")
async def chat_health():
    """Health check para el servicio de chat"""
    return {
        "status": "healthy",
        "service": "chat",
        "version": "1.0.0"
    }

