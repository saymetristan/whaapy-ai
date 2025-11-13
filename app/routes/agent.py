from fastapi import APIRouter, Depends, HTTPException
from app.services.agent_config import AgentConfigManager
from app.middleware.auth import verify_token
from pydantic import BaseModel
from typing import Optional


router = APIRouter()


class UpdateAgentConfigRequest(BaseModel):
    system_prompt: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    enabled: Optional[bool] = None


@router.get("/config/{business_id}")
async def get_agent_config(
    business_id: str,
    _: bool = Depends(verify_token)
):
    """Obtener configuración del agente"""
    try:
        config_manager = AgentConfigManager()
        config = config_manager.get_config(business_id)
        
        if not config:
            raise HTTPException(status_code=404, detail="Agent config not found")
        
        return {"data": config}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.put("/config/{business_id}")
async def update_agent_config(
    business_id: str,
    body: UpdateAgentConfigRequest,
    _: bool = Depends(verify_token)
):
    """Actualizar configuración del agente"""
    try:
        config_manager = AgentConfigManager()
        
        # Convertir a dict solo con campos presentes
        updates = body.model_dump(exclude_none=True)
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        config = config_manager.update_config(business_id, updates)
        
        return {"data": config}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
