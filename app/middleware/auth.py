from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings


security = HTTPBearer()


async def verify_token(credentials: HTTPAuthorizationCredentials = None):
    """
    Middleware para validar AI_SERVICE_TOKEN en header Authorization.
    
    Expected format: Authorization: Bearer <token>
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header"
        )
    
    if credentials.credentials != settings.ai_service_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    return True
