from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings


security = HTTPBearer()


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Middleware para validar AI_SERVICE_TOKEN en header Authorization.
    
    Expected format: Authorization: Bearer <token>
    """
    received_token = credentials.credentials
    expected_token = settings.ai_service_token
    
    # Debug logging (solo primeros/últimos 8 chars por seguridad)
    print(f"🔐 [AUTH] Token recibido: {received_token[:8]}...{received_token[-8:]}")
    print(f"🔐 [AUTH] Token esperado: {expected_token[:8]}...{expected_token[-8:]}")
    print(f"🔐 [AUTH] Match: {received_token == expected_token}")
    
    if received_token != expected_token:
        print(f"❌ [AUTH] Token inválido - rechazando request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    print(f"✅ [AUTH] Token válido - autorizando request")
    return True
