from fastapi import APIRouter
from app.db.database import get_db
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint que verifica:
    - Servicio está corriendo
    - DB connection está OK
    """
    try:
        # Verificar conexión a DB
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
        
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "service": "whaapy-ai",
        "version": "1.0.0",
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status
    }
