import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import time
from app.config import settings


# Connection pool global (lazy initialization)
_connection_pool = None
_pool_lock = False  # Simple flag para evitar race conditions

def get_connection_pool():
    """
    Obtener o crear el connection pool con retry logic.
    
    Pool size: 2-10 conexiones (optimizado para FastAPI async)
    - minconn=2: Conexiones iniciales (reduce latencia de startup)
    - maxconn=10: Máximo suficiente para carga normal
    
    Retry logic: 3 intentos con backoff exponencial para manejar
    problemas transitorios de DNS/red al inicio del contenedor.
    """
    global _connection_pool, _pool_lock
    
    if _connection_pool is None:
        # Evitar múltiples threads creando el pool simultáneamente
        if _pool_lock:
            time.sleep(0.1)  # Esperar a que otro thread complete
            return _connection_pool
        
        _pool_lock = True
        
        # Retry logic para crear el pool (máximo 3 intentos)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                _connection_pool = pool.SimpleConnectionPool(
                    minconn=2,      # Solo 2 conexiones iniciales (startup rápido)
                    maxconn=10,     # Máximo 10 conexiones (suficiente para async)
                    dsn=settings.database_url,
                    cursor_factory=RealDictCursor,
                    options="-c search_path=ai,public"
                )
                print(f"✅ Connection pool creado exitosamente (attempt {attempt})")
                break
            except Exception as e:
                print(f"⚠️ Error creando connection pool (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Backoff exponencial: 2s, 4s
                    print(f"🔄 Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ No se pudo crear connection pool después de {max_retries} intentos")
                    raise
        
        _pool_lock = False
    
    return _connection_pool


def get_db_connection():
    """
    Obtener conexión del pool (rápido, ~1ms vs ~100-500ms sin pool).
    
    search_path = "ai,public" permite:
    - Queries sin prefijo resuelven primero en schema ai
    - Fallback a schema public para tablas compartidas
    - Foreign keys cross-schema funcionan automáticamente
    """
    pool_instance = get_connection_pool()
    conn = pool_instance.getconn()
    return conn


def return_db_connection(conn):
    """
    Retornar conexión al pool (NO cerrarla).
    """
    pool_instance = get_connection_pool()
    pool_instance.putconn(conn)


@contextmanager
def get_db():
    """Context manager para conexiones de BD (usa pool)"""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        return_db_connection(conn)  # Retornar al pool, no cerrar
