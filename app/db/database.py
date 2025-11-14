import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from app.config import settings


# Connection pool global (creado al importar el módulo)
_connection_pool = None

def get_connection_pool():
    """
    Obtener o crear el connection pool.
    Pool size: 5-20 conexiones (suficiente para FastAPI con múltiples workers)
    """
    global _connection_pool
    
    if _connection_pool is None:
        _connection_pool = pool.SimpleConnectionPool(
            minconn=5,      # Mínimo 5 conexiones siempre abiertas
            maxconn=20,     # Máximo 20 conexiones concurrentes
            dsn=settings.database_url,
            cursor_factory=RealDictCursor,
            options="-c search_path=ai,public"
        )
    
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
