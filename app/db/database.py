import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from app.config import settings


def get_db_connection():
    """
    Crear conexión a PostgreSQL con search_path configurado.
    
    search_path = "ai,public" permite:
    - Queries sin prefijo resuelven primero en schema ai
    - Fallback a schema public para tablas compartidas
    - Foreign keys cross-schema funcionan automáticamente
    """
    conn = psycopg2.connect(
        settings.database_url,
        cursor_factory=RealDictCursor,
        options="-c search_path=ai,public"
    )
    return conn


@contextmanager
def get_db():
    """Context manager para conexiones de BD"""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
