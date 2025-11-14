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
    
    Pool size: 3-15 conexiones (optimizado para requests concurrentes)
    - minconn=3: Conexiones iniciales siempre disponibles
    - maxconn=15: Suficiente para múltiples requests simultáneos
    - statement_timeout=25s: Queries nunca bloquean >25s (fail fast)
    
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
                    minconn=3,      # 3 conexiones iniciales (balance startup/disponibilidad)
                    maxconn=15,     # Máximo 15 conexiones (para múltiples requests concurrentes)
                    dsn=settings.database_url,
                    cursor_factory=RealDictCursor,
                    options="-c search_path=ai,public -c statement_timeout=25000"  # 25s timeout
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
    
    Timeout: Si el pool está agotado, espera máximo 5s antes de fallar.
    """
    pool_instance = get_connection_pool()
    
    # getconn() con timeout (evita bloqueo infinito si pool agotado)
    max_wait = 5  # segundos
    start_time = time.time()
    
    while True:
        try:
            conn = pool_instance.getconn()
            return conn
        except pool.PoolError as e:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                print(f"❌ Pool exhausted después de {max_wait}s - no hay conexiones disponibles")
                raise Exception(f"Database pool exhausted (waited {max_wait}s)")
            
            # Pool temporalmente agotado, esperar un poco
            print(f"⚠️ Pool busy, esperando... ({elapsed:.1f}s elapsed)")
            time.sleep(0.1)  # Wait 100ms antes de reintentar


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
