"""
Funciones helper para tracking de analytics en AI executions.
"""
from typing import Dict, Any, Optional
from datetime import datetime
from app.db.database import get_db


def calculate_cost(tokens_used: int, model: str) -> float:
    """
    Calcular costo estimado basado en tokens y modelo.
    
    Pricing de OpenAI (Enero 2025):
    - gpt-5-mini: $0.15/1M input tokens, $0.60/1M output tokens (promedio: $0.375/1M)
    - gpt-5-nano: $0.05/1M input tokens, $0.20/1M output tokens (promedio: $0.125/1M)
    - gpt-4o: $2.50/1M input tokens, $10/1M output tokens (promedio: $6.25/1M)
    - gpt-4o-mini: $0.15/1M input tokens, $0.60/1M output tokens (promedio: $0.375/1M)
    """
    # Mapeo de costos por millón de tokens (promedio input/output)
    costs_per_million = {
        'gpt-5-mini': 0.375,
        'gpt-5-nano': 0.125,
        'gpt-4o': 6.25,
        'gpt-4o-mini': 0.375,
        'claude-sonnet-4-20250514': 3.0,  # Estimado
        'claude-3.5-sonnet': 3.0,  # Estimado
    }
    
    # Default para modelos desconocidos
    cost_per_million = costs_per_million.get(model, 0.375)
    
    # Calcular costo en USD
    return (tokens_used / 1_000_000) * cost_per_million


def save_tool_execution(
    execution_id: str,
    tool_name: str,
    duration_ms: int,
    success: bool,
    error: Optional[str] = None,
    request_data: Optional[Dict[str, Any]] = None,
    response_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Guardar ejecución de un tool en la DB.
    
    Args:
        execution_id: ID de la ejecución del agente
        tool_name: Nombre del tool ejecutado
        duration_ms: Duración en milisegundos
        success: Si la ejecución fue exitosa
        error: Mensaje de error (si falló)
        request_data: Datos de la request (opcional)
        response_data: Datos de la response (opcional)
    """
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ai.tool_executions (
                    execution_id,
                    tool_name,
                    executed_at,
                    duration_ms,
                    success,
                    error,
                    request_data,
                    response_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                execution_id,
                tool_name,
                datetime.now(),
                duration_ms,
                success,
                error,
                request_data,
                response_data
            ))
            
            conn.commit()
            cursor.close()
            
            status = "✅" if success else "❌"
            print(f"{status} Tool execution logged: {tool_name} ({duration_ms}ms)")
            
    except Exception as e:
        print(f"Error logging tool execution: {e}")

