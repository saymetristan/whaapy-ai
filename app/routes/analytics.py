"""
Analytics endpoints para token usage y costos de LLM.

GET /ai/analytics/token-usage - Analytics detallado de uso de tokens y costos
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from app.middleware.auth import verify_token
from app.db.database import get_db


router = APIRouter()


@router.get("/analytics/token-usage")
async def get_token_usage(
    business_id: str = Query(..., description="ID del negocio"),
    start_date: Optional[str] = Query(None, description="Fecha inicio (ISO format: 2025-01-01)"),
    end_date: Optional[str] = Query(None, description="Fecha fin (ISO format: 2025-01-15)"),
    operation_type: Optional[str] = Query(None, description="Tipo de operación: chat, embedding, ocr, analyze_prompt, generate_suggestion"),
    group_by: str = Query('day', description="Agrupar por: hour, day, week, month, operation, model"),
    _: bool = Depends(verify_token)
):
    """
    Obtener analytics detallados de uso de tokens y costos.
    
    Query params:
        - business_id: ID del negocio (requerido)
        - start_date: Fecha inicio en formato ISO (ej: "2025-01-01")
        - end_date: Fecha fin en formato ISO (ej: "2025-01-15")
        - operation_type: Filtrar por tipo de operación
        - group_by: Cómo agrupar los datos (hour, day, week, month, operation, model)
    
    Returns:
        {
            "summary": {...},
            "breakdown": [...],
            "by_operation": {...},
            "by_model": {...}
        }
    """
    # Validar group_by
    valid_group_by = ['hour', 'day', 'week', 'month', 'operation', 'model']
    if group_by not in valid_group_by:
        raise HTTPException(status_code=400, detail=f"group_by debe ser uno de: {', '.join(valid_group_by)}")
    
    # Fechas por defecto: últimos 30 días
    if not start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Convertir a datetime
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date) + timedelta(days=1)  # Incluir el día completo
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de fecha inválido. Usar ISO format: YYYY-MM-DD")
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        try:
            # 1. Summary total
            summary_query = """
                SELECT 
                    COUNT(*) as total_calls,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(total_cost) as total_cost,
                    AVG(total_tokens) as avg_tokens_per_call,
                    AVG(duration_ms) as avg_duration_ms
                FROM ai.llm_calls
                WHERE business_id = %s
                    AND created_at >= %s
                    AND created_at < %s
                    AND (%s IS NULL OR operation_type = %s)
            """
            
            cursor.execute(summary_query, (business_id, start_dt, end_dt, operation_type, operation_type))
            summary_row = cursor.fetchone()
            
            summary = {
                'total_calls': summary_row['total_calls'] or 0,
                'total_input_tokens': summary_row['total_input_tokens'] or 0,
                'total_output_tokens': summary_row['total_output_tokens'] or 0,
                'total_tokens': summary_row['total_tokens'] or 0,
                'total_cost': float(summary_row['total_cost'] or 0),
                'avg_tokens_per_call': int(summary_row['avg_tokens_per_call'] or 0),
                'avg_duration_ms': int(summary_row['avg_duration_ms'] or 0),
                'date_range': {
                    'start': start_date,
                    'end': end_date
                }
            }
            
            # 2. Breakdown según group_by
            breakdown = []
            
            if group_by == 'hour':
                breakdown_query = """
                    SELECT 
                        DATE_TRUNC('hour', created_at) as period,
                        COUNT(*) as calls,
                        SUM(input_tokens) as input_tokens,
                        SUM(output_tokens) as output_tokens,
                        SUM(total_tokens) as total_tokens,
                        SUM(total_cost) as total_cost
                    FROM ai.llm_calls
                    WHERE business_id = %s
                        AND created_at >= %s
                        AND created_at < %s
                        AND (%s IS NULL OR operation_type = %s)
                    GROUP BY DATE_TRUNC('hour', created_at)
                    ORDER BY period DESC
                    LIMIT 100
                """
            elif group_by == 'day':
                breakdown_query = """
                    SELECT 
                        DATE(created_at) as period,
                        COUNT(*) as calls,
                        SUM(input_tokens) as input_tokens,
                        SUM(output_tokens) as output_tokens,
                        SUM(total_tokens) as total_tokens,
                        SUM(total_cost) as total_cost
                    FROM ai.llm_calls
                    WHERE business_id = %s
                        AND created_at >= %s
                        AND created_at < %s
                        AND (%s IS NULL OR operation_type = %s)
                    GROUP BY DATE(created_at)
                    ORDER BY period DESC
                """
            elif group_by == 'week':
                breakdown_query = """
                    SELECT 
                        DATE_TRUNC('week', created_at) as period,
                        COUNT(*) as calls,
                        SUM(input_tokens) as input_tokens,
                        SUM(output_tokens) as output_tokens,
                        SUM(total_tokens) as total_tokens,
                        SUM(total_cost) as total_cost
                    FROM ai.llm_calls
                    WHERE business_id = %s
                        AND created_at >= %s
                        AND created_at < %s
                        AND (%s IS NULL OR operation_type = %s)
                    GROUP BY DATE_TRUNC('week', created_at)
                    ORDER BY period DESC
                """
            elif group_by == 'month':
                breakdown_query = """
                    SELECT 
                        DATE_TRUNC('month', created_at) as period,
                        COUNT(*) as calls,
                        SUM(input_tokens) as input_tokens,
                        SUM(output_tokens) as output_tokens,
                        SUM(total_tokens) as total_tokens,
                        SUM(total_cost) as total_cost
                    FROM ai.llm_calls
                    WHERE business_id = %s
                        AND created_at >= %s
                        AND created_at < %s
                        AND (%s IS NULL OR operation_type = %s)
                    GROUP BY DATE_TRUNC('month', created_at)
                    ORDER BY period DESC
                """
            elif group_by == 'operation':
                breakdown_query = """
                    SELECT 
                        operation_type as period,
                        COUNT(*) as calls,
                        SUM(input_tokens) as input_tokens,
                        SUM(output_tokens) as output_tokens,
                        SUM(total_tokens) as total_tokens,
                        SUM(total_cost) as total_cost
                    FROM ai.llm_calls
                    WHERE business_id = %s
                        AND created_at >= %s
                        AND created_at < %s
                        AND (%s IS NULL OR operation_type = %s)
                    GROUP BY operation_type
                    ORDER BY total_tokens DESC
                """
            elif group_by == 'model':
                breakdown_query = """
                    SELECT 
                        model as period,
                        COUNT(*) as calls,
                        SUM(input_tokens) as input_tokens,
                        SUM(output_tokens) as output_tokens,
                        SUM(total_tokens) as total_tokens,
                        SUM(total_cost) as total_cost
                    FROM ai.llm_calls
                    WHERE business_id = %s
                        AND created_at >= %s
                        AND created_at < %s
                        AND (%s IS NULL OR operation_type = %s)
                    GROUP BY model
                    ORDER BY total_tokens DESC
                """
            
            cursor.execute(breakdown_query, (business_id, start_dt, end_dt, operation_type, operation_type))
            breakdown_rows = cursor.fetchall()
            
            for row in breakdown_rows:
                breakdown.append({
                    'period': str(row['period']),
                    'calls': row['calls'],
                    'input_tokens': row['input_tokens'],
                    'output_tokens': row['output_tokens'],
                    'total_tokens': row['total_tokens'],
                    'total_cost': float(row['total_cost'])
                })
            
            # 3. Breakdown por operation_type (siempre incluido)
            by_operation_query = """
                SELECT 
                    operation_type,
                    COUNT(*) as calls,
                    SUM(total_tokens) as tokens,
                    SUM(total_cost) as cost
                FROM ai.llm_calls
                WHERE business_id = %s
                    AND created_at >= %s
                    AND created_at < %s
                GROUP BY operation_type
                ORDER BY tokens DESC
            """
            
            cursor.execute(by_operation_query, (business_id, start_dt, end_dt))
            by_operation_rows = cursor.fetchall()
            
            by_operation = {}
            for row in by_operation_rows:
                by_operation[row['operation_type']] = {
                    'calls': row['calls'],
                    'tokens': row['tokens'],
                    'cost': float(row['cost'])
                }
            
            # 4. Breakdown por model (siempre incluido)
            by_model_query = """
                SELECT 
                    model,
                    COUNT(*) as calls,
                    SUM(total_tokens) as tokens,
                    SUM(total_cost) as cost
                FROM ai.llm_calls
                WHERE business_id = %s
                    AND created_at >= %s
                    AND created_at < %s
                GROUP BY model
                ORDER BY tokens DESC
            """
            
            cursor.execute(by_model_query, (business_id, start_dt, end_dt))
            by_model_rows = cursor.fetchall()
            
            by_model = {}
            for row in by_model_rows:
                by_model[row['model']] = {
                    'calls': row['calls'],
                    'tokens': row['tokens'],
                    'cost': float(row['cost'])
                }
            
            # Retornar respuesta completa
            return {
                'summary': summary,
                'breakdown': breakdown,
                'by_operation': by_operation,
                'by_model': by_model
            }
        
        except Exception as e:
            print(f"Error obteniendo analytics: {e}")
            raise HTTPException(status_code=500, detail=f"Error obteniendo analytics: {str(e)}")
        
        finally:
            cursor.close()

