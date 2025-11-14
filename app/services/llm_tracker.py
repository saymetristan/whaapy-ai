"""
LLM Call Tracker - Context manager para trackear llamadas a LLMs automáticamente.

Uso:
    async with LLMCallTracker(
        business_id="xxx",
        operation_type="chat",
        provider="openai",
        model="gpt-5-mini",
        execution_id="yyy"  # opcional
    ) as tracker:
        response = await llm.call()
        tracker.record(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens
        )
"""

import time
import json
from typing import Dict, Any, Optional
from app.services.pricing import calculate_cost
from app.db.database import get_db


class LLMCallTracker:
    """
    Context manager para trackear llamadas a LLMs automáticamente.
    
    Guarda cada llamada en ai.llm_calls con tokens, costos y metadata.
    """
    
    def __init__(
        self,
        business_id: str,
        operation_type: str,
        provider: str,
        model: str,
        execution_id: Optional[str] = None,
        operation_context: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None
    ):
        """
        Inicializar tracker.
        
        Args:
            business_id: ID del negocio
            operation_type: Tipo de operación ('chat', 'embedding', 'ocr', etc)
            provider: Proveedor del LLM ('openai', 'groq')
            model: Modelo usado ('gpt-5-mini', 'text-embedding-3-small', etc)
            execution_id: ID de la ejecución del agente (opcional)
            operation_context: Contexto adicional como {conversation_id, document_id}
            reasoning_effort: Esfuerzo de razonamiento para GPT-5 ('minimal', 'low', 'medium', 'extended')
        """
        self.business_id = business_id
        self.operation_type = operation_type
        self.provider = provider
        self.model = model
        self.execution_id = execution_id
        self.operation_context = operation_context or {}
        self.reasoning_effort = reasoning_effort
        
        self.start_time = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self.cache_hit = False
        self.error = None
        
    async def __aenter__(self):
        """Entrada al context manager - inicia el timer"""
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Salida del context manager - calcula duración y guarda en DB.
        
        Se ejecuta automáticamente al salir del bloque async with.
        """
        duration_ms = int((time.time() - self.start_time) * 1000)
        
        # Si hubo error, capturarlo
        if exc_type:
            self.error = str(exc_val)
        
        # Calcular costos
        costs = calculate_cost(
            self.model, 
            self.input_tokens, 
            self.output_tokens,
            self.cached_tokens
        )
        
        # Guardar en DB
        await self._save_to_db(duration_ms, costs)
        
        # No suprimir excepciones
        return False
    
    def record(
        self, 
        input_tokens: int, 
        output_tokens: int,
        cached_tokens: int = 0,
        cache_hit: bool = False
    ):
        """
        Registrar tokens usados por la llamada a LLM.
        
        Args:
            input_tokens: Tokens de input (no cacheados)
            output_tokens: Tokens de output
            cached_tokens: Tokens de input que vinieron del cache (opcional)
            cache_hit: Si se usó prompt caching (opcional)
        """
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_tokens = cached_tokens
        self.cache_hit = cache_hit
    
    async def _save_to_db(self, duration_ms: int, costs: Dict[str, float]):
        """
        Guardar registro en ai.llm_calls.
        
        Args:
            duration_ms: Duración de la llamada en milisegundos
            costs: Dict con input_cost, output_cost, cached_cost, total_cost
        """
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO ai.llm_calls (
                        business_id, execution_id, operation_type, operation_context,
                        provider, model, input_tokens, output_tokens, total_tokens, cached_tokens,
                        input_cost, output_cost, cached_cost, total_cost, 
                        duration_ms, reasoning_effort, cache_hit, error
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.business_id,
                    self.execution_id,
                    self.operation_type,
                    json.dumps(self.operation_context),
                    self.provider,
                    self.model,
                    self.input_tokens,
                    self.output_tokens,
                    self.input_tokens + self.output_tokens,
                    self.cached_tokens,
                    costs['input_cost'],
                    costs['output_cost'],
                    costs['cached_cost'],
                    costs['total_cost'],
                    duration_ms,
                    self.reasoning_effort,
                    self.cache_hit,
                    self.error
                ))
                
                conn.commit()
                cursor.close()
                
                # Log success
                status = "❌" if self.error else "✅"
                print(f"{status} LLM call tracked: {self.model} - {self.input_tokens + self.output_tokens} tokens, ${costs['total_cost']:.6f} ({duration_ms}ms)")
                
        except Exception as e:
            print(f"❌ Error guardando LLM call tracking: {e}")
            # No lanzar error para no interrumpir flujo principal


# Helper para estimar tokens de embeddings
def estimate_embedding_tokens(text: str) -> int:
    """
    Estimar tokens para embeddings.
    
    OpenAI embeddings usan aproximadamente 1 token por cada 4 caracteres.
    Esta es una estimación conservadora.
    
    Args:
        text: Texto a embeddear
    
    Returns:
        Número estimado de tokens
    """
    return max(1, len(text) // 4)

