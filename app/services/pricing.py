"""
Pricing service para calcular costos de llamadas a LLMs.

Pricing actualizado según:
- OpenAI: https://openai.com/api/pricing/ (Standard Tier)
- Groq: https://console.groq.com/docs/models

Última actualización: Enero 2025
"""

from typing import Dict, Optional


# Pricing por millón de tokens (Standard Tier para OpenAI)
# Formato: {model: {"input": precio_por_1M, "output": precio_por_1M, "cached_input": precio_por_1M}}
PRICING = {
    # OpenAI GPT-5 (Standard Tier)
    'gpt-5.1': {'input': 1.25, 'output': 10.00, 'cached_input': 0.125},
    'gpt-5': {'input': 1.25, 'output': 10.00, 'cached_input': 0.125},
    'gpt-5-mini': {'input': 0.25, 'output': 2.00, 'cached_input': 0.025},
    'gpt-5-nano': {'input': 0.05, 'output': 0.40, 'cached_input': 0.005},
    'gpt-5-chat-latest': {'input': 1.25, 'output': 10.00, 'cached_input': 0.125},
    'gpt-5.1-chat-latest': {'input': 1.25, 'output': 10.00, 'cached_input': 0.125},
    'gpt-5-codex': {'input': 1.25, 'output': 10.00, 'cached_input': 0.125},
    'gpt-5.1-codex': {'input': 1.25, 'output': 10.00, 'cached_input': 0.125},
    'gpt-5-pro': {'input': 15.00, 'output': 120.00, 'cached_input': None},
    
    # OpenAI GPT-4.1 (Standard Tier)
    'gpt-4.1': {'input': 2.00, 'output': 8.00, 'cached_input': 0.50},
    'gpt-4.1-mini': {'input': 0.40, 'output': 1.60, 'cached_input': 0.10},
    'gpt-4.1-nano': {'input': 0.10, 'output': 0.40, 'cached_input': 0.025},
    
    # OpenAI GPT-4o (Standard Tier)
    'gpt-4o': {'input': 2.50, 'output': 10.00, 'cached_input': 1.25},
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60, 'cached_input': 0.075},
    'gpt-4o-2024-05-13': {'input': 5.00, 'output': 15.00, 'cached_input': None},
    'gpt-4o-2024-08-06': {'input': 2.50, 'output': 10.00, 'cached_input': 1.25},
    
    # OpenAI Reasoning Models (Standard Tier)
    'o1': {'input': 15.00, 'output': 60.00, 'cached_input': 7.50},
    'o1-pro': {'input': 150.00, 'output': 600.00, 'cached_input': None},
    'o3': {'input': 2.00, 'output': 8.00, 'cached_input': 0.50},
    'o3-pro': {'input': 20.00, 'output': 80.00, 'cached_input': None},
    'o3-deep-research': {'input': 10.00, 'output': 40.00, 'cached_input': 2.50},
    'o4-mini': {'input': 1.10, 'output': 4.40, 'cached_input': 0.275},
    'o4-mini-deep-research': {'input': 2.00, 'output': 8.00, 'cached_input': 0.50},
    'o3-mini': {'input': 1.10, 'output': 4.40, 'cached_input': 0.55},
    'o1-mini': {'input': 1.10, 'output': 4.40, 'cached_input': 0.55},
    
    # OpenAI Embeddings (Standard Tier)
    'text-embedding-3-small': {'input': 0.02, 'output': 0.0, 'cached_input': None},
    'text-embedding-3-large': {'input': 0.13, 'output': 0.0, 'cached_input': None},
    'text-embedding-ada-002': {'input': 0.10, 'output': 0.0, 'cached_input': None},
    
    # Groq Models (pricing según https://console.groq.com/)
    'openai/gpt-oss-120b': {'input': 0.15, 'output': 0.60, 'cached_input': 0.075},
    
    # Legacy OpenAI models (Standard Tier)
    'gpt-4-turbo-2024-04-09': {'input': 10.00, 'output': 30.00, 'cached_input': None},
    'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50, 'cached_input': None},
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0
) -> Dict[str, float]:
    """
    Calcular costos de una llamada a LLM con soporte para prompt caching.
    
    Args:
        model: Nombre del modelo (ej: 'gpt-5-mini', 'openai/gpt-oss-120b')
        input_tokens: Tokens de input (no cacheados)
        output_tokens: Tokens de output
        cached_tokens: Tokens de input que vienen del cache (más baratos)
    
    Returns:
        {
            'input_cost': float,      # Costo de tokens de input
            'output_cost': float,     # Costo de tokens de output
            'cached_cost': float,     # Costo de tokens cacheados
            'total_cost': float       # Costo total
        }
    
    Example:
        >>> calculate_cost('gpt-5-mini', input_tokens=1000, output_tokens=500)
        {'input_cost': 0.00025, 'output_cost': 0.001, 'cached_cost': 0.0, 'total_cost': 0.00125}
        
        >>> calculate_cost('gpt-5-mini', input_tokens=1000, output_tokens=500, cached_tokens=2000)
        {'input_cost': 0.00025, 'output_cost': 0.001, 'cached_cost': 0.00005, 'total_cost': 0.0013}
    """
    # Default a gpt-5-mini si modelo no existe
    pricing = PRICING.get(model, {'input': 0.25, 'output': 2.00, 'cached_input': 0.025})
    
    # Calcular costos (precio está por millón de tokens)
    input_cost = (input_tokens / 1_000_000) * pricing['input']
    output_cost = (output_tokens / 1_000_000) * pricing['output']
    
    # Cached input es más barato (si el modelo lo soporta)
    cached_cost = 0.0
    if cached_tokens > 0 and pricing.get('cached_input'):
        cached_cost = (cached_tokens / 1_000_000) * pricing['cached_input']
    
    return {
        'input_cost': round(input_cost, 8),
        'output_cost': round(output_cost, 8),
        'cached_cost': round(cached_cost, 8),
        'total_cost': round(input_cost + output_cost + cached_cost, 8)
    }


def get_model_pricing(model: str) -> Optional[Dict[str, float]]:
    """
    Obtener pricing de un modelo específico.
    
    Args:
        model: Nombre del modelo
    
    Returns:
        Dict con 'input', 'output', 'cached_input' (puede ser None) o None si no existe
    
    Example:
        >>> get_model_pricing('gpt-5-mini')
        {'input': 0.25, 'output': 2.00, 'cached_input': 0.025}
    """
    return PRICING.get(model)


def list_supported_models() -> list[str]:
    """
    Listar todos los modelos con pricing configurado.
    
    Returns:
        Lista de nombres de modelos
    """
    return list(PRICING.keys())

