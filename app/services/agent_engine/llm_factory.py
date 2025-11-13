import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


class LLMConfig:
    """Configuración para crear un LLM"""
    def __init__(
        self,
        provider: str = 'openai',
        model: str = 'gpt-5-mini',
        temperature: float = 0.2,
        max_tokens: int = 2000
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens


class LLMFactory:
    """Factory para crear instancias de LLM según configuración"""
    
    @staticmethod
    def create_llm(config: LLMConfig) -> BaseChatModel:
        """
        Crear LLM según configuración.
        
        Por ahora solo soporta OpenAI. Anthropic y OpenRouter en Fase 2.
        """
        if config.provider != 'openai':
            raise ValueError(f"Provider no soportado: {config.provider}. Solo 'openai' está disponible.")
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY no está configurada en variables de entorno")
        
        # gpt-5-mini solo acepta temperature=1 (default)
        # otros modelos pueden tener temperature custom
        llm_kwargs = {
            'api_key': api_key,
            'model': config.model,
            'max_completion_tokens': config.max_tokens
        }
        
        # Solo agregar temperature si NO es gpt-5-mini
        if 'gpt-5-mini' not in config.model:
            llm_kwargs['temperature'] = config.temperature
        
        return ChatOpenAI(**llm_kwargs)
    
    @staticmethod
    def create_from_dict(config_dict: Dict[str, Any]) -> BaseChatModel:
        """Crear LLM desde un diccionario de configuración"""
        config = LLMConfig(
            provider=config_dict.get('provider', 'openai'),
            model=config_dict.get('model', 'gpt-5-mini'),
            temperature=config_dict.get('temperature', 0.2),
            max_tokens=config_dict.get('max_tokens', 2000)
        )
        return LLMFactory.create_llm(config)
    
    @staticmethod
    def create_default() -> BaseChatModel:
        """Crear LLM con configuración por defecto (gpt-5-mini)"""
        config = LLMConfig()
        return LLMFactory.create_llm(config)
    
    @staticmethod
    def create_fast() -> BaseChatModel:
        """Crear LLM rápido para análisis (gpt-5-mini con temperatura baja)"""
        config = LLMConfig(
            model='gpt-5-mini',
            temperature=0.0,
            max_tokens=500
        )
        return LLMFactory.create_llm(config)

