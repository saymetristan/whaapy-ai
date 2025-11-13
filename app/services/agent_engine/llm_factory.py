import os
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult


class SafeChatOpenAI(ChatOpenAI):
    """
    Wrapper de ChatOpenAI que NUNCA pasa temperature.
    Los modelos nuevos de OpenAI (gpt-5-mini, gpt-4o) no aceptan temperature custom.
    """
    
    def __init__(self, **kwargs):
        # Remover temperature si existe en kwargs
        kwargs.pop('temperature', None)
        super().__init__(**kwargs)
    
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Override para asegurar que temperature nunca se envía"""
        params = super()._default_params
        params.pop('temperature', None)
        return params


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
        
        # Los modelos de OpenAI nuevos no aceptan temperature custom
        # Usar SafeChatOpenAI que elimina temperature automáticamente
        return SafeChatOpenAI(
            api_key=api_key,
            model=config.model,
            max_completion_tokens=config.max_tokens
        )
    
    @staticmethod
    def create_from_dict(config_dict: Dict[str, Any]) -> BaseChatModel:
        """Crear LLM desde un diccionario de configuración"""
        # Ignorar temperature del config - OpenAI nuevos modelos no lo aceptan
        config = LLMConfig(
            provider=config_dict.get('provider', 'openai'),
            model=config_dict.get('model', 'gpt-5-mini'),
            temperature=1.0,  # Usar default de OpenAI
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
        """Crear LLM rápido para análisis (gpt-5-mini)"""
        config = LLMConfig(
            model='gpt-5-mini',
            temperature=1.0,  # Usar default de OpenAI
            max_tokens=500
        )
        return LLMFactory.create_llm(config)

