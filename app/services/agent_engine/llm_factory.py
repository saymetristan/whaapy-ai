import os
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from openai import OpenAI


def is_gpt5_model(model: str) -> bool:
    """
    Verifica si un modelo soporta reasoning controls (GPT-5 family).
    
    Según la documentación de OpenAI:
    - gpt-5, gpt-5-mini, gpt-5-nano soportan reasoning.effort y text.verbosity
    - gpt-4o, gpt-4o-mini, gpt-4, gpt-3.5 NO soportan estos parámetros
    
    Los nombres de API son:
    - gpt-5 (system card: gpt-5-thinking)
    - gpt-5-mini (system card: gpt-5-thinking-mini)
    - gpt-5-nano (system card: gpt-5-thinking-nano)
    - gpt-5-chat-latest (system card: gpt-5-main)
    """
    gpt5_models = ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5-chat-latest']
    return any(model.startswith(m) for m in gpt5_models)


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
    
    @staticmethod
    def create_responses_client() -> OpenAI:
        """
        Crear cliente de OpenAI para Responses API.
        
        Responses API es la nueva API que reemplaza Chat Completions
        y soporta GPT-5 con reasoning controls.
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY no está configurada en variables de entorno")
        
        return OpenAI(api_key=api_key)
    
    @staticmethod
    async def call_gpt4o_mini(input_text: str, system_prompt: str = "") -> str:
        """
        Llamar a gpt-4o-mini para análisis rápido (intent classification, etc).
        
        Args:
            input_text: Texto a analizar
            system_prompt: System prompt opcional
        
        Returns:
            str: Respuesta del modelo
        """
        client = LLMFactory.create_responses_client()
        
        # Combinar system prompt con input si existe
        full_input = f"{system_prompt}\n\n{input_text}" if system_prompt else input_text
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": input_text})
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error llamando a gpt-4o-mini: {e}")
            raise
    
    @staticmethod
    async def call_gpt5_nano_minimal(input_text: str, system_prompt: str = "") -> str:
        """
        Llamar a gpt-5-nano con minimal reasoning para análisis rápido.
        
        Args:
            input_text: Texto a analizar
            system_prompt: System prompt opcional
        
        Returns:
            str: Respuesta del modelo
        """
        client = LLMFactory.create_responses_client()
        
        # Combinar system prompt con input si existe
        full_input = f"{system_prompt}\n\n{input_text}" if system_prompt else input_text
        
        try:
            response = client.responses.create(
                model="gpt-5-nano",
                input=full_input,
                reasoning={"effort": "minimal"},  # Mínimo razonamiento para velocidad
                text={"verbosity": "low"}  # Respuestas concisas
            )
            
            return response.output_text
            
        except Exception as e:
            print(f"Error llamando a gpt-5-nano: {e}")
            raise

