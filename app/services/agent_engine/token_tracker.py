"""
Token tracking callback handler for LangChain/LangGraph
"""

from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class TokenTrackerCallback(BaseCallbackHandler):
    """Callback handler que captura tokens usados por el LLM"""
    
    def __init__(self):
        super().__init__()
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM ends running."""
        if response.llm_output and 'token_usage' in response.llm_output:
            usage = response.llm_output['token_usage']
            self.total_tokens += usage.get('total_tokens', 0)
            self.prompt_tokens += usage.get('prompt_tokens', 0)
            self.completion_tokens += usage.get('completion_tokens', 0)
    
    def get_total_tokens(self) -> int:
        """Retorna el total de tokens usados"""
        return self.total_tokens
    
    def reset(self) -> None:
        """Resetear contadores"""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

