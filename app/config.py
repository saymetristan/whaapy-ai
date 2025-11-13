from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str
    
    # Auth
    ai_service_token: str
    
    # URLs
    backend_url: str = "https://api.whaapy.com"
    
    # LLM Provider
    openai_api_key: str
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
