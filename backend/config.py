"""Configuration settings for the application."""

from pydantic_settings import BaseSettings
from pydantic import BaseModel, field_validator
from functools import lru_cache
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # PostgreSQL Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "appuser"
    postgres_password: str = "apppassword"
    postgres_db: str = "appdb"
    
    # OpenAI Configuration
    openai_api_key: str = ""
    
    # Application Configuration
    watch_folder: str = "./watch_folder"
    
    @field_validator('postgres_host', mode='before')
    @classmethod
    def default_host(cls, v):
        return v if v else "localhost"
    
    @field_validator('postgres_port', mode='before')
    @classmethod
    def default_port(cls, v):
        if v is None or v == '':
            return 5432
        return int(v)
    
    @field_validator('postgres_user', mode='before')
    @classmethod
    def default_user(cls, v):
        return v if v else "appuser"
    
    @field_validator('postgres_password', mode='before')
    @classmethod
    def default_password(cls, v):
        return v if v else "apppassword"
    
    @field_validator('postgres_db', mode='before')
    @classmethod
    def default_db(cls, v):
        return v if v else "appdb"
    
    @property
    def database_url(self) -> str:
        """Construct the database URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class RuntimeConfig:
    """
    Runtime configuration that can be updated via API.
    Overrides Settings when values are set.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._db_config = None
            cls._instance._openai_key = None
        return cls._instance
    
    def set_db_config(self, host: str, port: int, user: str, password: str, database: str):
        """Set database configuration at runtime."""
        self._db_config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database
        }
    
    def set_openai_key(self, api_key: str):
        """Set OpenAI API key at runtime."""
        self._openai_key = api_key
    
    def clear_db_config(self):
        """Clear runtime database configuration."""
        self._db_config = None
    
    def clear_openai_key(self):
        """Clear runtime OpenAI API key."""
        self._openai_key = None
    
    @property
    def db_config(self) -> Optional[dict]:
        """Get runtime database config if set."""
        return self._db_config
    
    @property
    def openai_key(self) -> Optional[str]:
        """Get runtime OpenAI key if set."""
        return self._openai_key
    
    def get_effective_db_config(self, settings: Settings) -> dict:
        """Get effective DB config (runtime override or settings)."""
        if self._db_config:
            return self._db_config
        return {
            "host": settings.postgres_host,
            "port": settings.postgres_port,
            "user": settings.postgres_user,
            "password": settings.postgres_password,
            "database": settings.postgres_db
        }
    
    def get_effective_openai_key(self, settings: Settings) -> str:
        """Get effective OpenAI key (runtime override or settings)."""
        if self._openai_key:
            return self._openai_key
        return settings.openai_api_key


# Pydantic models for API requests
class DBConfigRequest(BaseModel):
    """Request model for database configuration."""
    host: str
    port: int = 5432
    user: str
    password: str
    database: str


class OpenAIConfigRequest(BaseModel):
    """Request model for OpenAI configuration."""
    api_key: str


class ConfigStatusResponse(BaseModel):
    """Response model for configuration status."""
    db_configured: bool
    db_connected: bool
    db_source: str  # "runtime" or "env" or "default"
    openai_configured: bool
    openai_source: str  # "runtime" or "env" or "none"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_runtime_config() -> RuntimeConfig:
    """Get runtime configuration singleton."""
    return RuntimeConfig()
