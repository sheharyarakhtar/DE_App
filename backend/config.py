"""Configuration settings for the application."""

from pydantic_settings import BaseSettings
from functools import lru_cache
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
    
    @property
    def database_url(self) -> str:
        """Construct the database URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

