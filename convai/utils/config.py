from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings."""

    # API Configuration
    API_TITLE: str = "Conversation AI"
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: float = 8000
    
    # Project Infomation
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    PROMPTS_DIR: Path = PROJECT_ROOT / "convai" / "prompts"

    # Logging Configuration
    LOG_LEVEL: str = "debug"
    LOG_FILE: Optional[str] = None
    LOG_FILE_LEVEL: Optional[str] = None
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./movielens.db"
    
    # Dataset Url
    MOVIELENS_DOWNLOAD_URL: str = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

    # LLM Information
    MODEL_PROVIDER: str = "ollama"
    MODEL_NAME: str = "qwen3:8b"
    MODEL_TEMPERATURE: float = 0.0
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
