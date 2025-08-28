import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    app_name: str = "ML Training Pipeline"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Database Configuration
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "genealpha_training"
    db_user: str = "postgres"
    db_password: str = "your_password_here"
    
    # W&B Configuration
    wandb_project: str = "ml-training-pipeline"
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None
    
    # HuggingFace Configuration
    huggingface_token: Optional[str] = None
    hf_token: Optional[str] = None  # Alternative field name
    hf_username: Optional[str] = None
    
    # Storage Configuration
    models_dir: str = "./models"
    data_dir: str = "./data"
    logs_dir: str = "./logs"
    
    # Training Configuration
    max_concurrent_jobs: int = 4
    default_batch_size: int = 32
    default_epochs: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def database_url(self) -> str:
        """Construct database URL manually"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
settings = Settings()
